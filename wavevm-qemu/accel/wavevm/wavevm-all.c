
#include "qemu/osdep.h"
#include "qemu/module.h"
#include "qapi/error.h"
#include "qapi/qapi-builtin-visit.h"
#include "sysemu/accel.h"
#include "sysemu/cpus.h"
#include "sysemu/sysemu.h"
#include "hw/boards.h"
#include "qemu/option.h"
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/socket.h> 
#include <netinet/in.h>
#include <sys/un.h>
#include <errno.h>
#include "qemu/thread.h"
#include "sysemu/kvm.h" 
#include "sysemu/kvm_int.h"
#include "linux/kvm.h"
#include "exec/cpu-common.h"
#include "exec/address-spaces.h"
#include "exec/exec-all.h"
#include <sys/socket.h>
#include <sys/un.h>
#include <poll.h>
#include "../../../common_include/wavevm_protocol.h"
#include "../../../common_include/wavevm_ioctl.h"
#include "wavevm-accel.h"

// 引用相关模块
extern void wavevm_user_mem_init(void *ram_ptr, size_t ram_size);
extern void wavevm_setup_memory_region(MemoryRegion *mr, uint64_t size, int fd);
extern void wvm_tcg_get_state(CPUState *cpu, wvm_tcg_context_t *ctx);
extern void wvm_tcg_set_state(CPUState *cpu, wvm_tcg_context_t *ctx);
extern void wvm_set_ttl_interval(int ms);
extern void wvm_register_volatile_ram(uint64_t gpa, uint64_t size);
extern void wvm_apply_remote_push(uint16_t msg_type, void *payload);
extern void wavevm_start_vcpu_thread(CPUState *cpu);

int g_wvm_local_split = 0;
static bool g_wvm_split_explicit = false;
static bool g_wvm_mode_explicit = false;

static int wavevm_auto_split_from_vcpus(int vcpus)
{
    if (vcpus <= 1) {
        return 1;
    }
    if (vcpus == 2) {
        return 1;
    }
    return vcpus / 2;
}

static void wavevm_resolve_split(MachineState *ms)
{
    int vcpus = 1;
    const char *env_split;

    if (ms && ms->smp.cpus > 0) {
        vcpus = ms->smp.cpus;
    }

    if (!g_wvm_split_explicit) {
        env_split = getenv("WVM_LOCAL_SPLIT");
        if (env_split && *env_split) {
            char *endptr = NULL;
            long parsed = strtol(env_split, &endptr, 10);
            if (endptr && *endptr == '\0' && parsed >= 0) {
                g_wvm_local_split = (int)parsed;
            } else {
                g_wvm_local_split = wavevm_auto_split_from_vcpus(vcpus);
            }
        } else {
            g_wvm_local_split = wavevm_auto_split_from_vcpus(vcpus);
        }
    }

    if (g_wvm_local_split < 0) {
        g_wvm_local_split = 0;
    }
    if (g_wvm_local_split > vcpus) {
        g_wvm_local_split = vcpus;
    }
}

void wavevm_apply_split_hint(int split)
{
    int vcpus = 1;

    if (g_wvm_split_explicit || split < 0) {
        return;
    }

    if (current_machine && current_machine->smp.cpus > 0) {
        vcpus = current_machine->smp.cpus;
    }

    g_wvm_local_split = split;
    if (g_wvm_local_split > vcpus) {
        g_wvm_local_split = vcpus;
    }
}

/* [V28 FIX] 坚如磐石的读取函数，处理 Partial Read 和 EINTR */
static int read_exact(int fd, void *buf, size_t len) {
    size_t received = 0;
    char *ptr = (char *)buf;
    
    while (received < len) {
        ssize_t ret = read(fd, ptr + received, len - received);
        
        if (ret > 0) {
            received += ret;
        } else if (ret == 0) {
            return -1; // EOF: 对端挂了
        } else {
            if (errno == EINTR) continue; // 信号中断，重试
            return -1; // 真正的错误
        }
    }
    return 0; // 成功读满
}

static bool wavevm_allowed = true;

#define SYNC_WINDOW_SIZE 64

static int connect_to_master_helper(void) {
    int sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock < 0) return -1;
    struct sockaddr_un addr = { .sun_family = AF_UNIX };
    
    // [V28 Fix] Dynamic Path
    const char *env_path = getenv("WVM_ENV_SOCK_PATH");
    if (!env_path) {
        char *inst_id = getenv("WVM_INSTANCE_ID");
        static char fallback[128]; // static to be safe scope-wise though redundant here
        snprintf(fallback, sizeof(fallback), "/tmp/wvm_user_%s.sock", inst_id ? inst_id : "0");
        env_path = fallback;
    }

    strncpy(addr.sun_path, env_path, sizeof(addr.sun_path) - 1);
    
    if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        close(sock);
        return -1;
    }
    return sock;
}

// Master Mode B (User) 的 IPC 监听线程
static void *wavevm_master_ipc_thread(void *arg) {
    WaveVMAccelState *s = (WaveVMAccelState *)arg;
    
    s->ipc_sock = connect_to_master_helper();
    if (s->ipc_sock < 0) {
        fprintf(stderr, "[WVM] Failed to connect IPC socket for IRQ listening.\n");
        return NULL;
    }

    struct wvm_ipc_header_t hdr;
    // 缓冲区用于接收 payload (最大可能的消息体)
    uint8_t payload_buf[4096]; 

    while (s->sync_thread_running) {
        // [V28 FIX] 使用 read_exact 替代原始 read
        if (read_exact(s->ipc_sock, &hdr, sizeof(hdr)) < 0) {
            // 连接断开或错误，尝试简单的重连或退出
            g_usleep(100000); 
            close(s->ipc_sock);
            s->ipc_sock = connect_to_master_helper();
            if (s->ipc_sock < 0) return NULL; // 重连失败则退出
            continue;
        }

        // [V28 FIX] 完整读取 Payload，防止粘包错位
        if (hdr.len > 0) {
            if (hdr.len > sizeof(payload_buf)) {
                fprintf(stderr, "[WVM] IPC Payload too large: %d\n", hdr.len);
                // 严重协议错误，无法恢复同步，必须断开
                close(s->ipc_sock);
                return NULL;
            }
            if (read_exact(s->ipc_sock, payload_buf, hdr.len) < 0) {
                continue; // 读取 Payload 失败
            }
        }

        // 1. 处理中断消息
        if (hdr.type == WVM_IPC_TYPE_IRQ) {
            qemu_mutex_lock_iothread();
            if (kvm_enabled()) {
                 struct kvm_irq_level irq;
                 irq.irq = 16; 
                 irq.level = 1;
                 kvm_vm_ioctl(kvm_state, KVM_IRQ_LINE, &irq);
                 irq.level = 0;
                 kvm_vm_ioctl(kvm_state, KVM_IRQ_LINE, &irq);
            }
            qemu_mutex_unlock_iothread();
        }
        // 2. 处理内存失效 (MESI Invalidate)
        else if (hdr.type == WVM_IPC_TYPE_MEM_WRITE) {
            struct wvm_ipc_write_req *req = (struct wvm_ipc_write_req *)payload_buf;
            // 约定：len=0 表示 Invalidate
            if (req->len == 0) {
                hwaddr len = 4096;
                void *host_addr = cpu_physical_memory_map(req->gpa, &len, 1);
                if (host_addr && len >= 4096) {
                    mprotect(host_addr, 4096, PROT_NONE);
                    cpu_physical_memory_unmap(host_addr, len, 1, 0);
                }
            }
        }
        else if (hdr.type == WVM_IPC_TYPE_INVALIDATE) { // Type 6
            // 解包内部的 wvm_header
            struct wvm_header *net_hdr = (struct wvm_header *)payload_buf;
            void *net_payload = payload_buf + sizeof(struct wvm_header);
            uint16_t msg_type = ntohs(net_hdr->msg_type);
            // 处理 Prophet 同步指令
            if (msg_type == MSG_RPC_BATCH_MEMSET) {
                // 1. TCG 模式下最关键的动作：冲刷所有翻译块 (Translation Blocks)
                // 必须在 BQL 锁保护下执行
                qemu_mutex_lock_iothread();
                
                // 冲刷所有缓存的代码。虽然这很重，但对于批量 memset 来说，
                // 它是保证指令一致性的唯一可靠手段。
                tb_flush(first_cpu); 
                
                // 2. 同时也需要更新 User-Mode 的版本号 (调用 user-mem.c)
                wvm_apply_remote_push(msg_type, net_payload);
                
                qemu_mutex_unlock_iothread();
                continue;
            }
            
            // 调用 user-mem 提供的逻辑应用更新
            wvm_apply_remote_push(msg_type, net_payload);
        }
    }
    close(s->ipc_sock);
    return NULL;
}

// Master 模式专用脏页同步线程 (仅 KVM 需此线程，TCG 由 user-mem 处理)
static void *wavevm_dirty_sync_thread(void *arg) {
    WaveVMAccelState *s = (WaveVMAccelState *)arg;
    if (!kvm_enabled()) {
        return NULL;
    }

    s->sync_sock = connect_to_master_helper();
    if (s->sync_sock < 0) {
        return NULL;
    }

    /* Keep the thread alive to preserve timing/ordering behavior.
     * Dirty-log harvesting is intentionally disabled in this compatibility path.
     */
    while (s->sync_thread_running) {
        g_usleep(50000);
    }

    close(s->sync_sock);
    return NULL;
}

// Kernel Mode IRQ 监听线程
static void *wavevm_kernel_irq_thread(void *arg) {
    WaveVMAccelState *s = (WaveVMAccelState *)arg;
    uint32_t irq_num;
    
    while (1) {
        // 陷入内核等待，直到收到 UDP 中断包
        if (ioctl(s->dev_fd, IOCTL_WAIT_IRQ, &irq_num) == 0) {
            qemu_mutex_lock_iothread();
            // 注入中断 (Pulse)
            if (kvm_enabled()) {
                 struct kvm_irq_level irq;
                 irq.irq = irq_num; // 例如 16
                 irq.level = 1;
                 kvm_vm_ioctl(kvm_state, KVM_IRQ_LINE, &irq);
                 irq.level = 0;
                 kvm_vm_ioctl(kvm_state, KVM_IRQ_LINE, &irq);
            }
            qemu_mutex_unlock_iothread();
        } else {
            // 出错或退出
            break;
        }
    }
    return NULL;
}

// Slave 网络处理线程 (支持 KVM 和 TCG 双模)
static void *wavevm_slave_net_thread(void *arg) {
    WaveVMAccelState *s = (WaveVMAccelState *)arg;
    CPUState *cpu = first_cpu; 
    #define BATCH_SIZE 64
    #define MAX_PKT_SIZE 4096 

    struct mmsghdr msgs[BATCH_SIZE];
    struct iovec iovecs[BATCH_SIZE];
    uint8_t *buffers = g_malloc(BATCH_SIZE * MAX_PKT_SIZE);
    struct sockaddr_in addrs[BATCH_SIZE];

    memset(msgs, 0, sizeof(msgs));
    for (int i = 0; i < BATCH_SIZE; i++) {
        iovecs[i].iov_base = buffers + i * MAX_PKT_SIZE;
        iovecs[i].iov_len = MAX_PKT_SIZE;
        msgs[i].msg_hdr.msg_iov = &iovecs[i];
        msgs[i].msg_hdr.msg_iovlen = 1;
        msgs[i].msg_hdr.msg_name = &addrs[i];
        msgs[i].msg_hdr.msg_namelen = sizeof(struct sockaddr_in);
    }

    printf("[WaveVM-Slave] Network Loop Active (Engine: %s, FD: %d).\n", 
           kvm_enabled() ? "KVM" : "TCG", s->master_sock);

    while (1) {
        int retval = recvmmsg(s->master_sock, msgs, BATCH_SIZE, 0, NULL);
        if (retval < 0) {
            if (errno == EINTR) continue;
            perror("recvmmsg");
            break;
        }

        for (int i = 0; i < retval; i++) {
            uint8_t *buf = (uint8_t *)iovecs[i].iov_base;
            int len = msgs[i].msg_len;
            
            if (len >= sizeof(struct wvm_header)) {
                struct wvm_header *hdr = (struct wvm_header *)buf;
                void *payload = buf + sizeof(struct wvm_header);

                // 1. 内存写 (Master -> Slave)
                if (hdr->msg_type == MSG_MEM_WRITE) {
                    qemu_mutex_lock_iothread(); 
                    if (hdr->payload_len > 8) {
                        uint64_t gpa = *(uint64_t *)payload;
                        uint32_t data_len = hdr->payload_len - 8;
                        void *data_ptr = (uint8_t *)payload + 8;
                        cpu_physical_memory_write(gpa, data_ptr, data_len);
                    }
                    qemu_mutex_unlock_iothread();
                    hdr->msg_type = MSG_MEM_ACK;
                    hdr->payload_len = 0;
                    sendto(s->master_sock, buf, sizeof(struct wvm_header), 0, 
                          (struct sockaddr *)&addrs[i], sizeof(struct sockaddr_in));
                }
                // 2. 内存读 (Master -> Slave)
                else if (hdr->msg_type == MSG_MEM_READ) {
                    uint64_t gpa = *(uint64_t *)payload;
                    uint32_t read_len = 4096; 
                    if (sizeof(struct wvm_header) + read_len <= MAX_PKT_SIZE) {
                        cpu_physical_memory_read(gpa, payload, read_len);
                        hdr->msg_type = MSG_MEM_ACK;
                        hdr->payload_len = read_len;
                        sendto(s->master_sock, buf, sizeof(struct wvm_header) + read_len, 0, 
                              (struct sockaddr *)&addrs[i], sizeof(struct sockaddr_in));
                    }
                }
                // 3. 远程执行 (Master -> Slave)
                else if (hdr->msg_type == MSG_VCPU_RUN) {
                    struct wvm_ipc_cpu_run_req *req = (struct wvm_ipc_cpu_run_req *)payload;
                    qemu_mutex_lock_iothread(); // TCG 必须持有 BQL
                    
                    // A. 恢复上下文
                    if (hdr->mode_tcg) {
                        // TCG 模式：直接写入 env
                        wvm_tcg_set_state(cpu, &req->ctx.tcg);
                    } else if (kvm_enabled()) {
                        // KVM 模式：ioctl 设置
                        struct kvm_regs kregs; struct kvm_sregs ksregs;
                        wvm_kvm_context_t *kctx = &req->ctx.kvm;
                        kregs.rax = kctx->rax; kregs.rbx = kctx->rbx; kregs.rcx = kctx->rcx;
                        kregs.rdx = kctx->rdx; kregs.rsi = kctx->rsi; kregs.rdi = kctx->rdi;
                        kregs.rsp = kctx->rsp; kregs.rbp = kctx->rbp;
                        kregs.r8  = kctx->r8;  kregs.r9  = kctx->r9;  kregs.r10 = kctx->r10;
                        kregs.r11 = kctx->r11; kregs.r12 = kctx->r12; kregs.r13 = kctx->r13;
                        kregs.r14 = kctx->r14; kregs.r15 = kctx->r15;
                        kregs.rip = kctx->rip; kregs.rflags = kctx->rflags;
                        memcpy(&ksregs, kctx->sregs_data, sizeof(ksregs));
                        kvm_vcpu_ioctl(cpu, KVM_SET_SREGS, &ksregs);
                        kvm_vcpu_ioctl(cpu, KVM_SET_REGS, &kregs);
                    }

                    // B. 执行循环
                    cpu->stop = false; cpu->halted = 0; cpu->exception_index = -1;
                    
                    if (hdr->mode_tcg) {
                        // [TCG 关键] 使用 cpu_exec 运行直到退出/中断/异常
                        cpu_exec(cpu);
                    } else if (kvm_enabled()) {
                        // [KVM 关键]
                        int ret;
                        do {
                            ret = kvm_vcpu_ioctl(cpu, KVM_RUN, 0);
                            if (ret == 0) {
                                int reason = cpu->kvm_run->exit_reason;
                                if (reason == KVM_EXIT_IO || reason == KVM_EXIT_MMIO || 
                                    reason == KVM_EXIT_HLT || reason == KVM_EXIT_SHUTDOWN ||
                                    reason == KVM_EXIT_FAIL_ENTRY) break;
                            }
                        } while (ret > 0 || ret == -EINTR);
                    }

                    // C. 导出上下文并回包
                    struct wvm_ipc_cpu_run_ack *ack = (struct wvm_ipc_cpu_run_ack *)payload;
                    if (hdr->mode_tcg) {
                        ack->mode_tcg = 1;
                        wvm_tcg_get_state(cpu, &ack->ctx.tcg);
                    } else if (kvm_enabled()) {
                        ack->mode_tcg = 0;
                        struct kvm_regs kregs; struct kvm_sregs ksregs;
                        wvm_kvm_context_t *kctx = &ack->ctx.kvm;
                        kvm_vcpu_ioctl(cpu, KVM_GET_REGS, &kregs);
                        kvm_vcpu_ioctl(cpu, KVM_GET_SREGS, &ksregs);
                        kctx->rax = kregs.rax; kctx->rbx = kregs.rbx; kctx->rcx = kregs.rcx;
                        kctx->rdx = kregs.rdx; kctx->rsi = kregs.rsi; kctx->rdi = kregs.rdi;
                        kctx->rsp = kregs.rsp; kctx->rbp = kregs.rbp;
                        kctx->r8  = kregs.r8;  kctx->r9  = kregs.r9;  kctx->r10 = kregs.r10;
                        kctx->r11 = kregs.r11; kctx->r12 = kregs.r12; kctx->r13 = kregs.r13;
                        kctx->r14 = kregs.r14; kctx->r15 = kregs.r15;
                        kctx->rip = kregs.rip; kctx->rflags = kregs.rflags;
                        memcpy(kctx->sregs_data, &ksregs, sizeof(ksregs));
                        
                        struct kvm_run *run = cpu->kvm_run;
                        kctx->exit_reason = run->exit_reason;
                        if (run->exit_reason == KVM_EXIT_IO) {
                            kctx->io.direction = run->io.direction;
                            kctx->io.size      = run->io.size;
                            kctx->io.port      = run->io.port;
                            kctx->io.count     = run->io.count;
                            if (run->io.direction == KVM_EXIT_IO_OUT) {
                                size_t io_bytes = run->io.size * run->io.count;
                                if (io_bytes > sizeof(kctx->io.data)) {
                                    io_bytes = sizeof(kctx->io.data);
                                }
                                memcpy(kctx->io.data,
                                       (uint8_t *)run + run->io.data_offset,
                                       io_bytes);
                            }
                        } else if (run->exit_reason == KVM_EXIT_MMIO) {
                            kctx->mmio.phys_addr = run->mmio.phys_addr;
                            kctx->mmio.len       = run->mmio.len;
                            kctx->mmio.is_write  = run->mmio.is_write;
                            memcpy(kctx->mmio.data, run->mmio.data, 8);
                        }
                    }
                    qemu_mutex_unlock_iothread();
                    sendto(s->master_sock, buf, msgs[i].msg_len, 0, 
                          (struct sockaddr *)&addrs[i], sizeof(struct sockaddr_in));
                } else if (hdr->msg_type == MSG_PING) {
                    // [FIX] 透传 PING 给 Gateway/Master，由真正的 Owner 回复 ACK
                    sendto(s->master_sock, buf, sizeof(struct wvm_header), 0, 
                          (struct sockaddr *)&addrs[i], sizeof(struct sockaddr_in));
                }
            }
            msgs[i].msg_len = 0; 
        }
    }
    g_free(buffers);
    return NULL;
}

static int wavevm_init_machine_kernel(WaveVMAccelState *s, MachineState *ms) {
    fprintf(stderr, "[WaveVM-QEMU] KERNEL MODE: Connecting to /dev/wavevm...\n");
    s->dev_fd = open("/dev/wavevm", O_RDWR);
    if (s->dev_fd < 0) return -errno;
    wavevm_setup_memory_region(ms->ram, ms->ram_size, s->dev_fd);
    // 启动线程
    qemu_thread_create(&s->irq_thread, "wvm-k-irq", wavevm_kernel_irq_thread, s, QEMU_THREAD_DETACHED);
    return 0;
}

static int wavevm_init_machine_user(WaveVMAccelState *s, MachineState *ms) {
    // Slave Mode Check (FD Inheritance)
    char *env_cmd = getenv("WVM_SOCK_CMD");
    
    if (env_cmd) {
        // [Slave Mode]
        s->master_sock = atoi(env_cmd); // Use CMD socket for control loop
    } else {
        // [Master Mode]
        // 读取环境变量以支持单机多实例
        const char *shm_path = getenv("WVM_SHM_FILE");
        if (!shm_path) shm_path = "/wavevm_ram"; // Default fallback (Keep sync with config.h)

        int shm_fd = shm_open(shm_path, O_CREAT | O_RDWR, 0666);
        if (shm_fd < 0) {
            fprintf(stderr, "WaveVM: Failed to open SHM file '%s': %s\n", shm_path, strerror(errno));
            exit(1);
        }
        
        // 确保文件大小足够 (Daemon 应该已经 truncate 过了，这里是双保险)
        if (ftruncate(shm_fd, ms->ram_size) < 0) {
            perror("ftruncate");
            close(shm_fd);
            exit(1);
        }

        wavevm_setup_memory_region(ms->ram, ms->ram_size, shm_fd);
        close(shm_fd);
        
        // Start KVM Sync Thread (Only needed for KVM Master)
        if (kvm_enabled()) {
            s->sync_thread_running = true;
            qemu_thread_create(&s->sync_thread, "wvm-sync", wavevm_dirty_sync_thread, s, QEMU_THREAD_DETACHED);
            // 启动 IPC 监听线程 (处理 VFIO 中断)
            qemu_thread_create(&s->ipc_thread, "wvm-ipc-rx", wavevm_master_ipc_thread, s, QEMU_THREAD_DETACHED);
        }
    }
        
    // 初始化拦截信号。某些机型在此时 ms->ram 仍可能为空，需做兜底查找。
    void *ram_ptr = NULL;
    if (ms->ram && memory_region_is_ram(ms->ram)) {
        ram_ptr = memory_region_get_ram_ptr(ms->ram);
    } else {
        MemoryRegionSection sec = memory_region_find(&address_space_memory, 0, ms->ram_size ? ms->ram_size : 1);
        if (sec.mr && memory_region_is_ram(sec.mr)) {
            ram_ptr = memory_region_get_ram_ptr(sec.mr) + sec.offset_within_region;
        }
    }
    if (!ram_ptr) {
        error_report("WaveVM: failed to resolve guest RAM pointer in user mode init");
        return -ENODEV;
    }
    wavevm_user_mem_init(ram_ptr, ms->ram_size);

    return 0;
}

static struct wvm_ioctl_mem_layout global_layout;

static void wavevm_region_add(MemoryListener *listener, MemoryRegionSection *section) {
    if (!memory_region_is_ram(section->mr)) return;

    uint64_t start_gpa = section->offset_within_address_space;
    uint64_t size = int128_get64(section->size);
    
    if (size < 4096) return;

    // 记录到本地表
    if (global_layout.count < 32) {
        global_layout.slots[global_layout.count].start = start_gpa;
        global_layout.slots[global_layout.count].size = size;
        global_layout.count++;
        
        // 同时通知 User-Mem 映射表
        void *hva = memory_region_get_ram_ptr(section->mr) + section->offset_within_region;
        extern void wavevm_register_ram_block(void *hva, uint64_t size, uint64_t gpa);
        wavevm_register_ram_block(hva, size, start_gpa);
    }
}

// 关键：在虚拟机启动完成前，将拓扑同步给内核
static void wavevm_sync_topology(int dev_fd) {
    if (ioctl(dev_fd, IOCTL_SET_MEM_LAYOUT, &global_layout) < 0) {
        perror("[WVM] Failed to sync memory layout to kernel (continue)");
    }
}

// 监听器结构体
static MemoryListener wavevm_mem_listener = {
    .region_add = wavevm_region_add,
};

static int wavevm_init_machine(MachineState *ms) {
    WaveVMAccelState *s = WAVEVM_ACCEL(ms->accelerator);
    bool has_wvm_dev = (access("/dev/wavevm", R_OK | W_OK) == 0);
    char *role = getenv("WVM_ROLE");
    bool is_slave = (role && strcmp(role, "SLAVE") == 0);

    if (!g_wvm_mode_explicit) {
        if (is_slave) {
            s->mode = WVM_MODE_USER;
        } else {
            s->mode = has_wvm_dev ? WVM_MODE_KERNEL : WVM_MODE_USER;
        }
    }

    if (s->mode == WVM_MODE_KERNEL && !has_wvm_dev) {
        return -1;
    }

    int ret = 0;

    wavevm_resolve_split(ms);

    memory_listener_register(&wavevm_mem_listener, &address_space_memory);

    if (s->mode == WVM_MODE_KERNEL) {
        s->dev_fd = open("/dev/wavevm", O_RDWR);
        if (s->dev_fd < 0) return -errno;
        
        wavevm_sync_topology(s->dev_fd); 
        
        wavevm_setup_memory_region(ms->ram, ms->ram_size, s->dev_fd);
        qemu_thread_create(&s->irq_thread, "wvm-k-irq", wavevm_kernel_irq_thread, s, QEMU_THREAD_DETACHED);
    } 

    if (s->mode == WVM_MODE_KERNEL) ret = wavevm_init_machine_kernel(s, ms);
    else ret = wavevm_init_machine_user(s, ms);
    if (ret < 0) return ret;

    // 只要是 Slave (无论 KVM 还是 TCG)，都启动网络处理线程
    // 这样 TCG Slave 也能接收 CPU 任务和内存请求
    if (is_slave) {
        // Now safe to run net thread in TCG mode because it uses a separate socket
        printf("[WaveVM] Starting Slave Net Thread on FD %d...\n", s->master_sock);
        qemu_thread_create(&s->net_thread, "wvm-slave-net", wavevm_slave_net_thread, s, QEMU_THREAD_DETACHED);
        // 暂停主 vCPU 线程，控制权移交给 net_thread 驱动
        if (current_cpu) { current_cpu->stop = true; current_cpu->halted = true; }
    }

    if (s->mode == WVM_MODE_KERNEL) {
        wavevm_sync_topology(s->dev_fd); 
    }
    return 0;
}

static char *wavevm_get_mode(Object *obj, Error **errp)
{
    WaveVMAccelState *s = WAVEVM_ACCEL(obj);
    return g_strdup(s->mode == WVM_MODE_USER ? "user" : "kernel");
}

static void wavevm_set_mode(Object *obj, const char *value, Error **errp)
{
    WaveVMAccelState *s = WAVEVM_ACCEL(obj);
    if (g_strcmp0(value, "user") == 0) {
        s->mode = WVM_MODE_USER;
        g_wvm_mode_explicit = true;
        return;
    }
    if (g_strcmp0(value, "kernel") == 0) {
        s->mode = WVM_MODE_KERNEL;
        g_wvm_mode_explicit = true;
        return;
    }
    error_setg(errp, "Invalid mode '%s' (expected 'kernel' or 'user')", value);
}

static void wavevm_set_split(Object *obj, Visitor *v, const char *name, void *opaque, Error **errp) {
    uint32_t value; if (!visit_type_uint32(v, name, &value, errp)) return;
    g_wvm_local_split = value;
    g_wvm_split_explicit = true;
}
static void wavevm_get_split(Object *obj, Visitor *v, const char *name, void *opaque, Error **errp) {
    uint32_t value = g_wvm_local_split; visit_type_uint32(v, name, &value, errp);
}
static void wavevm_set_ttl(Object *obj, Visitor *v, const char *name, void *opaque, Error **errp) {
    uint32_t value;
    if (!visit_type_uint32(v, name, &value, errp)) return;
    wvm_set_ttl_interval((int)value);
}
static void wavevm_get_ttl(Object *obj, Visitor *v, const char *name, void *opaque, Error **errp) {
    // Getter implementation usually needed for QOM but can be stubbed if write-only intent
    // keeping it simple
    uint32_t val = 0; 
    visit_type_uint32(v, name, &val, errp);
}
static void wavevm_set_watch(Object *obj, Visitor *v, const char *name, void *opaque, Error **errp) {
    char *value;
    if (!visit_type_str(v, name, &value, errp)) return;
    
    // 解析格式: "0xC0000000:16M;0x800000000:128K"
    // 使用副本进行分割，防止破坏原字符串
    char *dup = g_strdup(value);
    char *saveptr;
    char *token = strtok_r(dup, ";", &saveptr);
    
    while (token) {
        uint64_t gpa = 0;
        uint64_t size_bytes = 0;
        uint64_t size_val = 0;
        char unit = 0;
        
        // sscanf 解析 hex:size+unit
        if (sscanf(token, "%lx:%lu%c", &gpa, &size_val, &unit) >= 2) {
            size_bytes = size_val;
            if (unit == 'M' || unit == 'm') size_bytes *= 1024 * 1024;
            else if (unit == 'G' || unit == 'g') size_bytes *= 1024 * 1024 * 1024;
            else if (unit == 'K' || unit == 'k') size_bytes *= 1024;
            
            wvm_register_volatile_ram(gpa, size_bytes);
        }
        token = strtok_r(NULL, ";", &saveptr);
    }
    g_free(dup);
    g_free(value);
}
static void wavevm_get_watch(Object *obj, Visitor *v, const char *name, void *opaque, Error **errp) {
    char *val = g_strdup("");
    visit_type_str(v, name, &val, errp);
    g_free(val);
}
static void wavevm_accel_init(Object *obj) {
    WaveVMAccelState *s = WAVEVM_ACCEL(obj);
    s->mode = WVM_MODE_KERNEL;
}
static void wavevm_accel_class_init(ObjectClass *oc, void *data) {
    AccelClass *ac = ACCEL_CLASS(oc);
    static const CpusAccel wavevm_cpus = {
        .create_vcpu_thread = wavevm_start_vcpu_thread,
    };

    ac->name = "wavevm";
    ac->init_machine = wavevm_init_machine;
    ac->allowed = &wavevm_allowed;

    cpus_register_accel(&wavevm_cpus);

    object_class_property_add_str(oc, "mode", wavevm_get_mode, wavevm_set_mode);
    object_class_property_set_description(oc, "mode",
        "Execution mode: 'kernel' or 'user'");

    object_class_property_add(oc, "split", "int",
        wavevm_get_split, wavevm_set_split,
        NULL, NULL);
    object_class_property_set_description(oc, "split",
        "Number of vCPUs to run locally (Tier 1)");

    object_class_property_add(oc, "watch", "string",
        wavevm_get_watch, wavevm_set_watch,
        NULL, NULL);
    object_class_property_set_description(oc, "watch",
        "List of volatile RAM ranges (e.g. 0xC0000:16M)");

    #ifndef CONFIG_USER_ONLY
    #endif
}
static const TypeInfo wavevm_accel_type = {
    .name = TYPE_WAVEVM_ACCEL, .parent = TYPE_ACCEL, .instance_size = sizeof(WaveVMAccelState),
    .class_init = wavevm_accel_class_init, .instance_init = wavevm_accel_init,
};
static void wavevm_type_init(void) { type_register_static(&wavevm_accel_type); }
type_init(wavevm_type_init);

// [Helper] 健壮写：处理 EINTR 和 EAGAIN
static int write_all(int fd, const void *buf, size_t len) {
    size_t written = 0;
    const char *ptr = buf;
    while (written < len) {
        ssize_t ret = write(fd, ptr + written, len - written);
        if (ret < 0) {
            if (errno == EINTR) continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                struct pollfd pfd = { .fd = fd, .events = POLLOUT };
                poll(&pfd, 1, 100); // Wait 100ms
                continue;
            }
            return -1;
        }
        written += ret;
    }
    return 0;
}

// [Helper] 健壮读：处理 EINTR 和 EAGAIN
static int read_all(int fd, void *buf, size_t len) {
    size_t received = 0;
    char *ptr = buf;
    while (received < len) {
        ssize_t ret = read(fd, ptr + received, len - received);
        if (ret < 0) {
            if (errno == EINTR) continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                struct pollfd pfd = { .fd = fd, .events = POLLIN };
                poll(&pfd, 1, 100); 
                continue;
            }
            return -1;
        }
        if (ret == 0) return -1; // EOF (Daemon died)
        received += ret;
    }
    return 0;
}

// [V29 Prophet Core] 同步 RPC 发送
// 返回 0 成功，-1 失败
int wvm_send_rpc_sync(uint16_t msg_type, void *payload, size_t len) {
    WaveVMAccelState *s = WAVEVM_ACCEL(current_machine->accelerator);
    int fd = -1;
    int needs_close = 0;

    // 1. 获取连接 FD
    if (s->mode == WVM_MODE_USER) {
        char *role = getenv("WVM_ROLE");
        // Slave: 复用 CMD 通道
        if (role && strcmp(role, "SLAVE") == 0) {
            fd = s->master_sock; 
        } else {
            // Master: 复用 Sync 通道或新建
            if (s->sync_sock > 0) fd = s->sync_sock;
            else { fd = connect_to_master_helper(); needs_close = 1; }
        }
    } else {
        return -1; // Kernel Mode 不支持此路径
    }

    if (fd < 0) return -1;

    // 2. 构造数据包 (IPC头 + WVM头 + Payload)
    size_t wvm_pkt_len = sizeof(struct wvm_header) + len;
    size_t total_size = sizeof(struct wvm_ipc_header_t) + wvm_pkt_len;
    
    uint8_t *buffer = g_malloc(total_size);
    if (!buffer) { if (needs_close) close(fd); return -1; }

    struct wvm_ipc_header_t *ipc_hdr = (struct wvm_ipc_header_t *)buffer;
    ipc_hdr->type = 99; // WVM_IPC_TYPE_RPC_PASSTHROUGH
    ipc_hdr->len = wvm_pkt_len;

    struct wvm_header *hdr = (struct wvm_header *)(buffer + sizeof(struct wvm_ipc_header_t));
    hdr->magic = htonl(WVM_MAGIC);
    hdr->msg_type = htons(msg_type);
    hdr->payload_len = htons(len);
    hdr->slave_id = 0; 
    hdr->req_id = WVM_HTONLL(SYNC_MAGIC); // "SYNC" Magic
    hdr->qos_level = 1; 
    hdr->crc32 = 0; 

    if (len > 0) memcpy(buffer + sizeof(struct wvm_ipc_header_t) + sizeof(struct wvm_header), payload, len);

    // 3. 发送指令
    if (write_all(fd, buffer, total_size) < 0) {
        g_free(buffer); if (needs_close) close(fd); return -1;
    }
    g_free(buffer);

    // 4. [BLOCKING] 等待 ACK
    // 10秒超时，防止 Daemon 执行 memset 耗时过长或崩溃
    struct pollfd pfd = { .fd = fd, .events = POLLIN };
    int ret = poll(&pfd, 1, 10000); 
    
    if (ret > 0 && (pfd.revents & POLLIN)) {
        uint8_t ack_byte;
        if (read_all(fd, &ack_byte, 1) == 0) {
            if (needs_close) close(fd);
            return 0; // Success
        }
    }

    if (needs_close) close(fd);
    return -1; // Fail/Timeout
}
