
#include "qemu/osdep.h"
#include "cpu.h"
#include "sysemu/cpus.h"
#include "sysemu/kvm.h" 
#include "linux/kvm.h"
#include "qemu/main-loop.h"
#include "exec/address-spaces.h"
#include "../../../common_include/wavevm_protocol.h" 
#include "../../../common_include/wavevm_config.h"
#include "wavevm-accel.h"
#include "qemu/thread.h"
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

// Per-vCPU Socket Pool to eliminate lock contention
static int *g_vcpu_socks = NULL;
static int g_configured_vcpus = 0;

// TCG Helper Declarations (Defined in wavevm-tcg.c)
extern void wvm_tcg_get_state(CPUState *cpu, wvm_tcg_context_t *ctx);
extern void wvm_tcg_set_state(CPUState *cpu, wvm_tcg_context_t *ctx);

// 引用 wavevm-all.c 中定义的全局变量
extern int g_wvm_local_split;

struct wavevm_policy_ops {
    int (*schedule_policy)(int cpu_index);
};

// [Policy] Tiered Scheduling: Local vs Remote
static int remote_rpc_policy(int cpu_index) {
    //不再使用 WVM_LOCAL_CPU_COUNT 宏，而是使用动态变量
    if (cpu_index >= g_wvm_local_split) return 1; // 远程执行
    return 0; // 本地执行
}

static struct wavevm_policy_ops ops = { .schedule_policy = remote_rpc_policy };

/* 
 * [物理意图] 充当远程 Slave 的“本地代理执行人”。
 * [关键逻辑] 当远程计算节点触发 PIO/MMIO 退出时，Master 在本地 QEMU 中代为执行该 I/O 操作并返回结果。
 * [后果] 解决了 I/O 设备物理位置的透明性。它让 Guest 以为显卡就在本地，即便真实的读写指令是在千里之外执行的。
 */
static void wavevm_handle_io(CPUState *cpu) {
    struct kvm_run *run = cpu->kvm_run;
    uint16_t port = run->io.port;
    void *data = (uint8_t *)run + run->io.data_offset;
    
    address_space_rw(&address_space_io, port, MEMTXATTRS_UNSPECIFIED,
                     data, run->io.size,
                     run->io.direction == KVM_EXIT_IO_OUT);
}

static void wavevm_handle_mmio(CPUState *cpu) {
    struct kvm_run *run = cpu->kvm_run;
    hwaddr addr = run->mmio.phys_addr;
    void *data = run->mmio.data;

    address_space_rw(&address_space_memory, addr, MEMTXATTRS_UNSPECIFIED,
                     data, run->mmio.len,
                     run->mmio.is_write);
}

/* 
 * [物理意图] 实现 CPU 核心的“跨物理节点迁移（Context Serialization）”。
 * [关键逻辑] 将 CPU 的全量寄存器状态、MSR 及时钟计数器进行二进制打包，并发射至远程 Slave。
 * [后果] 实现了算力的透明流动。若序列化逻辑不严谨（如漏掉 rflags），远程执行会立即触发 Guest Kernel Panic。
 */
static void wavevm_remote_exec(CPUState *cpu) {
    // 动态边界检查
    if (cpu->cpu_index >= g_configured_vcpus) return;
    
    int vcpu_sock = g_vcpu_socks[cpu->cpu_index];
    
    // 如果没有 socket (说明是 Kernel Mode), 走 IOCTL 路径
    if (vcpu_sock < 0) { 
        WaveVMAccelState *s = WAVEVM_ACCEL(current_machine->accelerator);
        if (s->mode != WVM_MODE_KERNEL) {
            g_usleep(1000); // 异常状态，非 Kernel 且无 Socket
            return;
        }

        // 1. 准备请求结构体
        struct wvm_ipc_cpu_run_req req;
        struct wvm_ipc_cpu_run_ack ack;
        memset(&req, 0, sizeof(req));
        
        req.slave_id = WVM_NODE_AUTO_ROUTE;

        // 2. 序列化 CPU 状态
        if (kvm_enabled()) {
            struct kvm_regs kregs;
            struct kvm_sregs ksregs;
            cpu_synchronize_state(cpu);
            ioctl(cpu->kvm_fd, KVM_GET_REGS, &kregs);
            ioctl(cpu->kvm_fd, KVM_GET_SREGS, &ksregs);

            req.mode_tcg = 0;
            req.ctx.kvm.rax = kregs.rax; req.ctx.kvm.rbx = kregs.rbx; req.ctx.kvm.rcx = kregs.rcx;
            req.ctx.kvm.rdx = kregs.rdx; req.ctx.kvm.rsi = kregs.rsi; req.ctx.kvm.rdi = kregs.rdi;
            req.ctx.kvm.rsp = kregs.rsp; req.ctx.kvm.rbp = kregs.rbp;
            req.ctx.kvm.r8  = kregs.r8;  req.ctx.kvm.r9  = kregs.r9;  req.ctx.kvm.r10 = kregs.r10;
            req.ctx.kvm.r11 = kregs.r11; req.ctx.kvm.r12 = kregs.r12; req.ctx.kvm.r13 = kregs.r13;
            req.ctx.kvm.r14 = kregs.r14; req.ctx.kvm.r15 = kregs.r15;
            req.ctx.kvm.rip = kregs.rip; req.ctx.kvm.rflags = kregs.rflags;
            memcpy(req.ctx.kvm.sregs_data, &ksregs, sizeof(ksregs));
        } else {
            req.mode_tcg = 1;
            wvm_tcg_get_state(cpu, &req.ctx.tcg);
        }

        // 3. 陷入内核 (Trap into Kernel)
        // 这一步会阻塞，直到远程执行完毕并返回结果
        int ret = ioctl(s->dev_fd, IOCTL_WVM_REMOTE_RUN, &req);
        
        if (ret < 0) {
            //fprintf(stderr, "WaveVM: Remote Run IOCTL failed: %s\n", strerror(errno));
            return;
        }

        // 4. 反序列化结果 (直接复用 req 的内存空间读取 ack，或者使用 memcpy)
        // 注意：内核将 Ack 数据回写到了 req 指针所在的内存
        memcpy(&ack, &req, sizeof(ack)); 

        if (ack.mode_tcg) {
            wvm_tcg_set_state(cpu, &ack.ctx.tcg);
        } else {
            struct kvm_regs kregs;
            struct kvm_sregs ksregs;
            wvm_kvm_context_t *kctx = &ack.ctx.kvm;

            kregs.rax = kctx->rax; kregs.rbx = kctx->rbx; kregs.rcx = kctx->rcx; 
            kregs.rdx = kctx->rdx; kregs.rsi = kctx->rsi; kregs.rdi = kctx->rdi;
            kregs.rsp = kctx->rsp; kregs.rbp = kctx->rbp;
            kregs.r8 = kctx->r8;   kregs.r9 = kctx->r9;   kregs.r10 = kctx->r10; 
            kregs.r11 = kctx->r11; kregs.r12 = kctx->r12; kregs.r13 = kctx->r13;
            kregs.r14 = kctx->r14; kregs.r15 = kctx->r15;
            kregs.rip = kctx->rip; kregs.rflags = kctx->rflags;
            
            memcpy(&ksregs, kctx->sregs_data, sizeof(ksregs));
            ioctl(cpu->kvm_fd, KVM_SET_SREGS, &ksregs);
            ioctl(cpu->kvm_fd, KVM_SET_REGS, &kregs);
            
            struct kvm_run *run = cpu->kvm_run;
            run->exit_reason = kctx->exit_reason;

            if (kctx->exit_reason == KVM_EXIT_IO) {
                run->io.direction = kctx->exit_info.io.direction;
                run->io.size      = kctx->exit_info.io.size;
                run->io.port      = kctx->exit_info.io.port;
                run->io.count     = kctx->exit_info.io.count;
                if (run->io.direction == KVM_EXIT_IO_OUT) {
                    size_t mmap_size = ioctl(cpu->kvm_fd, KVM_GET_VCPU_MMAP_SIZE, 0);
                    if (run->io.data_offset + run->io.size * run->io.count <= cpu->mmap_size) {
                        uint8_t *io_ptr = (uint8_t *)run + run->io.data_offset;
                        memcpy(io_ptr, kctx->exit_info.io.data, run->io.size * run->io.count);
                    }
                }
                qemu_mutex_lock_iothread();
                wavevm_handle_io(cpu);
                qemu_mutex_unlock_iothread();
            } 
            else if (kctx->exit_reason == KVM_EXIT_MMIO) {
                run->mmio.phys_addr = kctx->exit_info.mmio.phys_addr;
                run->mmio.len       = kctx->exit_info.mmio.len;
                run->mmio.is_write  = kctx->exit_info.mmio.is_write;
                memcpy(run->mmio.data, kctx->exit_info.mmio.data, 8);
                qemu_mutex_lock_iothread();
                wavevm_handle_mmio(cpu);
                qemu_mutex_unlock_iothread();
            }
        }
        return; 
    }

    // 准备发送缓冲区
    uint8_t buf[16384]; //需要调大一点点
    struct wvm_header *hdr = (struct wvm_header *)buf;
    struct wvm_ipc_cpu_run_req *req = (struct wvm_ipc_cpu_run_req *)(buf + sizeof(struct wvm_header));
    req->slave_id = WVM_NODE_AUTO_ROUTE; 
    req->vcpu_index = cpu->cpu_index;
    
    uint32_t target_slave = WVM_NODE_AUTO_ROUTE; 

    hdr->magic = WVM_MAGIC;
    hdr->msg_type = MSG_VCPU_RUN;
    hdr->slave_id = target_slave;
    // 使用全1作为异步标记，避开 GPA 0
    hdr->req_id = WVM_HTONLL(~0ULL);
    hdr->is_frag = 0;

    // 将 vCPU ID 埋入 Payload 的冗余字段中，供 Slave 端的 Proxy 负载均衡使用
    req->slave_id = cpu->cpu_index; 

    // 1. 序列化 CPU 状态 (Serialization)
    if (kvm_enabled()) {
        struct kvm_regs kregs;
        struct kvm_sregs ksregs;
        cpu_synchronize_state(cpu);
        ioctl(cpu->kvm_fd, KVM_GET_REGS, &kregs);
        ioctl(cpu->kvm_fd, KVM_GET_SREGS, &ksregs);

        hdr->mode_tcg = 0;
        hdr->payload_len = sizeof(wvm_kvm_context_t);
        
        wvm_kvm_context_t *kctx = &req->ctx.kvm;
        kctx->rax = kregs.rax; kctx->rbx = kregs.rbx; kctx->rcx = kregs.rcx;
        kctx->rdx = kregs.rdx; kctx->rsi = kregs.rsi; kctx->rdi = kregs.rdi;
        kctx->rsp = kregs.rsp; kctx->rbp = kregs.rbp;
        kctx->r8  = kregs.r8;  kctx->r9  = kregs.r9;  kctx->r10 = kregs.r10;
        kctx->r11 = kregs.r11; kctx->r12 = kregs.r12; kctx->r13 = kregs.r13;
        kctx->r14 = kregs.r14; kctx->r15 = kregs.r15;
        kctx->rip = kregs.rip; kctx->rflags = kregs.rflags;
        memcpy(kctx->sregs_data, &ksregs, sizeof(ksregs));
    } else {
        hdr->mode_tcg = 1;
        hdr->payload_len = sizeof(wvm_tcg_context_t);
        wvm_tcg_get_state(cpu, &req->ctx.tcg);
    }

    // 2. 网络发送 (Lock-Free)
    size_t packet_len = sizeof(struct wvm_header) + hdr->payload_len;
    // [V25.4 REMOVED] qemu_mutex_lock(&s->ipc_lock);
    if (write(vcpu_sock, buf, packet_len) != packet_len) {
        // Log error or handle reconnect
        return;
    }

    // 3. 网络接收 (阻塞本线程，不影响其他 vCPU)
    ssize_t len = read(vcpu_sock, buf, sizeof(buf));
    
    if (len < sizeof(struct wvm_header)) return;
    
    // 4. 反序列化 CPU 状态
    struct wvm_ipc_cpu_run_ack *ack = (struct wvm_ipc_cpu_run_ack *)(buf + sizeof(struct wvm_header));

    if (ack->mode_tcg) {
        wvm_tcg_set_state(cpu, &ack->ctx.tcg);
    } else {
        struct kvm_regs kregs;
        struct kvm_sregs ksregs;
        wvm_kvm_context_t *kctx = &ack->ctx.kvm;

        kregs.rax = kctx->rax; kregs.rbx = kctx->rbx; kregs.rcx = kctx->rcx; 
        kregs.rdx = kctx->rdx; kregs.rsi = kctx->rsi; kregs.rdi = kctx->rdi;
        kregs.rsp = kctx->rsp; kregs.rbp = kctx->rbp;
        kregs.r8 = kctx->r8;   kregs.r9 = kctx->r9;   kregs.r10 = kctx->r10; 
        kregs.r11 = kctx->r11; kregs.r12 = kctx->r12; kregs.r13 = kctx->r13;
        kregs.r14 = kctx->r14; kregs.r15 = kctx->r15;
        kregs.rip = kctx->rip; kregs.rflags = kctx->rflags;
        
        memcpy(&ksregs, kctx->sregs_data, sizeof(ksregs));
        ioctl(cpu->kvm_fd, KVM_SET_SREGS, &ksregs);
        ioctl(cpu->kvm_fd, KVM_SET_REGS, &kregs);
        
        // 5. Replay IO/MMIO
        struct kvm_run *run = cpu->kvm_run;
        run->exit_reason = kctx->exit_reason;

        if (kctx->exit_reason == KVM_EXIT_IO) {
            run->io.direction = kctx->exit_info.io.direction;
            run->io.size      = kctx->exit_info.io.size;
            run->io.port      = kctx->exit_info.io.port;
            run->io.count     = kctx->exit_info.io.count;
            
            if (run->io.direction == KVM_EXIT_IO_OUT) {
                size_t mmap_size = ioctl(cpu->kvm_fd, KVM_GET_VCPU_MMAP_SIZE, 0);
                if (run->io.data_offset + run->io.size * run->io.count <= mmap_size) {
                    uint8_t *io_ptr = (uint8_t *)run + run->io.data_offset;
                    memcpy(io_ptr, kctx->exit_info.io.data, run->io.size * run->io.count);
                }
            }
            wavevm_handle_io(cpu);
        } 
        else if (kctx->exit_reason == KVM_EXIT_MMIO) {
            run->mmio.phys_addr = kctx->exit_info.mmio.phys_addr;
            run->mmio.len       = kctx->exit_info.mmio.len;
            run->mmio.is_write  = kctx->exit_info.mmio.is_write;
            memcpy(run->mmio.data, kctx->exit_info.mmio.data, 8);
            wavevm_handle_mmio(cpu);
        }
    }
}

/* 
 * [物理意图] 重新定义 vCPU 的“心脏跳动规律”。
 * [关键逻辑] 拦截标准的 KVM_RUN 循环，根据调度策略决定本轮指令是交给本地 KVM 还是进行远程上下文序列化。
 * [后果] 这是超级虚拟机的总节拍器。它保证了在异构算力环境下，vCPU 能够平滑地在本地与远程之间切换执行流。
 */
static void *wavevm_cpu_thread_fn(void *arg) {
    CPUState *cpu = arg;
    int ret;

    rcu_register_thread();
    cpu->halted = 0;
    
    if (kvm_enabled()) {
        qemu_mutex_lock_iothread();
        cpu_synchronize_state(cpu);
        qemu_mutex_unlock_iothread();
    }

    while (1) {
        if (cpu->unplug || cpu->stop) break;

        if (ops.schedule_policy(cpu->cpu_index) == 1) {
            wavevm_remote_exec(cpu);
            continue;
        }

        if (cpu_can_run(cpu)) {
            if (kvm_enabled()) {
                qemu_mutex_lock_iothread();
                ret = kvm_vcpu_ioctl(cpu, KVM_RUN, 0);
                qemu_mutex_unlock_iothread();

                if (ret < 0) {
                    if (errno == EINTR || errno == EAGAIN) continue;
                    fprintf(stderr, "KVM_RUN failed: %s\n", strerror(errno));
                    break;
                }
                
                struct kvm_run *run = cpu->kvm_run;
                switch (run->exit_reason) {
                    case KVM_EXIT_IO: wavevm_handle_io(cpu); break;
                    case KVM_EXIT_MMIO: wavevm_handle_mmio(cpu); break;
                    case KVM_EXIT_SHUTDOWN: 
                        qemu_system_reset_request(SHUTDOWN_CAUSE_GUEST_SHUTDOWN);
                        goto out;
                    case KVM_EXIT_HLT:
                        qemu_wait_io_event(cpu);
                        break;
                    default: break;
                }
            } else {
                qemu_mutex_lock_iothread();
                qemu_tcg_cpu_exec(cpu);
                qemu_mutex_unlock_iothread();
            }
        } else {
            qemu_wait_io_event(cpu);
        }
    }
out:
    rcu_unregister_thread();
    return NULL;
}

// 替换 QEMU 默认的 vCPU 线程启动逻辑
// 导出 connect_to_master_helper 以便调用
extern int connect_to_master_helper(void);

/* 
 * [物理意图] 为每个逻辑核心开辟专属的“高速公路（Per-vCPU Socket Pool）”。
 * [关键逻辑] 在 vCPU 创建瞬间建立独立的 IPC 通道，彻底消除多核并发时的 Socket 锁竞争。
 * [后果] 实现了算力的线性扩展能力。这是 V30 支撑一亿核心的关键，保证了 vCPU 数量增加时通信延迟不上升。
 */
void wavevm_start_vcpu_thread(CPUState *cpu) {
    char thread_name[VCPU_THREAD_NAME_SIZE];
    WaveVMAccelState *s = WAVEVM_ACCEL(current_machine->accelerator);
    char *role = getenv("WVM_ROLE");

    static pthread_mutex_t g_init_lock = PTHREAD_MUTEX_INITIALIZER;
    
    // 双重检查锁定 (Double-Checked Locking) 优化性能，或者直接加锁也行（毕竟只执行一次）
    if (!g_vcpu_socks) {
        pthread_mutex_lock(&g_init_lock);
        // 再次检查，防止在等待锁的过程中已被其他线程初始化
        if (!g_vcpu_socks) {
            MachineState *ms = MACHINE(qdev_get_machine());
            g_configured_vcpus = ms->smp.cpus; 
            
            g_vcpu_socks = g_malloc0(sizeof(int) * g_configured_vcpus);
            for (int i = 0; i < g_configured_vcpus; i++) {
                g_vcpu_socks[i] = -1;
            }
        }
        pthread_mutex_unlock(&g_init_lock);
    }

    if (s->mode == WVM_MODE_USER && !(role && strcmp(role, "SLAVE") == 0)) {
        // 动态边界检查
        if (cpu->cpu_index < g_configured_vcpus) {
            g_vcpu_socks[cpu->cpu_index] = connect_to_master_helper();
            if (g_vcpu_socks[cpu->cpu_index] < 0) {
                fprintf(stderr, "vCPU %d failed to connect master!\n", cpu->cpu_index);
                exit(1);
            }
        }
    } else {
        // Kernel 模式和 Slave 模式不使用此机制
        g_vcpu_socks[cpu->cpu_index] = -1; 
    }
    
    cpu->thread = g_malloc0(sizeof(QemuThread));
    cpu->halt_cond = g_malloc0(sizeof(QemuCond));
    qemu_cond_init(cpu->halt_cond);
    
    snprintf(thread_name, VCPU_THREAD_NAME_SIZE, "CPU %d/WVM", cpu->cpu_index);
    
    qemu_thread_create(cpu->thread, thread_name, wavevm_cpu_thread_fn, cpu, QEMU_THREAD_JOINABLE);
}
