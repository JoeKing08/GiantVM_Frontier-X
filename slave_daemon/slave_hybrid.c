
/*
 * [IDENTITY] Slave Hybrid - The Execution Muscle
 * ---------------------------------------------------------------------------
 * 物理角色：远程节点上的"计算代理"。
 * 职责边界：
 * 1. 托管 KVM/TCG 上下文执行，将 IO/MMIO 退出转发回 Master。
 * 2. 实现 MPSC 脏页收割队列，将 Slave 端的写入异步同步给 Master。
 * 3. 驱动三通道 (CMD/REQ/PUSH) 隔离协议，防止网络拥塞。
 * 
 * [性能红线]
 * - 脏页收割频率严禁低于 1ms，否则 Master 端的读缓存失效太久。
 * - 严禁在 KVM_RUN 路径上引入同步网络等待。
 * ---------------------------------------------------------------------------
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/kvm.h>
#include <errno.h>
#include <sched.h>
#include <sys/wait.h>
#include <poll.h>
#include "slave_vfio.h"

#include "../common_include/giantvm_protocol.h"

// --- 全局配置变量 ---
static int g_service_port = 9000;
static long g_num_cores = 0;
static int g_ram_mb = 1024;
static uint64_t g_slave_ram_size = 1024UL * 1024 * 1024;
// 用于通知 IRQ 线程 Master 地址已就绪
static pthread_cond_t g_master_cond = PTHREAD_COND_INITIALIZER;
static pthread_mutex_t g_master_mutex = PTHREAD_MUTEX_INITIALIZER;
static int g_master_ready = 0;
static struct sockaddr_in g_master_addr;
static char *g_vfio_config_path = NULL; 

// MPSC 队列数据结构

// 任务单元：封装一个需要同步的脏页
typedef struct {
    uint64_t gpa;
    uint8_t  data[4096];
} dirty_page_task_t;

// MPSC 环形队列
#define MPSC_QUEUE_SIZE 1024 // 可缓存 1024 个脏页（4MB），足以应对网络瞬时抖动
static dirty_page_task_t* g_mpsc_queue[MPSC_QUEUE_SIZE];
static volatile uint32_t g_mpsc_head = 0;
static volatile uint32_t g_mpsc_tail = 0;

// 用于保护队列的互斥锁（生产者侧使用）和用于唤醒消费者的条件变量
static pthread_mutex_t g_mpsc_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_mpsc_cond = PTHREAD_COND_INITIALIZER;

#define BATCH_SIZE 32
static int g_kvm_available = 0;

/* 
 * [物理意图] 实现具备“阻塞感知”能力的可靠 UDP 发送，作为 Slave 到 Master 的数据回传通道。
 * [关键逻辑] 实时监测 socket 缓冲区状态，当网络拥塞（EAGAIN）时执行毫秒级退避重试，而非暴力丢包。
 * [后果] 这是 Slave 端状态回传的生命线。若无此机制，高频的中断或脏页同步会冲垮网络，导致 Master 端逻辑断层。
 */
static int robust_sendto(int fd, const void *buf, size_t len, struct sockaddr_in *dest) {
    int retries = 0;
    struct pollfd pfd = { .fd = fd, .events = POLLOUT };

    while (retries < 100) { // 设定最大尝试次数，防止网络彻底断开时死循环
        ssize_t sent = sendto(fd, buf, len, MSG_DONTWAIT, (struct sockaddr*)dest, sizeof(*dest));
        
        if (sent > 0) return 0; // 发送成功

        if (sent < 0) {
            if (errno == EINTR) continue; // 被信号中断，立即重试
            if (errno == EAGAIN || errno == EWOULDBLOCK || errno == ENOBUFS) {
                // 缓冲区满，挂起等待 5ms 
                poll(&pfd, 1, 5);
                retries++;
                continue;
            }
            return -1; // 致命错误
        }
    }
    return -ETIMEDOUT;
}

/* 
 * [物理意图] 将 Slave 端的“物理硬件中断”注入到分布式虚拟化总线中。
 * [关键逻辑] 监听本地物理 GPU 的 eventfd，一旦触发，立即封装为 MSG_VFIO_IRQ 并发送至 Master 的 Fast Lane。
 * [后果] 实现了分布式 I/O 的闭环。它让远在千里之外的物理显卡中断，能在 100µs 内“瞬移”到 Master 节点的 vCPU 中。
 */
void *vfio_irq_thread_adapter(void *arg) {
    pthread_mutex_lock(&g_master_mutex);
    while (!g_master_ready) {
        pthread_cond_wait(&g_master_cond, &g_master_mutex);
    }
    struct sockaddr_in target = g_master_addr;
    pthread_mutex_unlock(&g_master_mutex);

    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
        perror("[Slave] Failed to create IRQ socket");
        return NULL;
    }

    printf("[Slave] IRQ Forwarder Connected to Master %s:%d\n", 
           inet_ntoa(target.sin_addr), ntohs(target.sin_port));

    // 调用 slave_vfio.c 中的核心轮询逻辑
    gvm_vfio_poll_irqs(sock, &target);
    
    close(sock);
    return NULL;
}

// CPU 核心数探测
int get_allowed_cores() {
    char *env_override = getenv("GVM_CORE_OVERRIDE");
    if (env_override) {
        int val = atoi(env_override);
        if (val > 0) {
            printf("[Hybrid] CPU Cores forced by Env: %d\n", val);
            return val;
        }
    }

    long quota = -1;
    long period = 100000; 

    FILE *fp = fopen("/sys/fs/cgroup/cpu.max", "r");
    if (fp) {
        char buf[64];
        if (fscanf(fp, "%63s %ld", buf, &period) >= 1) {
            if (strcmp(buf, "max") != 0) quota = atol(buf);
        }
        fclose(fp);
    }

    if (quota <= 0) {
        fp = fopen("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r");
        if (fp) {
            if (fscanf(fp, "%ld", &quota) != 1) quota = -1;
            fclose(fp);
        }
        fp = fopen("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r");
        if (fp) {
            long p;
            if (fscanf(fp, "%ld", &p) == 1) period = p;
            fclose(fp);
        }
    }

    int logical_cores = 0;
    if (quota > 0 && period > 0) {
        logical_cores = (int)((quota + period - 1) / period);
        printf("[Hybrid] Container Quota Detected: %ld / %ld = %d Cores\n", quota, period, logical_cores);
    } else {
        logical_cores = (int)sysconf(_SC_NPROCESSORS_ONLN);
        printf("[Hybrid] Using Physical Cores: %d\n", logical_cores);
    }

    if (logical_cores <= 0) logical_cores = 1;
    return logical_cores;
}

// 辅助：头部转本机序
static void ntoh_header(struct gvm_header *hdr) {
    hdr->magic = ntohl(hdr->magic);
    hdr->msg_type = ntohs(hdr->msg_type);
    hdr->payload_len = ntohs(hdr->payload_len);
    hdr->slave_id = ntohl(hdr->slave_id);
    hdr->req_id = GVM_NTOHLL(hdr->req_id);
}

// ==========================================
// [Fast Path] KVM Engine (V28 Fixed)
// ==========================================

static int g_kvm_fd = -1;
static int g_vm_fd = -1;
static uint8_t *g_phy_ram = NULL;
static __thread int t_vcpu_fd = -1;
static __thread struct kvm_run *t_kvm_run = NULL;
static pthread_spinlock_t g_master_lock;
static int g_gvm_dev_fd = -1;

/* 
 * [物理意图] 在远程节点初始化硬件虚拟化容器（KVM），为“算力托管”圈定物理内存边界。
 * [关键逻辑] 创建 KVM 虚拟机实例，利用 mmap 映射大页内存，并根据是否有内核模块（Mode A）选择最优路径。
 * [后果] 确立了远程执行的基石。若初始化失败，Slave 将无法使用硬件加速，只能回退到慢速的 TCG 模拟模式。
 */
void init_kvm_global() {
    g_kvm_fd = open("/dev/kvm", O_RDWR);
    if (g_kvm_fd < 0) return; 

    g_gvm_dev_fd = open("/dev/giantvm", O_RDWR);
    
    // MAP_SHARED 是必须的，否则 madvise 无法正确通知 KVM 释放页面
    if (g_gvm_dev_fd >= 0) {
        printf("[Hybrid] KVM: Detected /dev/giantvm. Enabling On-Demand Paging (Fast Path).\n");
        g_phy_ram = mmap(NULL, g_slave_ram_size, PROT_READ|PROT_WRITE, MAP_SHARED, g_gvm_dev_fd, 0);
    } else {
        // [FIXED] 优先使用 SHM 文件，方便单机测试隔离
        const char *shm_path = getenv("GVM_SHM_FILE");
        if (shm_path) {
            printf("[Hybrid] KVM: Kernel module not found. Using SHM File: %s\n", shm_path);
            int shm_fd = shm_open(shm_path, O_CREAT | O_RDWR, 0666);
            if (shm_fd >= 0) {
                ftruncate(shm_fd, g_slave_ram_size);
                g_phy_ram = mmap(NULL, g_slave_ram_size, PROT_READ|PROT_WRITE, MAP_SHARED, shm_fd, 0);
                close(shm_fd);
            }
        }
        
        // 如果 SHM 失败或未设置，回退到匿名内存
        if (!g_phy_ram || g_phy_ram == MAP_FAILED) {
            printf("[Hybrid] KVM: Using Anonymous RAM.\n");
            g_phy_ram = mmap(NULL, g_slave_ram_size, PROT_READ|PROT_WRITE, 
                             MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
        }
    }

    if (g_phy_ram == MAP_FAILED) { perror("mmap ram"); exit(1); }

    madvise(g_phy_ram, g_slave_ram_size, MADV_HUGEPAGE);
    // [V28 Fix] 不要 MADV_RANDOM，保持默认，让 KVM 能够利用 THP
    
    g_vm_fd = ioctl(g_kvm_fd, KVM_CREATE_VM, 0);
    if (g_vm_fd < 0) { close(g_kvm_fd); g_kvm_fd = -1; return; }

    ioctl(g_vm_fd, KVM_SET_TSS_ADDR, 0xfffbd000);
    __u64 map_addr = 0xfffbc000;
    ioctl(g_vm_fd, KVM_SET_IDENTITY_MAP_ADDR, &map_addr);
    ioctl(g_vm_fd, KVM_CREATE_IRQCHIP, 0);

    struct kvm_userspace_memory_region region = {
        .slot = 0,
        .flags = KVM_MEM_LOG_DIRTY_PAGES,
        .guest_phys_addr = 0,
        .memory_size = g_slave_ram_size,
        .userspace_addr = (uint64_t)g_phy_ram
    };
    ioctl(g_vm_fd, KVM_SET_USER_MEMORY_REGION, &region);

    g_kvm_available = 1;
    pthread_spin_init(&g_master_lock, 0);
    printf("[Hybrid] KVM Hardware Acceleration Active (RAM: %d MB).\n", g_ram_mb);
}

void init_thread_local_vcpu(int vcpu_id) {
    if (t_vcpu_fd >= 0) return;
    t_vcpu_fd = ioctl(g_vm_fd, KVM_CREATE_VCPU, vcpu_id);
    if (t_vcpu_fd < 0) return; 
    int mmap_size = ioctl(g_kvm_fd, KVM_GET_VCPU_MMAP_SIZE, 0);
    t_kvm_run = mmap(NULL, mmap_size, PROT_READ|PROT_WRITE, MAP_SHARED, t_vcpu_fd, 0);
    if (t_kvm_run == MAP_FAILED) exit(1);
}

// [FIX] Thread-Local 缓存，避免高频 malloc/free
static __thread unsigned long *t_dirty_bitmap_cache = NULL;
static __thread size_t t_dirty_bitmap_size = 0;

/* 
 * [物理意图] 充当远程执行节点的“脏页回写引擎”，实现计算与同步的解耦。
 * [关键逻辑] 作为 MPSC 队列的唯一消费者，将各个 vCPU 产生的脏页任务（Dirty Pages）进行有序的异步网络发送。
 * [后果] 它解决了“同步锁死”问题。vCPU 可以在执行完指令后立即继续，无需等待网络 ACK，极大提升了远程主频。
 */
void* dirty_sync_sender_thread(void* arg) {
    printf("[SenderThread] KVM Dirty Page Sender is running.\n");
    
    // 这个 socket 只用于发送，所以不需要 bind
    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        perror("SenderThread socket");
        return NULL;
    }

    while (1) {
        pthread_mutex_lock(&g_mpsc_lock);

        // 如果队列为空，则阻塞等待
        while (g_mpsc_head == g_mpsc_tail) {
            pthread_cond_wait(&g_mpsc_cond, &g_mpsc_lock);
        }

        // 从队列头部取出一个任务
        uint32_t current_head = g_mpsc_head;
        dirty_page_task_t* task = g_mpsc_queue[current_head];
        g_mpsc_head = (current_head + 1) % MPSC_QUEUE_SIZE;
        
        pthread_mutex_unlock(&g_mpsc_lock);
        
        // 网络发送
        uint8_t tx_buf[sizeof(struct gvm_header) + 8 + 4096];
        struct gvm_header *wh = (struct gvm_header *)tx_buf;
        size_t total_len = sizeof(tx_buf);

        wh->magic = htonl(GVM_MAGIC);
        wh->msg_type = htons(MSG_MEM_WRITE);
        wh->payload_len = htons(8 + 4096);
        wh->slave_id = 0; // 源ID在Slave模式下意义不大，可设为0
        wh->req_id = 0;
        wh->qos_level = 0;

        uint64_t net_gpa = GVM_HTONLL(task->gpa);
        memcpy(tx_buf + sizeof(*wh), &net_gpa, 8);
        memcpy(tx_buf + sizeof(*wh) + 8, task->data, 4096);
        
        // 使用 robust_sendto 发送，它内部包含了反压处理逻辑
        robust_sendto(sockfd, tx_buf, total_len, &g_master_addr);
        
        // 任务完成，释放内存
        free(task);
    }
    
    close(sockfd);
    return NULL;
}

/* 
 * [物理意图] 在 Slave 硬件上“瞬间复活”来自 Master 的 CPU 寄存器上下文并执行。
 * [关键逻辑] 1. 注入寄存器状态；2. 启动硬件 KVM_RUN；3. 拦截 MMIO 退出；4. 导出最新状态并回传。
 * [后果] 这是分布式计算的“肌肉”。它让 Master 节点能像调用本地函数一样，将指令流分发到全网任意节点执行。
 */
void handle_kvm_run_stateless(int sockfd, struct sockaddr_in *client, struct gvm_header *hdr, void *payload, int vcpu_id) {
    if (t_vcpu_fd < 0) init_thread_local_vcpu(vcpu_id);
    struct gvm_ipc_cpu_run_req *req = (struct gvm_ipc_cpu_run_req *)payload;
    struct kvm_regs kregs; struct kvm_sregs ksregs;
    gvm_kvm_context_t *ctx = &req->ctx.kvm;

    kregs.rax = ctx->rax; kregs.rbx = ctx->rbx; kregs.rcx = ctx->rcx; kregs.rdx = ctx->rdx;
    kregs.rsi = ctx->rsi; kregs.rdi = ctx->rdi; kregs.rsp = ctx->rsp; kregs.rbp = ctx->rbp;
    kregs.r8  = ctx->r8;  kregs.r9  = ctx->r9;  kregs.r10 = ctx->r10; kregs.r11 = ctx->r11;
    kregs.r12 = ctx->r12; kregs.r13 = ctx->r13; kregs.r14 = ctx->r14; kregs.r15 = ctx->r15;
    kregs.rip = ctx->rip; kregs.rflags = ctx->rflags;
    memcpy(&ksregs, ctx->sregs_data, sizeof(ksregs));

    ioctl(t_vcpu_fd, KVM_SET_SREGS, &ksregs); ioctl(t_vcpu_fd, KVM_SET_REGS, &kregs);
    
    int ret;
    do {
        ret = ioctl(t_vcpu_fd, KVM_RUN, 0);
        if (ret == 0 && t_kvm_run->exit_reason == KVM_EXIT_MMIO) {
            if (gvm_vfio_intercept_mmio(
                    t_kvm_run->mmio.phys_addr,
                    t_kvm_run->mmio.data,
                    t_kvm_run->mmio.len,
                    t_kvm_run->mmio.is_write)) {
                continue; 
            }
        }
        if (ret == 0) break;
    } while (ret > 0 || ret == -EINTR);

    if (g_gvm_dev_fd < 0) { 
        // [V28.5 FIXED] KVM Dirty Log Sync (Full Implementation)
        // 完整实现：获取位图 -> 遍历 -> 封包 -> 发送
        // 这里的 slot 0 对应整个 RAM
        struct kvm_dirty_log log = { .slot = 0 };
        size_t bitmap_size = (g_slave_ram_size / 4096) / 8;
        if (bitmap_size == 0) bitmap_size = 1; // Safety check

        if (!t_dirty_bitmap_cache || t_dirty_bitmap_size < bitmap_size) {
            if (t_dirty_bitmap_cache) free(t_dirty_bitmap_cache);
            t_dirty_bitmap_cache = malloc(bitmap_size);
            t_dirty_bitmap_size = bitmap_size;
        }

        if (t_dirty_bitmap_cache) {
            log.dirty_bitmap = t_dirty_bitmap_cache;
            memset(log.dirty_bitmap, 0, bitmap_size); // 必须清零，因为是复用的
        
            // 从 KVM 获取脏页位图
            if (ioctl(g_vm_fd, KVM_GET_DIRTY_LOG, &log) == 0) {
                unsigned long *p = (unsigned long *)log.dirty_bitmap;
                uint64_t num_longs = bitmap_size / sizeof(unsigned long);
            
                // 在栈上分配发送缓冲区，避免频繁 malloc
                // Header + GPA(8) + PageData(4096)
                uint8_t tx_buf[sizeof(struct gvm_header) + 8 + 4096];
                struct gvm_header *wh = (struct gvm_header *)tx_buf;
                size_t total_len = sizeof(tx_buf); // 包总长

                for (uint64_t i = 0; i < num_longs; i++) {
                if (p[i] == 0) continue; // 快速跳过无脏页的块
                
                    for (int b = 0; b < 64; b++) {
                        if ((p[i] >> b) & 1) {
                            uint64_t gpa = (i * 64 + b) * 4096;
                            if (gpa >= g_slave_ram_size) continue;
                       
                            // 1. 检查队列是否已满
                            uint32_t next_tail = (g_mpsc_tail + 1) % MPSC_QUEUE_SIZE;
                            if (next_tail == g_mpsc_head) {
                                // 队列已满，选择丢弃。这是新的丢包点。
                                // 生产环境可以加一个计数器来监控丢包率。
                                continue;
                            }

                            // 2. 分配任务内存
                            dirty_page_task_t* task = malloc(sizeof(dirty_page_task_t));
                            if (!task) {
                                // 内存分配失败，也只能放弃
                                continue;
                            }
                            
                            // 3. 填充任务数据
                            task->gpa = gpa;
                            memcpy(task->data, g_phy_ram + gpa, 4096);

                            // 4. 加锁，将任务指针放入队列，更新 tail
                            pthread_mutex_lock(&g_mpsc_lock);
                            g_mpsc_queue[g_mpsc_tail] = task;
                            g_mpsc_tail = next_tail;
                            pthread_mutex_unlock(&g_mpsc_lock);

                            // 5. 唤醒消费者线程
                            pthread_cond_signal(&g_mpsc_cond);
                        }
                    }
                }
            }
        } else {
            perror("malloc dirty bitmap");
        }
    }
    // 导出寄存器状态并回包
    ioctl(t_vcpu_fd, KVM_GET_REGS, &kregs); ioctl(t_vcpu_fd, KVM_GET_SREGS, &ksregs);
    
    struct gvm_header ack_hdr;
    memset(&ack_hdr, 0, sizeof(ack_hdr));
    ack_hdr.magic = htonl(GVM_MAGIC);              
    ack_hdr.msg_type = htons(MSG_VCPU_EXIT);       
    ack_hdr.payload_len = htons(sizeof(struct gvm_ipc_cpu_run_ack));
    ack_hdr.slave_id = htonl(hdr->slave_id);       
    ack_hdr.req_id = GVM_HTONLL(hdr->req_id);      
    
    struct gvm_ipc_cpu_run_ack *ack = (struct gvm_ipc_cpu_run_ack *)payload;
    ack->mode_tcg = 0;
    gvm_kvm_context_t *ack_kctx = &ack->ctx.kvm;
    ack_kctx->rax = kregs.rax; ack_kctx->rbx = kregs.rbx; ack_kctx->rcx = kregs.rcx; ack_kctx->rdx = kregs.rdx;
    ack_kctx->rsi = kregs.rsi; ack_kctx->rdi = kregs.rdi; ack_kctx->rsp = kregs.rsp; ack_kctx->rbp = kregs.rbp;
    ack_kctx->r8  = kregs.r8;  ack_kctx->r9  = kregs.r9;  ack_kctx->r10 = kregs.r10; ack_kctx->r11 = kregs.r11;
    ack_kctx->r12 = kregs.r12; ack_kctx->r13 = kregs.r13; ack_kctx->r14 = kregs.r14; ack_kctx->r15 = kregs.r15;
    ack_kctx->rip = kregs.rip; ack_kctx->rflags = kregs.rflags;
    memcpy(ack_kctx->sregs_data, &ksregs, sizeof(ksregs));
    ack_kctx->exit_reason = t_kvm_run->exit_reason;

    if (t_kvm_run->exit_reason == KVM_EXIT_IO) {
        ack_kctx->exit_info.io.direction = t_kvm_run->io.direction;
        ack_kctx->exit_info.io.size = t_kvm_run->io.size;
        ack_kctx->exit_info.io.port = t_kvm_run->io.port;
        ack_kctx->exit_info.io.count = t_kvm_run->io.count;
        if (t_kvm_run->io.direction == KVM_EXIT_IO_OUT) 
            memcpy(ack_kctx->exit_info.io.data, (uint8_t*)t_kvm_run + t_kvm_run->io.data_offset, t_kvm_run->io.size * t_kvm_run->io.count);
    } else if (t_kvm_run->exit_reason == KVM_EXIT_MMIO) {
        ack_kctx->exit_info.mmio.phys_addr = t_kvm_run->mmio.phys_addr;
        ack_kctx->exit_info.mmio.len = t_kvm_run->mmio.len;
        ack_kctx->exit_info.mmio.is_write = t_kvm_run->mmio.is_write;
        memcpy(ack_kctx->exit_info.mmio.data, t_kvm_run->mmio.data, 8);
    }

    uint8_t tx[sizeof(ack_hdr) + sizeof(*ack)]; memcpy(tx, &ack_hdr, sizeof(ack_hdr)); memcpy(tx+sizeof(ack_hdr), ack, sizeof(*ack));
    sendto(sockfd, tx, sizeof(tx), 0, (struct sockaddr*)client, sizeof(*client));
}

/* 
 * [物理意图] 充当远程页面的“受控提供者”，响应来自 Master/Directory 的 MESI 指令。
 * [关键逻辑] 处理 INVALIDATE（失效本地页）、DOWNGRADE（权限降级）及 READ（数据提取），操作本地物理页。
 * [后果] 保证了 Slave 侧物理内存与全网真理的一致。若响应失败，Slave 会持有一份陈旧的数据副本，导致计算错误。
 */
void handle_kvm_mem(int sockfd, struct sockaddr_in *client, struct gvm_header *hdr, void *payload) {
    uint16_t type = hdr->msg_type; 
    uint64_t gpa;

    if (type == MSG_INVALIDATE || type == MSG_DOWNGRADE) {
        gpa = GVM_NTOHLL(hdr->req_id);
    } else {
        if (hdr->payload_len < 8) return; 
        // [FIX] 安全读取 Payload 中的 GPA
        gpa = gvm_get_u64_unaligned(payload);
    }

    if (gpa >= g_slave_ram_size) return;

    if (type == MSG_MEM_READ) {
        struct gvm_header ack_hdr = { 
            .magic = htonl(GVM_MAGIC), .msg_type = htons(MSG_MEM_ACK), 
            .payload_len = htons(4096), .req_id = GVM_HTONLL(hdr->req_id) 
        };
        uint8_t tx[sizeof(ack_hdr) + 4096];
        memcpy(tx, &ack_hdr, sizeof(ack_hdr));
        memcpy(tx+sizeof(ack_hdr), g_phy_ram+gpa, 4096);
        if (robust_sendto(sockfd, tx, sizeof(tx), client) == 0) {
            //此处无需擦除
        } else {
            // 如果发不出去，让 Master 稍后重试
            //fprintf(stderr, "[GVM-Slave] Critical: Failed to send back data for GPA %lx, aborting unmap\n", gpa);
        }
    } 
    else if (type == MSG_MEM_WRITE) {
        if (hdr->payload_len >= 8+4096) {
            memcpy(g_phy_ram+gpa, (uint8_t*)payload+8, 4096);
        }
    }
    else if (type == MSG_INVALIDATE) {
        madvise(g_phy_ram + gpa, 4096, MADV_DONTNEED);
    }
    else if (type == MSG_DOWNGRADE) {
        if (hdr->payload_len < 16) return;
        
        // [FIX] 安全读取 Payload 中的复杂数据
        uint64_t requester_u64 = gvm_get_u64_unaligned(payload);
        uint64_t orig_req_id;
        memcpy(&orig_req_id, (uint8_t*)payload + 8, 8); // 已经是网络序，直接考

        uint32_t target_node = (uint32_t)requester_u64;

        struct gvm_header wb_hdr = {
            .magic = htonl(GVM_MAGIC), .msg_type = htons(MSG_WRITE_BACK),
            .payload_len = htons(8 + 4096), 
            .slave_id = htonl(target_node), 
            .req_id = orig_req_id,          
            .qos_level = 0
        };
        
        uint8_t tx[sizeof(wb_hdr) + 8 + 4096];
        memcpy(tx, &wb_hdr, sizeof(wb_hdr));
        *(uint64_t*)(tx + sizeof(wb_hdr)) = GVM_HTONLL(gpa);
        memcpy(tx + sizeof(wb_hdr) + 8, g_phy_ram + gpa, 4096);
        
        if (robust_sendto(sockfd, tx, sizeof(tx), client) == 0) {
            // 只有确保网络层已经接纳了这个包，才敢擦除本地物理内存
            madvise(g_phy_ram + gpa, 4096, MADV_DONTNEED);
        } else {
            // 如果发不出去，宁可让 Master 稍后重试，也不要擦除数据
            //fprintf(stderr, "[GVM-Slave] Critical: Failed to send back data for GPA %lx, aborting unmap\n", gpa);
        }
    }
    else if (type == MSG_FETCH_AND_INVALIDATE) {
        // [FIX] 安全读取
        uint64_t tmp_target = gvm_get_u64_unaligned(payload);
        uint32_t target_node = (uint32_t)tmp_target;
        
        uint64_t orig_req_id;
        memcpy(&orig_req_id, (uint8_t*)payload + 8, 8);

        if (gpa < g_slave_ram_size) {
            struct gvm_header wb_hdr;
            wb_hdr.magic = htonl(GVM_MAGIC);
            wb_hdr.msg_type = htons(MSG_WRITE_BACK);
            wb_hdr.payload_len = htons(8 + 4096);
            wb_hdr.slave_id = htonl(target_node);
            wb_hdr.req_id = orig_req_id;
            wb_hdr.qos_level = 0;
            
            uint8_t tx[sizeof(struct gvm_header) + 8 + 4096];
            memcpy(tx, &wb_hdr, sizeof(wb_hdr));
            
            uint64_t net_gpa = GVM_HTONLL(gpa);
            memcpy(tx + sizeof(wb_hdr), &net_gpa, 8);
            memcpy(tx + sizeof(wb_hdr) + 8, g_phy_ram + gpa, 4096);
            
            if (robust_sendto(sockfd, tx, sizeof(tx), client) == 0) {
                // 同上
                madvise(g_phy_ram + gpa, 4096, MADV_DONTNEED);
            } else {
                //fprintf(stderr, "[GVM-Slave] Critical: Failed to send back data for GPA %lx, aborting unmap\n", gpa);
            }
        }
    }
}

/* 
 * [物理意图] Slave 节点的“任务分发中心”，负责监听并调度来自全网的 RPC 指令。
 * [关键逻辑] 绑定多播/复用端口，利用 recvmmsg 批量捕获包头，根据消息类型分流至执行引擎或内存处理程序。
 * [后果] 它是 Slave 节点的总入口。其效率直接决定了 Slave 对 Master 请求的“首包响应时间（TTFB）”。
 */
void* kvm_worker_thread(void *arg) {
    long core = (long)arg;
    int s = socket(AF_INET, SOCK_DGRAM, 0); int opt=1; setsockopt(s, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt));
    struct sockaddr_in a = { .sin_family=AF_INET, .sin_addr.s_addr=htonl(INADDR_ANY), .sin_port=htons(g_service_port) };
    bind(s, (struct sockaddr*)&a, sizeof(a));
    
    // Worker 0 初始化 VFIO
    if (core == 0 && g_vfio_config_path) {
        gvm_vfio_init(g_vfio_config_path);
    }
    
    struct mmsghdr msgs[BATCH_SIZE]; struct iovec iov[BATCH_SIZE]; uint8_t bufs[BATCH_SIZE][4200]; struct sockaddr_in c[BATCH_SIZE];
    for(int i=0;i<BATCH_SIZE;i++) { iov[i].iov_base=bufs[i]; iov[i].iov_len=4200; msgs[i].msg_hdr.msg_iov=&iov[i]; msgs[i].msg_hdr.msg_iovlen=1; msgs[i].msg_hdr.msg_name=&c[i]; msgs[i].msg_hdr.msg_namelen=sizeof(c[i]); }

    while(1) {
        int n = recvmmsg(s, msgs, BATCH_SIZE, 0, NULL);
        if (n<=0) continue;
        for(int i=0;i<n;i++) {
            struct gvm_header *h = (struct gvm_header*)bufs[i];
            if (h->magic != htonl(GVM_MAGIC)) continue;
            
            pthread_spin_lock(&g_master_lock);
            if (g_master_addr.sin_port != c[i].sin_port || g_master_addr.sin_addr.s_addr != c[i].sin_addr.s_addr) {
                g_master_addr = c[i];
                pthread_mutex_lock(&g_master_mutex);
                if (!g_master_ready) {
                    g_master_ready = 1;
                    pthread_cond_broadcast(&g_master_cond);
                }
                pthread_mutex_unlock(&g_master_mutex);
            }
            g_master_addr = c[i]; 
            pthread_spin_unlock(&g_master_lock); 

            uint16_t type = ntohs(h->msg_type);
            ntoh_header(h);

            if (type == MSG_VCPU_RUN) {
                // [FIX] 必须重建 IPC 请求结构体，不能直接强转 payload
                // 因为 payload 里只有 Context 数据，没有 IPC 头部的 slave_id 等字段
                struct gvm_ipc_cpu_run_req local_req;
    
                // 清空结构体，防止垃圾数据影响逻辑
                memset(&local_req, 0, sizeof(local_req));

                // 1. 从网络包头提取元数据
                // QEMU 发送端将这些信息放在了 gvm_header 中
                local_req.mode_tcg = h->mode_tcg; 
                // slave_id 和 vcpu_index 在 stateless 模式下通常由调度器指定
                // 这里我们直接透传包头里的 source id 作为请求方
                local_req.slave_id = ntohl(h->slave_id); 

                // 2. 从 Payload 提取 Context
                // 指针 arithmetic: bufs[i] 是包头起始，+sizeof(*h) 是 payload 起始
                void *net_payload_ptr = bufs[i] + sizeof(struct gvm_header);
    
                if (local_req.mode_tcg) {
                    // 安全检查：防止 payload 长度不足导致越界
                    if (ntohs(h->payload_len) >= sizeof(gvm_tcg_context_t)) {
                        memcpy(&local_req.ctx.tcg, net_payload_ptr, sizeof(gvm_tcg_context_t));
                    }
                } else {
                    if (ntohs(h->payload_len) >= sizeof(gvm_kvm_context_t)) {
                        memcpy(&local_req.ctx.kvm, net_payload_ptr, sizeof(gvm_kvm_context_t));
                    }
                }

                // 3. 调用核心执行函数
                // 传递栈上构造的 local_req 指针
                handle_kvm_run_stateless(s, &c[i], h, &local_req, (int)core);
            }
            else handle_kvm_mem(s, &c[i], h, bufs[i]+sizeof(*h));
        }
    }
}

// ==========================================
// [Fixed Path] TCG Proxy Engine (Tri-Channel)
// ==========================================

// 三通道地址表
typedef struct {
    struct sockaddr_in cmd_addr;  // 控制流
    struct sockaddr_in req_addr;  // 内存请求 (Slave -> Master -> Slave)
    struct sockaddr_in push_addr; // 内存推送 (Master -> Slave)
} slave_endpoint_t;

static slave_endpoint_t *tcg_endpoints = NULL; 
static struct sockaddr_in g_upstream_gateway = {0};
static volatile int g_gateway_init_done = 0;
static volatile int g_gateway_known = 0;
static int g_base_id = 0; 

/* 
 * [物理意图] 在不支持 KVM 的环境下，通过多进程模拟实现“多核算力聚合”。
 * [关键逻辑] 为每个逻辑核心孵化一个独立的 QEMU-TCG 实例，并通过环境变量注入“三通道” Socket 句柄。
 * [后果] 提供了极致的兼容性。即便是在廉价的 ARM 树莓派或锁定内核的容器里，也能作为 Slave 贡献算力。
 */
void spawn_tcg_processes(int base_id) {
    printf("[Hybrid] Spawning %ld QEMU-TCG instances (Tri-Channel Isolation)...\n", g_num_cores);
    
    tcg_endpoints = malloc(sizeof(slave_endpoint_t) * g_num_cores);
    if (!tcg_endpoints) { perror("malloc endpoints"); exit(1); }
    
    int internal_base = 20000 + (g_service_port % 1000) * 256;
    char ram_str[32]; snprintf(ram_str, sizeof(ram_str), "%d", g_ram_mb);

    for (long i = 0; i < g_num_cores; i++) {
        int port_cmd  = internal_base + i * 3 + 0;
        int port_req  = internal_base + i * 3 + 1;
        int port_push = internal_base + i * 3 + 2;
        
        tcg_endpoints[i].cmd_addr.sin_family = AF_INET;
        tcg_endpoints[i].cmd_addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        tcg_endpoints[i].cmd_addr.sin_port = htons(port_cmd);

        tcg_endpoints[i].req_addr.sin_family = AF_INET;
        tcg_endpoints[i].req_addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        tcg_endpoints[i].req_addr.sin_port = htons(port_req);

        tcg_endpoints[i].push_addr.sin_family = AF_INET;
        tcg_endpoints[i].push_addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        tcg_endpoints[i].push_addr.sin_port = htons(port_push);

        if (fork() == 0) {
            int sock_cmd = socket(AF_INET, SOCK_DGRAM, 0);
            struct sockaddr_in addr_cmd = { .sin_family=AF_INET, .sin_addr.s_addr=htonl(INADDR_LOOPBACK), .sin_port=htons(port_cmd) };
            bind(sock_cmd, (struct sockaddr*)&addr_cmd, sizeof(addr_cmd));

            int sock_req = socket(AF_INET, SOCK_DGRAM, 0);
            struct sockaddr_in addr_req = { .sin_family=AF_INET, .sin_addr.s_addr=htonl(INADDR_LOOPBACK), .sin_port=htons(port_req) };
            bind(sock_req, (struct sockaddr*)&addr_req, sizeof(addr_req));

            int sock_push = socket(AF_INET, SOCK_DGRAM, 0);
            struct sockaddr_in addr_push = { .sin_family=AF_INET, .sin_addr.s_addr=htonl(INADDR_LOOPBACK), .sin_port=htons(port_push) };
            bind(sock_push, (struct sockaddr*)&addr_push, sizeof(addr_push));

            struct sockaddr_in proxy = { .sin_family=AF_INET, .sin_addr.s_addr=htonl(INADDR_LOOPBACK), .sin_port=htons(g_service_port) };
            connect(sock_cmd, (struct sockaddr*)&proxy, sizeof(proxy));
            connect(sock_req, (struct sockaddr*)&proxy, sizeof(proxy));
            connect(sock_push, (struct sockaddr*)&proxy, sizeof(proxy));

            char fd_c[16], fd_r[16], fd_p[16];
            int f;
            f=fcntl(sock_cmd, F_GETFD); f&=~FD_CLOEXEC; fcntl(sock_cmd, F_SETFD, f);
            f=fcntl(sock_req, F_GETFD); f&=~FD_CLOEXEC; fcntl(sock_req, F_SETFD, f);
            f=fcntl(sock_push, F_GETFD); f&=~FD_CLOEXEC; fcntl(sock_push, F_SETFD, f);
            
            snprintf(fd_c, 16, "%d", sock_cmd);
            snprintf(fd_r, 16, "%d", sock_req);
            snprintf(fd_p, 16, "%d", sock_push);

            setenv("GVM_SOCK_CMD", fd_c, 1);
            setenv("GVM_SOCK_REQ", fd_r, 1);  
            setenv("GVM_SOCK_PUSH", fd_p, 1); 
            setenv("GVM_ROLE", "SLAVE", 1);
            char id_str[32];
            snprintf(id_str, sizeof(id_str), "%ld", base_id + i); 
            setenv("GVM_SLAVE_ID", id_str, 1); 

            const char *shm_path = getenv("GVM_SHM_FILE");
            if (shm_path) {
                setenv("GVM_SHM_FILE", shm_path, 1);
            }

            cpu_set_t cpuset; CPU_ZERO(&cpuset); CPU_SET(i, &cpuset); sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
            
            execlp("qemu-system-x86_64", "qemu-system-x86_64", 
                   "-accel", "tcg,thread=single", 
                   "-m", ram_str, 
                   "-nographic", "-S", "-nodefaults", 
                   "-icount", "shift=5,sleep=off", NULL);
            exit(1);
        }
        close(sock_cmd);
        close(sock_req);
        close(sock_push);
    }
}

/* 
 * [物理意图] 充当 TCG 实例与 P2P 网络之间的“物理层协议翻译官”。
 * [关键逻辑] 维护 CMD（控制）、REQ（同步请求）、PUSH（异步推送）三通道的路由，并实现 VFIO 的本地拦截。
 * [后果] 解决了模拟模式下的网络竞争问题，通过物理通道隔离，保证了高频内存同步不会阻塞关键的 CPU 控制指令。
 */
void* tcg_proxy_thread(void *arg) {
    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    int opt = 1; setsockopt(sockfd, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt));
    struct sockaddr_in addr = { .sin_family=AF_INET, .sin_addr.s_addr=htonl(INADDR_ANY), .sin_port=htons(g_service_port) };
    bind(sockfd, (struct sockaddr*)&addr, sizeof(addr));

    struct mmsghdr msgs[BATCH_SIZE]; struct iovec iovecs[BATCH_SIZE]; uint8_t buffers[BATCH_SIZE][4200]; struct sockaddr_in src_addrs[BATCH_SIZE];
    memset(msgs, 0, sizeof(msgs));
    for(int i=0;i<BATCH_SIZE;i++) { iovecs[i].iov_base=buffers[i]; iovecs[i].iov_len=4200; msgs[i].msg_hdr.msg_iov=&iovecs[i]; msgs[i].msg_hdr.msg_iovlen=1; msgs[i].msg_hdr.msg_name=&src_addrs[i]; msgs[i].msg_hdr.msg_namelen=sizeof(src_addrs[i]); }

    printf("[Proxy] Tri-Channel NAT Active (CMD/REQ/PUSH) + MESI Support.\n");

    while(1) {
        int n = recvmmsg(sockfd, msgs, BATCH_SIZE, 0, NULL);
        if (n <= 0) continue;

        for (int i=0; i<n; i++) {
            struct gvm_header *hdr = (struct gvm_header *)buffers[i];
            if (hdr->magic != htonl(GVM_MAGIC)) continue;

            // 1. Upstream (Local QEMU -> Gateway)
            if (src_addrs[i].sin_addr.s_addr == htonl(INADDR_LOOPBACK)) {
                // [VFIO Intercept] TCG 模式下的本地显卡拦截
                uint16_t msg_type = ntohs(hdr->msg_type);
                if (msg_type == MSG_MEM_WRITE) {
                    uint64_t gpa = GVM_NTOHLL(*(uint64_t*)(buffers[i] + sizeof(struct gvm_header)));
                    void *data = buffers[i] + sizeof(struct gvm_header) + 8;
                    int len = ntohs(hdr->payload_len) - 8;
                    if (gvm_vfio_intercept_mmio(gpa, data, len, 1)) {
                        hdr->msg_type = htons(MSG_MEM_ACK);
                        hdr->payload_len = 0;
                        sendto(sockfd, buffers[i], sizeof(struct gvm_header), 0, 
                               (struct sockaddr*)&src_addrs[i], sizeof(struct sockaddr_in));
                        continue; 
                    }
                }
                
                if (g_gateway_known) 
                    sendto(sockfd, buffers[i], msgs[i].msg_len, 0, (struct sockaddr*)&g_upstream_gateway, sizeof(struct sockaddr_in));
            }
            // 2. Downstream (Gateway -> Local QEMU)
            else {
                if (!g_gateway_init_done) {
                    if (__sync_bool_compare_and_swap(&g_gateway_init_done, 0, 1)) {
                        memcpy(&g_upstream_gateway, &src_addrs[i], sizeof(struct sockaddr_in));
                        g_gateway_known = 1;
                        printf("[Proxy] Gateway Locked: %s:%d\n", inet_ntoa(src_addrs[i].sin_addr), ntohs(src_addrs[i].sin_port));
                    }
                }
                
                uint32_t slave_id = ntohl(hdr->slave_id);
                uint16_t msg_type = ntohs(hdr->msg_type);
                int core_idx = (int)(slave_id - g_base_id);
    
                if (core_idx < 0 || core_idx >= g_num_cores) continue;

                // [V28 分流逻辑升级]
                if (msg_type == MSG_MEM_WRITE || msg_type == MSG_MEM_READ ||
                    msg_type == MSG_INVALIDATE || msg_type == MSG_DOWNGRADE || 
                    msg_type == MSG_FETCH_AND_INVALIDATE ||
                    msg_type == MSG_PAGE_PUSH_FULL || 
                    msg_type == MSG_PAGE_PUSH_DIFF || 
                    msg_type == MSG_FORCE_SYNC) {
                     sendto(sockfd, buffers[i], msgs[i].msg_len, 0, 
                          (struct sockaddr*)&tcg_endpoints[core_idx].push_addr, sizeof(struct sockaddr_in));
                }
                else if (msg_type == MSG_MEM_ACK) {
                    // 如果 req_id 是 ~0ULL，说明是异步回包，也走 PUSH
                    if (GVM_NTOHLL(hdr->req_id) == ~0ULL)
                        sendto(sockfd, buffers[i], msgs[i].msg_len, 0, 
                              (struct sockaddr*)&tcg_endpoints[core_idx].push_addr, sizeof(struct sockaddr_in));
                    else
                        sendto(sockfd, buffers[i], msgs[i].msg_len, 0, 
                              (struct sockaddr*)&tcg_endpoints[core_idx].req_addr, sizeof(struct sockaddr_in));
                }
                else {
                    sendto(sockfd, buffers[i], msgs[i].msg_len, 0, 
                          (struct sockaddr*)&tcg_endpoints[core_idx].cmd_addr, sizeof(struct sockaddr_in));
                }
            }
        }
    }
    return NULL;
}

int main(int argc, char **argv) {
    g_num_cores = get_allowed_cores();
    if (argc >= 2) g_service_port = atoi(argv[1]);
    if (argc >= 3) {
        g_num_cores = atoi(argv[2]);
        if (g_num_cores <= 0) g_num_cores = 1;
    }
    if (argc >= 4) { g_ram_mb = atoi(argv[3]); if(g_ram_mb<=0) g_ram_mb=1024; g_slave_ram_size = (uint64_t)g_ram_mb * 1024 * 1024; }
    if (argc >= 5) {
        g_base_id = atoi(argv[4]);
    }

    printf("[Init] GiantVM Hybrid Slave V28.0 (Swarm Edition)\n");
    printf("[Init] Config: Port=%d, Cores=%ld, RAM=%d MB, BaseID=%d\n", 
           g_service_port, g_num_cores, g_ram_mb, g_base_id);
    
    // 解析 -vfio 参数
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-vfio") == 0 && i + 1 < argc) {
            g_vfio_config_path = argv[i+1];
        }
    }
    init_kvm_global();

    if (g_kvm_available) {
        printf("[Hybrid] Mode: KVM FAST PATH. Listening on 0.0.0.0:%d\n", g_service_port);
        pthread_t sender_thread_id;
        if (pthread_create(&sender_thread_id, NULL, dirty_sync_sender_thread, NULL) != 0) {
            perror("Failed to create dirty page sender thread");
            return 1;
        }
        pthread_detach(sender_thread_id); // 设为分离模式，不需 join
        // 启动 VFIO IRQ 转发线程
        if (g_vfio_config_path) {
            pthread_t irq_th;
            pthread_create(&irq_th, NULL, vfio_irq_thread_adapter, NULL);
        }
        pthread_t *threads = malloc(sizeof(pthread_t) * g_num_cores);
        for(long i=0; i<g_num_cores; i++) pthread_create(&threads[i], NULL, kvm_worker_thread, (void*)i);
        for(long i=0; i<g_num_cores; i++) pthread_join(threads[i], NULL);
    } else {
        printf("[Hybrid] Mode: TCG PROXY (Tri-Channel). Listening on 0.0.0.0:%d\n", g_service_port);
        spawn_tcg_processes(g_base_id);
        sleep(1);
        int proxy_threads = g_num_cores / 2; 
        if (proxy_threads < 1) proxy_threads = 1;
        pthread_t *threads = malloc(sizeof(pthread_t) * proxy_threads);
        for(long i=0; i<proxy_threads; i++) pthread_create(&threads[i], NULL, tcg_proxy_thread, NULL);
        while(wait(NULL) > 0);
    }
    return 0;
}

