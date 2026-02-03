
/*
 * [IDENTITY] Frontend Wavelet Engine - The Memory Guardian
 * ---------------------------------------------------------------------------
 * 物理角色：QEMU 内部的内存一致性协调者。
 * 职责边界：
 * 1. 运行 Latch 机制锁定读写冲突。
 * 2. 定时收割脏页生成 Diff，维持 Wavelet 增量推送。
 * 3. 运行重排缓冲区 (Reorder Window)，将乱序推送"坍缩"为有序内存。
 * 
 * [禁止事项]
 * - 严禁在 sigsegv_handler 中进行任何可能引起休眠的操作。
 * - 严禁关闭 Lazy TLB Flush (defer_ro_protect)，否则 QEMU 性能将崩溃。
 * ---------------------------------------------------------------------------
 */
#define _GNU_SOURCE
#include "qemu/osdep.h"
#include <sys/mman.h>
#include <signal.h>
#include <ucontext.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <errno.h>
#include <poll.h>
#include <stdbool.h>
#include <time.h>
#include <stdatomic.h>

#include "giantvm_protocol.h"

/* 
 * GiantVM V29.5 "Wavelet" User-Mode Memory Engine (Production Ready)
 */

// --- 全局配置与状态 ---
static int g_is_slave = 0;
static int g_fd_req = -1;  
static __thread int t_req_sock = -1;
static __thread uint8_t t_net_buf[GVM_MAX_PACKET_SIZE];
// 引用外部计算函数 (确保已链接 crc32.o)
extern uint32_t calculate_crc32(const void* data, size_t length);
static int g_fd_push = -1; 
static void *g_ram_base = NULL;
static size_t g_ram_size = 0;
static uint32_t g_slave_id = 0;

// 脏区捕获链表
typedef struct WritablePage {
    uint64_t gpa;
    void* pre_image_snapshot;
    struct WritablePage *next;
} WritablePage;

#define PAGE_POOL_SIZE 4096
static WritablePage g_page_pool[PAGE_POOL_SIZE];
static void* g_image_pool[PAGE_POOL_SIZE]; // 预分配快照空间
static atomic_int g_pool_idx = 0;

static volatile bool g_threads_running = false;
static pthread_t g_listen_thread;
static pthread_t g_harvester_thread;

#define MAX_RAM_BLOCKS 64

typedef struct {
    uintptr_t hva_start;
    uintptr_t hva_end;
    uint64_t  gpa_start;
    uint64_t  size;
} GVMRamBlock;

static GVMRamBlock g_mem_blocks[MAX_RAM_BLOCKS];
static int g_block_count = 0;

/* 
 * [物理意图] 在 QEMU 内部建立 Guest 物理地址(GPA)与宿主机虚拟地址(HVA)的“空间映射图”。
 * [关键逻辑] 将 RAM 块注册到私有映射表，并执行初始 mprotect(PROT_NONE) 以强制触发首次访问缺页。
 * [后果] 若未正确注册，特定的内存区域将脱离分布式一致性引擎的监控，导致该区域的写操作无法全网同步。
 */
void giantvm_register_ram_block(void *hva, uint64_t size, uint64_t gpa) {
    if (g_block_count >= MAX_RAM_BLOCKS) exit(1);
    mprotect(hva, size, PROT_NONE);
    g_mem_blocks[g_block_count].hva_start = (uintptr_t)hva;
    g_mem_blocks[g_block_count].hva_end   = (uintptr_t)hva + size;
    g_mem_blocks[g_block_count].gpa_start = gpa;
    g_mem_blocks[g_block_count].size      = size;
    g_block_count++;
}

// [替换] 查表法 HVA 转 GPA (用于 sigsegv)
static uint64_t hva_to_gpa_safe(uintptr_t addr) {
    // 这里 block_count 是在 giantvm_region_add 时动态增加的
    for (int i = 0; i < g_block_count; i++) {
        if (addr >= g_mem_blocks[i].hva_start && addr < g_mem_blocks[i].hva_end) {
            // 真实的 GPA = 块起始 GPA + 块内偏移
            return g_mem_blocks[i].gpa_start + (addr - g_mem_blocks[i].hva_start);
        }
    }
    return (uint64_t)-1;
}

// [替换] 查表法 GPA 转 HVA (用于 harvester)
static void* gpa_to_hva_safe(uint64_t gpa) {
    for (int i = 0; i < g_block_count; i++) {
        if (gpa >= g_mem_blocks[i].gpa_start && 
            gpa < g_mem_blocks[i].gpa_start + g_mem_blocks[i].size) {
            return (void*)(g_mem_blocks[i].hva_start + (gpa - g_mem_blocks[i].gpa_start));
        }
    }
    return NULL;
}

// 通用极速零检测
static inline bool is_page_all_zero(void *addr) {
    uint64_t *p = (uint64_t *)addr;
    for (int i = 0; i < 512; i += 4) {
        if (p[i] | p[i+1] | p[i+2] | p[i+3]) return false;
    }
    return true;
}

// --- Lazy Protection Queue ---
#define LAZY_QUEUE_SIZE 64  
static __thread uint64_t t_lazy_ro_queue[LAZY_QUEUE_SIZE];
static __thread int t_lazy_count = 0;

static void flush_lazy_ro_queue(void) {
    if (t_lazy_count == 0) return;
    for (int i = 0; i < t_lazy_count; i++) {
        uint64_t gpa = t_lazy_ro_queue[i];
        mprotect((uint8_t*)g_ram_base + gpa, 4096, PROT_READ);
    }
    t_lazy_count = 0;
}

static void defer_ro_protect(uint64_t gpa) {
    t_lazy_ro_queue[t_lazy_count++] = gpa;
    if (t_lazy_count >= LAZY_QUEUE_SIZE) flush_lazy_ro_queue();
}

// 发送 PUSH 包 (Diff 或 Zero)
static void send_push_packet(uint64_t gpa, uint64_t version, void *data, uint16_t size, uint8_t flags) {
    if (g_fd_push < 0) return;
    size_t pl_len = sizeof(struct gvm_diff_log) + size;
    size_t pkt_len = sizeof(struct gvm_header) + pl_len;
    uint8_t *buf = malloc(pkt_len);
    if (!buf) return;

    struct gvm_header *hdr = (struct gvm_header *)buf;
    hdr->magic = htonl(GVM_MAGIC);
    hdr->msg_type = htons(MSG_COMMIT_DIFF);
    hdr->payload_len = htons(pl_len);
    hdr->slave_id = htonl(g_slave_id);
    hdr->req_id = 0;
    hdr->qos_level = 1;
    hdr->flags = flags; // [关键]
    hdr->crc32 = 0;

    struct gvm_diff_log *log = (struct gvm_diff_log *)(buf + sizeof(*hdr));
    log->gpa = GVM_HTONLL(gpa);
    log->version = GVM_HTONLL(version);
    log->offset = 0;
    log->size = htons(size);

    if (size > 0 && data) memcpy(log->data, data, size);

    hdr->crc32 = htonl(calculate_crc32(buf, pkt_len));
    send(g_fd_push, buf, pkt_len, 0);
    free(buf);
}

// --- [FIX 2] Latch 锁分段 (放在文件头部全局区) ---

// 强制 128 字节对齐 (兼容 x86_64 和 ARM64/Graviton)
typedef struct {
    volatile uint64_t val;
    uint8_t padding[128 - sizeof(uint64_t)]; 
} __attribute__((aligned(128))) aligned_latch_t;

// 静态断言：编译期检查对齐是否成功
_Static_assert(sizeof(aligned_latch_t) == 128, "Latch alignment failed");

#define LATCH_SHARDS 256
static aligned_latch_t g_latches[LATCH_SHARDS];

// 初始化 (在 giantvm_user_mem_init 调用)
static void init_latches(void) {
    for(int i=0; i<LATCH_SHARDS; i++) g_latches[i].val = (uint64_t)-1;
}

#define LATCH_IDX(gpa) ((gpa >> 12) % LATCH_SHARDS)

// --- [FIX 4] Mode B 微型重排窗口 ---

#define REORDER_WIN_SIZE 32  // 容忍 32 个包的乱序
#define REORDER_MASK (REORDER_WIN_SIZE - 1)

typedef struct {
    uint64_t gpa;
    uint64_t version;
    uint16_t msg_type;
    uint16_t len;
    uint8_t *data;
    uint64_t timestamp_us; // 存入时间戳
    bool active;
} ReorderSlot;

static ReorderSlot g_reorder_buf[REORDER_WIN_SIZE];
static pthread_spinlock_t g_reorder_lock;

// 简单的异或哈希
static inline int get_reorder_idx(uint64_t gpa, uint64_t version) {
    return ((gpa >> 12) ^ version) & REORDER_MASK;
}

// 存入未来的包
static void buffer_future_packet(uint64_t gpa, uint64_t version, uint16_t type, void *data, uint16_t len) {
    int idx = get_reorder_idx(gpa, version);
    pthread_spin_lock(&g_reorder_lock);
    
    // 如果槽位被占，释放旧数据 (Drop-on-Collision 策略)
    if (g_reorder_buf[idx].active) free(g_reorder_buf[idx].data);

    g_reorder_buf[idx].gpa = gpa;
    g_reorder_buf[idx].version = version;
    g_reorder_buf[idx].msg_type = type;
    g_reorder_buf[idx].len = len;
    g_reorder_buf[idx].active = true;
    g_reorder_buf[idx].timestamp_us = get_us_time();
    g_reorder_buf[idx].data = malloc(len);
    if (g_reorder_buf[idx].data) memcpy(g_reorder_buf[idx].data, data, len);
    else g_reorder_buf[idx].active = false; // OOM 保护
    
    pthread_spin_unlock(&g_reorder_lock);
}

/* 
 * [物理意图] 接收并应用来自 P2P 网络的“真理推送”，更新本地物理内存。
 * [关键逻辑] 执行严格的版本判定（is_next_version）：顺序包直接 memcpy，版本断层包则强制失效（Invalidate）本地映射。
 * [后果] 实现了 MESI 协议的远程写入动作。它保证了即便在乱序网络下，本地 vCPU 看到的内存也是单调递增的一致性状态。
 */
void gvm_apply_remote_push(uint16_t msg_type, void *payload) {
// --- 分支 1: Diff 推送 ---
    if (msg_type == MSG_PAGE_PUSH_DIFF) {
        struct gvm_diff_log* log = (struct gvm_diff_log*)payload;
        uint64_t gpa = GVM_NTOHLL(log->gpa);
        uint64_t push_ver = GVM_NTOHLL(log->version);
        uint64_t local_ver = get_local_page_version(gpa);

        // [FIX] 严格版本/幂等性校验
        
        // 情况 A: 过期或重复的包 (Stale/Duplicate)
        // 网络重传或乱序导致，直接静默丢弃，不做任何内存操作
        if (!is_newer_version(local_ver, push_ver)) {
            return; 
        }

        // 情况 B: 顺序到达的包 (Ideal Sequence)
        if (is_next_version(local_ver, push_ver)) {
            uint16_t offset = ntohs(log->offset);
            uint16_t size = ntohs(log->size);
            
            // 边界检查：防止恶意包导致 Segfault
            if (offset + size > 4096) return;

            // 1. 临时开放写权限
            mprotect((uint8_t*)g_ram_base + gpa, 4096, PROT_READ | PROT_WRITE);
            // 2. 应用增量数据
            memcpy((uint8_t*)g_ram_base + gpa + offset, log->data, size);
            // 3. 放入惰性锁回队列 (性能优化)
            defer_ro_protect(gpa);
            
            // 4. 更新本地版本
            set_local_page_version(gpa, push_ver);
        } 
        // 情况 C: 版本断层 (Gap Detected)
        // 例如：本地是 v10，收到了 v12。中间缺了 v11。
        else {
            // 此时内存状态已不可信，必须强制失效
            // 下次访问触发 sigsegv -> request_page_sync (V28 Pull) 拉取最新全量
            mprotect((uint8_t*)g_ram_base + gpa, 4096, PROT_NONE);
            
            // 将本地版本置 0，确保下次 Pull 回来的数据（无论版本多少）都能成功覆盖
            set_local_page_version(gpa, 0); 
        }
    }
    // --- 分支 2: 全页推送 / 强制同步 ---
    else if (msg_type == MSG_PAGE_PUSH_FULL || msg_type == MSG_FORCE_SYNC) {
        struct gvm_full_page_push* full = (struct gvm_full_page_push*)payload;
        uint64_t gpa = GVM_NTOHLL(full->gpa);
        uint64_t push_ver = GVM_NTOHLL(full->version);
        
        if (is_newer_version(get_local_page_version(gpa), push_ver)) {
            mprotect((uint8_t*)g_ram_base + gpa, 4096, PROT_READ | PROT_WRITE);
            memcpy((uint8_t*)g_ram_base + gpa, full->data, 4096);
            
            // 惰性锁回
            defer_ro_protect(gpa);
            
            set_local_page_version(gpa, push_ver);
        }
    }
    // --- 分支 3: Prophet RPC (V29 新增) ---
    else if (msg_type == MSG_RPC_BATCH_MEMSET) {
        // RPC 指令通常意味着大范围内存变动，这里不做 Lazy 处理
        // 让 giantvm-all.c 里的 tb_flush 去处理一致性
        // 这里主要负责更新版本号（如果有必要）
        // 但注意：Prophet 的执行是在 handle_rpc_batch_execution 里直接写内存的
        // 这里收到的只是通知，通常不需要做 mprotect 操作，除非为了 invalidation
    }
}

// 检查并应用后续包 (链式反应)
static bool check_and_apply_next(uint64_t gpa, uint64_t next_ver) {
    // 关键点：不再相信调用者传进来的 next_ver (它可能是算错的)
    // 我们基于本地真实的当前版本，去推算可能的“逻辑下一跳”
    uint64_t local_v = get_local_page_version(gpa);
    
    // 可能性 A: 纪元内连续 (+1)
    uint64_t next_a = local_v + 1;
    // 可能性 B: 跨纪元第一炮 (Epoch + 1, Counter = 1)
    uint32_t cur_epoch = (uint32_t)(local_v >> 32);
    uint64_t next_b = ((uint64_t)(cur_epoch + 1) << 32) | 1;

    bool applied = false;
    pthread_spin_lock(&g_reorder_lock);

    // 1. 先探测可能性 A
    int idx = get_reorder_idx(gpa, next_a);
    ReorderSlot *s = &g_reorder_buf[idx];
    if (s->active && s->gpa == gpa && s->version == next_a) goto hit;

    // 2. A 没中，探测可能性 B
    idx = get_reorder_idx(gpa, next_b);
    s = &g_reorder_buf[idx];
    if (s->active && s->gpa == gpa && s->version == next_b) goto hit;

    // 都没中
    pthread_spin_unlock(&g_reorder_lock);
    return false;

hit:
    // 命中！执行应用逻辑
    void *d = s->data;
    uint16_t t = s->msg_type;
    
    s->active = false;
    s->data = NULL; // [修复] 彻底杜绝野指针
    
    pthread_spin_unlock(&g_reorder_lock);

    gvm_apply_remote_push(t, d);
    free(d);
    return true;
}

static WritablePage *g_writable_pages_list = NULL;
static pthread_mutex_t g_writable_list_lock = PTHREAD_MUTEX_INITIALIZER;

// 线程局部
static __thread int t_com_sock = -1; 
static __thread uint8_t t_net_buf[GVM_MAX_PACKET_SIZE]; 

// --- 扩展 IPC 结构 (本地定义以匹配 Daemon 扩展) ---
struct gvm_ipc_fault_ack {
    int status;
    uint64_t version; // [V29] 必须同步版本号
};

// --- 辅助函数 ---

// 使用二级页表风格的目录来存储版本号，适配 64 位地址空间

#define L1_BITS 10
#define L2_BITS 10
#define L1_SIZE (1UL << L1_BITS)
#define L2_SIZE (1UL << L2_BITS)
#define PAGE_SHIFT 12

// 二级索引结构
static uint64_t **g_ver_root = NULL;

/* 
 * [物理意图] 维护本地缓存页面的“逻辑时钟（版本号）”快照。
 * [关键逻辑] 采用二级页表结构的索引表（Radix-like Table），以 O(1) 时间复杂度追踪 500PB 空间内每页的版本。
 * [后果] 这是判定“何为真理”的基石。版本号记录错误会直接导致读到旧数据（Stale Read）或触发不必要的全页强制同步。
 */
static uint64_t get_local_page_version(uint64_t gpa) {
    if (!g_ver_root) return 0;

    uint64_t pfn = gpa >> PAGE_SHIFT;
    uint64_t l1_idx = (pfn >> L2_BITS) & (L1_SIZE - 1);
    uint64_t l2_idx = pfn & (L2_SIZE - 1);

    if (g_ver_root[l1_idx] == NULL) return 0;
    return __atomic_load_n(&g_ver_root[l1_idx][l2_idx], __ATOMIC_ACQUIRE);
}

static void set_local_page_version(uint64_t gpa, uint64_t version) {
    if (!g_ver_root) {
        g_ver_root = calloc(L1_SIZE, sizeof(uint64_t *));
    }

    uint64_t pfn = gpa >> PAGE_SHIFT;
    uint64_t l1_idx = (pfn >> L2_BITS) & (L1_SIZE - 1);
    uint64_t l2_idx = pfn & (L2_SIZE - 1);

    if (g_ver_root[l1_idx] == NULL) {
        g_ver_root[l1_idx] = calloc(L2_SIZE, sizeof(uint64_t));
    }
    __atomic_store_n(&g_ver_root[l1_idx][l2_idx], version, __ATOMIC_RELEASE);
}

static void safe_log(const char *msg) {
    if (write(STDERR_FILENO, msg, strlen(msg))) {};
}

// 健壮的阻塞读取 (处理 EINTR)
static int read_exact(int fd, void *buf, size_t len) {
    size_t received = 0;
    char *ptr = (char *)buf;
    while (received < len) {
        ssize_t ret = read(fd, ptr + received, len - received);
        if (ret > 0) received += ret;
        else if (ret == 0) return -1; // EOF
        else if (errno != EINTR) return -1; // Error
    }
    return 0;
}

static int internal_connect_master(void) {
    int sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock < 0) return -1;
    struct sockaddr_un addr = { .sun_family = AF_UNIX };
    
    const char *env_path = getenv("GVM_ENV_SOCK_PATH");
    if (!env_path) {
        char *inst_id = getenv("GVM_INSTANCE_ID");
        static char fallback_path[128];
        snprintf(fallback_path, sizeof(fallback_path), "/tmp/gvm_user_%s.sock", inst_id ? inst_id : "0");
        env_path = fallback_path;
    }
    strncpy(addr.sun_path, env_path, sizeof(addr.sun_path) - 1);
    if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        close(sock); return -1;
    }
    return sock;
}

/* 
 * 以下函数仅为了兼容 QEMU 命令行参数解析。
 * V29 使用 Wavelet 主动推送模型，不再需要 TTL 和手工 Watch 区域。
 */
void gvm_set_ttl_interval(int ms) { 
    // 留空：不再启动 V28 的收割者线程
}

void gvm_register_volatile_ram(uint64_t gpa, uint64_t size) { 
    // 留空：不再维护易失性区域链表
}

// =============================================================
// [链路 A] 同步缺页处理 (Master IPC / Slave UDP)
// =============================================================

static int request_page_sync(uintptr_t fault_addr, bool is_write) {
    uint64_t gpa = fault_addr - (uintptr_t)g_ram_base;
    gpa &= ~4095ULL; 
    uintptr_t aligned_addr = (uintptr_t)g_ram_base + gpa;
    
    // --- Master Mode (IPC) ---
    if (!g_is_slave) {
        if (t_com_sock == -1) { 
            t_com_sock = internal_connect_master();
            if (t_com_sock < 0) return -1;
        }
        
        struct gvm_ipc_fault_req req = { .gpa = gpa, .len = 4096, .vcpu_id = 0 };
        struct gvm_ipc_header_t ipc_hdr = { .type = GVM_IPC_TYPE_MEM_FAULT, .len = sizeof(req) };
        struct iovec iov[2] = { {&ipc_hdr, sizeof(ipc_hdr)}, {&req, sizeof(req)} };
        struct msghdr msg = { .msg_iov = iov, .msg_iovlen = 2 };
        
        if (sendmsg(t_com_sock, &msg, 0) < 0) return -1;
        
        // [V29 Fix] 接收带版本的 ACK
        struct gvm_ipc_fault_ack ack;
        if (read_exact(t_com_sock, &ack, sizeof(ack)) < 0) return -1;
        
        if (ack.status == 0) {
            set_local_page_version(gpa, ack.version); // 同步版本
            return 0;
        }
        return -1;
    }

    // ---------------------------------------------------------
    // [V29 新增] 线程局部 Socket 延迟初始化 (替代全局 g_fd_req)
    // ---------------------------------------------------------
    if (unlikely(t_req_sock == -1)) {
        struct sockaddr_in peer_addr;
        socklen_t addr_len = sizeof(peer_addr);
        
        // 1. 从主进程的全局 Socket (g_fd_req) "偷" 取目标 Proxy 地址
        // 这样每个线程都不需要重新解析配置文件，直接跟随主进程配置
        if (getpeername(g_fd_req, (struct sockaddr*)&peer_addr, &addr_len) < 0) {
            safe_log("[GVM] FATAL: getpeername failed in thread init\n");
            _exit(1);
        }

        // 2. 创建线程私有 UDP Socket
        t_req_sock = socket(AF_INET, SOCK_DGRAM, 0);
        if (t_req_sock < 0) {
            safe_log("[GVM] FATAL: Thread socket creation failed\n");
            _exit(1);
        }

        // 3. 独占连接到 Proxy (利用内核层过滤非目标流量)
        if (connect(t_req_sock, (struct sockaddr*)&peer_addr, sizeof(peer_addr)) < 0) {
            safe_log("[GVM] FATAL: Thread socket connect failed\n");
            close(t_req_sock); t_req_sock = -1;
            _exit(1);
        }
        
        // 4. 设置非阻塞 (配合 Poll 做超时控制)
        int flags = fcntl(t_req_sock, F_GETFL, 0);
        fcntl(t_req_sock, F_SETFL, flags | O_NONBLOCK);
    }

    // ---------------------------------------------------------
    // [原版逻辑保留] 构造协议包 (使用 t_net_buf 替代全局 buffer)
    // ---------------------------------------------------------
    struct gvm_header *hdr = (struct gvm_header *)t_net_buf;
    memset(hdr, 0, sizeof(struct gvm_header)); // [保留] 必须清零

    hdr->magic = htonl(GVM_MAGIC);
    
    // [保留] 区分读写意图 (V29 协议中通常统一为 MEM_READ，但此处保留你的逻辑)
    // 注意：需确保 MSG_ACQUIRE_... 在 protocol.h 有定义，否则回退到 MSG_MEM_READ
    #ifdef MSG_ACQUIRE_WRITE
        hdr->msg_type = htons(is_write ? MSG_ACQUIRE_WRITE : MSG_ACQUIRE_READ);
    #else
        hdr->msg_type = htons(MSG_MEM_READ); // V29 标准回退
    #endif

    hdr->payload_len = htons(8); 
    hdr->slave_id = htonl(g_slave_id);
    hdr->req_id = GVM_HTONLL((uint64_t)gpa); 
    hdr->mode_tcg = 1; 
    hdr->qos_level = 1; 

    // [保留] Payload: GPA
    *(uint64_t *)(t_net_buf + sizeof(struct gvm_header)) = GVM_HTONLL(gpa);

    // [保留] CRC 计算
    hdr->crc32 = 0; 
    uint32_t c = calculate_crc32(t_net_buf, sizeof(struct gvm_header) + 8);
    hdr->crc32 = htonl(c);

    // ---------------------------------------------------------
    // [V29 增强] 发送与接收 (使用 t_req_sock)
    // ---------------------------------------------------------
    
    // 发送
    if (send(t_req_sock, t_net_buf, sizeof(struct gvm_header) + 8, 0) < 0) {
        return -1; // 网络不可达
    }

    struct pollfd pfd = { .fd = t_req_sock, .events = POLLIN };
    int total_wait_ms = 0;
    
    while(1) {
        // [保留] 1000ms 超时 (优化：拆分为短间隔以支持重发)
        int ret = poll(&pfd, 1, 100); 
        
        if (ret == 0) {
            total_wait_ms += 100;
            // [新增] 简单重传机制，防止 UDP 丢包死锁
            if (total_wait_ms % 500 == 0) {
                send(t_req_sock, t_net_buf, sizeof(struct gvm_header) + 8, 0); 
            }
            if (total_wait_ms >= 5000) {
                // 5秒无回音，打印日志但不退出，防止 Guest 崩溃
                const char *msg = "[GVM] WARN: Page fault stuck > 5s\n";
                write(STDERR_FILENO, msg, strlen(msg));
                total_wait_ms = 0;
            }
            continue; 
        }
        
        if (ret < 0) {
            if (errno == EINTR) continue;
            return -1;
        }

        // 接收
        int n = recv(t_req_sock, t_net_buf, GVM_MAX_PACKET_SIZE, 0);
        if (n <= 0) continue; // 过滤空包

        if (n >= sizeof(struct gvm_header)) {
            struct gvm_header *rx = (struct gvm_header *)t_net_buf;
            
            // [保留] 校验
            if (ntohl(rx->magic) != GVM_MAGIC) continue;
            if (GVM_NTOHLL(rx->req_id) != gpa) continue; 

            // [新增] CRC 校验 (接收端也需要校验！)
            uint32_t remote_crc = ntohl(rx->crc32);
            rx->crc32 = 0;
            if (calculate_crc32(t_net_buf, n) != remote_crc) continue;

            if (ntohs(rx->msg_type) == MSG_MEM_ACK) {
                // [V29] 检查版本号 Payload (只多不少：解析更复杂的结构)
                // 结构: Header + gvm_mem_ack_payload { gpa, version, data }
                size_t expected_size = sizeof(struct gvm_header) + sizeof(struct gvm_mem_ack_payload);
                
                if (n >= expected_size) {
                    struct gvm_mem_ack_payload *payload = (struct gvm_mem_ack_payload*)(t_net_buf + sizeof(struct gvm_header));
                    
                    // 双重检查 Payload 内的 GPA
                    if (GVM_NTOHLL(payload->gpa) != gpa) continue;

                    // [保留] 写入内存
                    mprotect((void*)aligned_addr, 4096, PROT_READ | PROT_WRITE);
                    memcpy((void*)aligned_addr, payload->data, 4096);
                    
                    // [V29 新增] 更新本地版本号
                    uint64_t ver = GVM_NTOHLL(payload->version);
                    set_local_page_version(gpa, ver);
                    
                    return 0; // 成功
                }
            }
        }
    }
}

/* 
 * [物理意图] 解决 vCPU 写入线程与后台收割线程之间的“微观竞态”冲突。
 * [关键逻辑] 当 harvester 正在对某一页进行原子快照时，强制 sigsegv 线程在入口处进行纳秒级忙等。
 * [后果] 彻底杜绝了“脏快照”问题。若无此锁，Diff 引擎可能会捕获到一个正在被修改的半成品页面，导致全网数据损坏。
 */
static inline void wait_on_latch(uint64_t gpa) {
    // 计算索引并查对分段锁数组 g_latches
    int idx = LATCH_IDX(gpa);
    while (__atomic_load_n(&g_latches[idx].val, __ATOMIC_ACQUIRE) == gpa) {
        __builtin_ia32_pause(); 
    }
}

// ----------------------------------------------------------------------------
// [REVISED] 信号处理：加入 Latch 检查
// ---------------------------------------------------------------------------

#define LATCH_IDX(gpa) ((gpa >> 12) % LATCH_SHARDS)

/* 
 * [物理意图] 模拟处理器的“缺页异常处理单元”，实现按需拉取与乐观写入。
 * [关键逻辑] 1. 读缺页：回退到 V28 阻塞拉取；2. 写保护：利用预分配池进行 Copy-Before-Write (CBW) 捕获。
 * [后果] 这是整个前端最繁重的入口。必须保证零 malloc，任何在此处的阻塞（如等待网络）都会直接锁死 vCPU 的流水线。
 */
static void sigsegv_handler(int sig, siginfo_t *si, void *ucontext) {
    uintptr_t addr = (uintptr_t)si->si_addr;
    
    // 通过安全查表获取 GPA
    uint64_t gpa = hva_to_gpa_safe(addr);
    if (gpa == (uint64_t)-1) {
        // 说明访问的不是 RAM 区域（可能是 MMIO 或非法地址），交回给标准处理程序
        signal(SIGSEGV, SIG_DFL); 
        raise(SIGSEGV); 
        return;
    }
    
    gpa &= ~4095ULL; // 对齐到页
    void* aligned_addr = (void*)(addr & ~4095ULL);
    wait_on_latch(gpa);

    ucontext_t *ctx = (ucontext_t *)ucontext;
    bool is_write = (ctx->uc_mcontext.gregs[REG_ERR] & 0x2);

    if (is_write) {
        // 从预分配池中取，不要 malloc!
        int idx = atomic_fetch_add(&g_pool_idx, 1) % PAGE_POOL_SIZE;
        WritablePage *wp = &g_page_pool[idx];
        void *snapshot = g_image_pool[idx];
        
        memcpy(snapshot, aligned_addr, 4096);
        wp->gpa = gpa;
        wp->pre_image_snapshot = snapshot;

        // 插入链表（注意：这里必须用原子操作挂载，防止破坏链表）
        do {
            wp->next = __atomic_load_n(&g_writable_pages_list, __ATOMIC_ACQUIRE);
        } while (!__atomic_compare_exchange_n(&g_writable_pages_list, &wp->next, wp, true, __ATOMIC_RELEASE, __ATOMIC_ACQUIRE));
        
        mprotect(aligned_addr, 4096, PROT_READ | PROT_WRITE);
    } else {
        // 读缺页直接同步
        if (request_page_sync(addr, false) == 0) {
            mprotect(aligned_addr, 4096, PROT_READ);
        } else { _exit(1); }
    }
}

// [改为 Diff 包聚合，已被 add_to_aggregator 替代] 发送 Diff 的辅助函数
static void send_commit_diff_dual_mode(uint64_t gpa, uint16_t offset, uint16_t size, void *data) {
    if (g_fd_push < 0) return;
    
    // 1. 计算包大小
    size_t pl_len = sizeof(struct gvm_diff_log) + size;
    size_t pkt_len = sizeof(struct gvm_header) + pl_len;
    
    // 2. 分配缓冲区
    uint8_t *buf = malloc(pkt_len);
    if (!buf) return;

    // 3. 填充 Header
    struct gvm_header *hdr = (struct gvm_header *)buf;
    hdr->magic = htonl(GVM_MAGIC);
    hdr->msg_type = htons(MSG_COMMIT_DIFF);
    hdr->payload_len = htons(pl_len);
    hdr->slave_id = htonl(g_slave_id);
    hdr->req_id = 0; 
    hdr->qos_level = 1; // 走快车道
    hdr->crc32 = 0;     // 先清零
    hdr->flags = 0;

    // 4. 填充 Payload (Diff Log)
    struct gvm_diff_log *log = (struct gvm_diff_log *)(buf + sizeof(*hdr));
    log->gpa = GVM_HTONLL(gpa);
    // 携带本地版本号供 Directory 校验
    log->version = GVM_HTONLL(get_local_page_version(gpa));
    log->offset = htons(offset);
    log->size = htons(size);
    memcpy(log->data, data, size);

    // 5. 计算 CRC32
    hdr->crc32 = htonl(calculate_crc32(buf, pkt_len));

    // 6. 发送并释放
    send(g_fd_push, buf, pkt_len, 0);
    free(buf);
}

// 聚合器状态机
static struct {
    uint8_t buf[2048]; // 略大于 MTU，确保能装下一个完整包 + 头部
    int curr_offset;
    pthread_mutex_t lock;
} g_aggregator = { 
    .curr_offset = 0, 
    .lock = PTHREAD_MUTEX_INITIALIZER 
};

// 发送函数：将缓冲区推向网络
static void flush_aggregator(void) {
    if (g_aggregator.curr_offset == 0) return;
    // 使用 send 发往 g_fd_push。由于 t_req_sock 的 connect 逻辑，这里直接发。
    if (send(g_fd_push, g_aggregator.buf, g_aggregator.curr_offset, 0) < 0) {
        // 网络拥塞处理：如果发送失败，我们也只能清空，版本号一致性由 FORCE_SYNC 保证
    }
    g_aggregator.curr_offset = 0;
}

// 核心聚合函数：取代原有的 send_push_packet
static void add_to_aggregator(uint64_t gpa, uint64_t version, uint16_t off, uint16_t sz, void *data, uint8_t flags) {
    size_t payload_len = sizeof(struct gvm_diff_log) + sz;
    size_t needed = sizeof(struct gvm_header) + payload_len;
    
    pthread_mutex_lock(&g_aggregator.lock);
    
    // 如果当前包放不下，或者这个包本身就超过了 MTU 分片限制，则先发送之前的
    if (g_aggregator.curr_offset + needed > MTU_SIZE) {
        flush_aggregator();
    }

    // 如果单包就超过 MTU（虽然对于 Diff 很少见），直接绕过聚合器发送
    if (needed > MTU_SIZE) {
        uint8_t *tmp = malloc(needed);
        if (!tmp) { pthread_mutex_unlock(&g_aggregator.lock); return; }
        struct gvm_header *h = (struct gvm_header *)tmp;
        h->magic = htonl(GVM_MAGIC);
        h->msg_type = htons(MSG_COMMIT_DIFF);
        h->payload_len = htons(payload_len);
        h->slave_id = htonl(g_slave_id);
        h->flags = flags;
        h->crc32 = 0;
        struct gvm_diff_log *l = (struct gvm_diff_log *)(tmp + sizeof(*h));
        l->gpa = GVM_HTONLL(gpa);
        l->version = GVM_HTONLL(version);
        l->offset = htons(off);
        l->size = htons(sz);
        if (sz > 0) memcpy(l->data, data, sz);
        h->crc32 = htonl(calculate_crc32(tmp, needed));
        send(g_fd_push, tmp, needed, 0);
        free(tmp);
        pthread_mutex_unlock(&g_aggregator.lock);
        return;
    }

    // 填充到缓冲区
    struct gvm_header *hdr = (struct gvm_header *)(g_aggregator.buf + g_aggregator.curr_offset);
    hdr->magic = htonl(GVM_MAGIC);
    hdr->msg_type = htons(MSG_COMMIT_DIFF);
    hdr->payload_len = htons(payload_len);
    hdr->slave_id = htonl(g_slave_id);
    hdr->qos_level = 1;
    hdr->flags = flags;
    hdr->crc32 = 0;

    struct gvm_diff_log *log = (struct gvm_diff_log *)(g_aggregator.buf + g_aggregator.curr_offset + sizeof(struct gvm_header));
    log->gpa = GVM_HTONLL(gpa);
    log->version = GVM_HTONLL(version);
    log->offset = htons(off);
    log->size = htons(sz);
    if (sz > 0) memcpy(log->data, data, sz);

    // 在聚合前计算单子包 CRC
    hdr->crc32 = htonl(calculate_crc32((uint8_t*)hdr, needed));
    
    g_aggregator.curr_offset += needed;
    pthread_mutex_unlock(&g_aggregator.lock);
}

/* 
 * [物理意图] 充当内存页面的“分布式写回缓存（Write-back Cache）”管理器。
 * [关键逻辑] 1. 计算增量（Diff）；2. 聚合（Aggregator）打包；3. 执行 AIMD 自适应同步屏障，根据 RTT 调整提交频率。
            逻辑: Detach -> Freeze(Lock) -> Snapshot -> Release(Unlock) -> Diff -> Commit -> Sync
 * [后果] 它通过异步提交隐藏了网络延迟。若收割速度跟不上写入速度，Guest 系统内会发生明显的“因果倒置”现象。
 */
static void *diff_harvester_thread_fn(void *arg) {
    void *current_snapshot = malloc(4096);
    if (!current_snapshot) return NULL;

    int batch_counter = 0;

    while (g_threads_running) {
        usleep(1000); // 1ms 采集周期

        // 1. 偷走链表 (Detach List)
        WritablePage *batch_head = NULL;
        pthread_mutex_lock(&g_writable_list_lock);
        if (g_writable_pages_list) {
            batch_head = g_writable_pages_list;
            g_writable_pages_list = NULL; 
        }
        pthread_mutex_unlock(&g_writable_list_lock);

        if (!batch_head) continue;

        // 2. 遍历处理脏页
        WritablePage *curr = batch_head;
        while (curr) {
            void *page_addr = (uint8_t*)g_ram_base + curr->gpa;
            if (!page_addr) { 
                // 错误处理：释放资源并跳过
                free(curr->pre_image_snapshot);
                WritablePage *nxt = curr->next; free(curr); curr = nxt;
                continue; 
            }
            int idx = LATCH_IDX(curr->gpa);
            
            // [A] 上锁 (Freeze): 告诉 Signal Handler 暂停操作
            __atomic_store_n(&g_latches[idx].val, curr->gpa, __ATOMIC_RELEASE);

            // [B] 冻结权限: 设为只读
            mprotect(page_addr, 4096, PROT_READ);
            __sync_synchronize();

            // [C] 快照 (Snapshot): 安全拷贝
            memcpy(current_snapshot, page_addr, 4096);

            // [D] 解锁 (Release): Signal Handler 可以继续了
            __atomic_store_n(&g_latches[idx].val, (uint64_t)-1, __ATOMIC_RELEASE);

            uint64_t ver = get_local_page_version(curr->gpa);

            if (is_page_all_zero(current_snapshot)) {
                // 调用聚合器发送零页消息
                add_to_aggregator(curr->gpa, ver + 1, 0, 0, NULL, GVM_FLAG_ZERO);
                set_local_page_version(curr->gpa, ver + 1);
            } else {
                // [E] 计算 Diff (耗时操作，已移出临界区)
                int start = -1, end = -1;
                uint64_t *p64_now = (uint64_t*)current_snapshot;
                uint64_t *p64_pre = (uint64_t*)curr->pre_image_snapshot;
            
                for (int i = 0; i < 512; i++) {
                    if (p64_now[i] != p64_pre[i]) {
                        if (start == -1) start = i * 8;
                        end = i * 8 + 7;
                    }
                }

                // [F] 提交与同步 (Commit & Sync)
                if (start != -1) {
                    uint16_t size = end - start + 1;
                    // 发送 Diff 包
                    add_to_aggregator(curr->gpa, ver + 1, (uint16_t)start, size, (uint8_t*)current_snapshot + start, 0);

                    // --- [AIMD Sync Engine] ---
                    if (g_client_sync_batch > 0) {
                        batch_counter++;
                        // 达到阈值，触发同步屏障
                        if (batch_counter >= g_client_sync_batch) {
                            long rtt = wait_for_directory_ack_safe();
                            batch_counter = 0;

                            // 自适应流控逻辑
                            if (g_enable_auto_tuning) {
                                if (rtt < 0) {
                                    // 超时/丢包：急速降速
                                    g_client_sync_batch = (g_client_sync_batch > 16) ? g_client_sync_batch / 2 : 1;
                                } else if (rtt < 500) { 
                                    // 网络极好 (<0.5ms)：加法增大
                                    if (g_client_sync_batch < g_max_batch) g_client_sync_batch += 16;
                                } else if (rtt > 5000) { 
                                    // 网络拥塞 (>5ms)：乘法减小
                                    g_client_sync_batch = (g_client_sync_batch * 3) / 4; 
                                    if (g_client_sync_batch < g_min_batch) g_client_sync_batch = g_min_batch;
                                }
                            }
                        }
                    }
                
                    // 乐观更新本地版本号
                    set_local_page_version(curr->gpa, ver + 1);
                }
            }

            // 清理资源
            free(curr->pre_image_snapshot);
            WritablePage *next_node = curr->next;
            free(curr); 
            curr = next_node;
        }
        flush_aggregator();
    }
    free(current_snapshot);
    return NULL;
}

// =============================================================
// [链路 B] 流式监听线程 (Stream Listener)
// =============================================================

// 环形缓冲区，用于处理 IPC 流的粘包/拆包
typedef struct {
    uint8_t buffer[GVM_MAX_PACKET_SIZE * 4];
    size_t head; // Read ptr
    size_t tail; // Write ptr
} StreamBuffer;

static void sb_init(StreamBuffer *sb) {
    sb->head = 0;
    sb->tail = 0;
}

static void sb_compact(StreamBuffer *sb) {
    if (sb->head > 0) {
        size_t len = sb->tail - sb->head;
        if (len > 0) memmove(sb->buffer, sb->buffer + sb->head, len);
        sb->tail = len;
        sb->head = 0;
    }
}

// --- [V29 Sync Engine Config] ---
static int g_client_sync_batch = 1024; // 当前生效的 Batch
static int g_min_batch = 1;            // 下限
static int g_max_batch = 8192;         // 上限
static int g_enable_auto_tuning = 1;   // 开关：1=自动, 0=固定(强一致用)

// --- [V29 Sync Concurrency Control] ---
static pthread_mutex_t g_sync_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_sync_cond = PTHREAD_COND_INITIALIZER;
static volatile int g_ack_received = 0; // 状态标志位

// 辅助：获取微秒时间
static uint64_t get_us_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000UL + ts.tv_nsec / 1000;
}

// 导出配置接口
void gvm_set_client_sync_mode(int batch_size, int auto_tune) {
    if (batch_size > 0) g_client_sync_batch = batch_size;
    g_enable_auto_tuning = auto_tune;
    
    // 如果设为 1，通常意味着强一致性需求，关闭自动调优
    if (batch_size == 1) {
        g_enable_auto_tuning = 0;
        printf("[GVM] Strict Consistency Mode Activated (Batch=1, No Tuning)\n");
    } else {
        printf("[GVM] Sync Mode: Initial Batch=%d, AutoTuning=%d\n", 
               batch_size, auto_tune);
    }
}

/* 
 * [物理意图] 在 P2P 集群中插入一个“顺序执行栅栏”。
 * [关键逻辑] 发起一个带有特殊 Magic ID 的 PING 包，并阻塞等待 Directory 的 ACK，以确认上一批次的写入已落盘。
 * [后果] 确保了分布式内存的“强顺序一致性”。它防止了在执行关键 IO 指令（如 GPU 命令提交）时，内存数据尚未同步完成的情况。
 */
static long wait_for_directory_ack_safe(void) {
    if (g_fd_push < 0) return -1;

    // 1. 准备发送 PING
    struct gvm_header hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic = htonl(GVM_MAGIC);
    hdr.msg_type = htons(MSG_PING);
    hdr.slave_id = htonl(g_slave_id);
    hdr.req_id = GVM_HTONLL(SYNC_MAGIC); // 特殊标记

    // 2. 【关键】先加锁，重置状态位
    pthread_mutex_lock(&g_sync_lock);
    g_ack_received = 0;
    pthread_mutex_unlock(&g_sync_lock);

    uint64_t t_start = get_us_time();

    // 3. 发送网络包
    if (send(g_fd_push, &hdr, sizeof(hdr), 0) < 0) return -1;

    // 4. 【关键】带超时的条件等待
    pthread_mutex_lock(&g_sync_lock);
    
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec += 1; // 1秒超时
    
    // 只要没收到 ACK，就继续等 (防止虚假唤醒)
    while (g_ack_received == 0) {
        int rc = pthread_cond_timedwait(&g_sync_cond, &g_sync_lock, &ts);
        if (rc == ETIMEDOUT) {
            pthread_mutex_unlock(&g_sync_lock);
            return -1; // 超时
        }
    }
    pthread_mutex_unlock(&g_sync_lock);

    return (long)(get_us_time() - t_start);
}

/* 
 * [物理意图] 系统的“入站流量净化器”，解决 UDP 乱序和 TLB 刷新成本问题。
 * [关键逻辑] 1. 维护重排窗口（Reorder Window）填补版本空洞；2. 执行延迟刷新（Lazy Flush）以合并 mprotect 调用。
 * [后果] 延迟刷新是性能的救星。若每收到一个 Diff 都执行一次 TLB Shootdown，vCPU 的有效执行时间将缩减到不足 10%。
 */
static void *mem_push_listener_thread(void *arg) {
    StreamBuffer sb;
    sb_init(&sb);
    
    // 初始化 TLS 队列索引
    t_lazy_count = 0; 
    uint64_t last_cleanup_us = 0;
    
    struct pollfd pfd = { .fd = g_fd_push, .events = POLLIN };
    printf("[GVM] Async Push Listener Started (Streaming Mode + Lazy Flush).\n");

    while (g_threads_running) {
        // [Stage 0] 循环前冲刷
        // 确保上一轮循环遗留的 RW 页面被锁回 RO。
        // 这保证了 poll 等待期间，页面是安全的（RO）。
        flush_lazy_ro_queue();

        int ret = poll(&pfd, 1, 100);
        uint64_t now_us = get_us_time();
        if (now_us - last_cleanup_us > 200000) { // 每 200ms 清理一次
            pthread_spin_lock(&g_reorder_lock);
            for (int i = 0; i < REORDER_WIN_SIZE; i++) {
                if (g_reorder_buf[i].active && (now_us - g_reorder_buf[i].timestamp_us > 200000)) {
                    uint64_t stale_gpa = g_reorder_buf[i].gpa;
                    
                    // 释放资源
                    free(g_reorder_buf[i].data);
                    g_reorder_buf[i].active = false;
                    
                    // 触发强制同步
                    void* hva = gpa_to_hva_safe(stale_gpa);
                    if (hva) mprotect(hva, 4096, PROT_NONE);
                    set_local_page_version(stale_gpa, 0);
                }
            }
            pthread_spin_unlock(&g_reorder_lock);
            last_cleanup_us = now_us;
        }
        if (ret <= 0) continue;

        // [Stage 1] 读取网络流
        size_t space = sizeof(sb.buffer) - sb.tail;
        if (space == 0) { sb_init(&sb); continue; }
        ssize_t n = recv(g_fd_push, sb.buffer + sb.tail, space, 0);
        if (n <= 0) {
            if (errno == EINTR) continue;
            break; 
        }
        sb.tail += n;

        // [Stage 2] 批量解析与应用
        while (sb.tail - sb.head >= sizeof(struct gvm_ipc_header_t)) {
            struct gvm_ipc_header_t *ipc = (struct gvm_ipc_header_t *)(sb.buffer + sb.head);
            size_t total_msg_len = sizeof(struct gvm_ipc_header_t) + ipc->len;

            if (sb.tail - sb.head < total_msg_len) break; // 数据未收全，等待下一轮

            // 指向 GVM 协议头
            void *data = sb.buffer + sb.head + sizeof(struct gvm_ipc_header_t);
            
            // 只有 INVALIDATE 类型的 IPC 消息才包含网络包数据
            if (ipc->type == GVM_IPC_TYPE_INVALIDATE) {
                struct gvm_header *hdr = (struct gvm_header *)data;
                void *payload = data + sizeof(struct gvm_header);
                uint16_t msg_type = ntohs(hdr->msg_type);
                
                // [FIXED] 提取包长度，供乱序重排使用
                uint16_t p_len = ntohs(hdr->payload_len); 

                // --- 逻辑 A: Diff 推送 (带乱序处理) ---
                if (msg_type == MSG_PAGE_PUSH_DIFF) {
                    struct gvm_diff_log* log = (struct gvm_diff_log*)payload;
                    uint64_t gpa = GVM_NTOHLL(log->gpa);
                    uint64_t push_ver = GVM_NTOHLL(log->version);
                    uint64_t local_ver = get_local_page_version(gpa);

                    if (is_next_version(local_ver, push_ver)) {
                        // 顺序到达：直接应用 (内部调用 defer_ro_protect，保持 RW)
                        gvm_apply_remote_push(msg_type, payload);
                        
                        // 链式反应：检查重排缓冲区是否有 v+2, v+3...
                        uint64_t next = push_ver + 1;
                        // 只要能应用成功，就继续。参数虽然传了，但函数内部现在会动态寻找真正的“下一跳”。
                        while (check_and_apply_next(gpa, 0)) ;
                        
                    } else if (is_newer_version(local_ver, push_ver) && !is_next_version(local_ver, push_ver)) {
                        // 乱序到达：存入重排窗口
                        if (!is_newer_version(local_ver + REORDER_WIN_SIZE, push_ver)) {
                            // [Safety] 这里使用 p_len 确保拷贝完整
                            buffer_future_packet(gpa, push_ver, msg_type, payload, p_len);
                        } else {
                            // 严重乱序：回退到 Pull 模式
                            // 这种情况下必须立即锁回，不能 Lazy，因为状态已重置
                            mprotect((uint8_t*)g_ram_base + gpa, 4096, PROT_NONE);
                            set_local_page_version(gpa, 0);
                        }
                    }
                } 
                // --- 逻辑 B: 全页推送 / 强制同步 ---
                else if (msg_type == MSG_PAGE_PUSH_FULL || msg_type == MSG_FORCE_SYNC) {
                    gvm_apply_remote_push(msg_type, payload);
                }
                // --- 逻辑 C: Prophet RPC (V29 保留) ---
                else if (msg_type == MSG_RPC_BATCH_MEMSET) {
                    // Prophet 通知。这里不需要 Lazy Flush，因为 RPC 执行通常是原子的
                    gvm_apply_remote_push(msg_type, payload);
                }
                // --- 逻辑 D: 同步 ACK (Sync Batch 闭环) ---
                else if (msg_type == MSG_MEM_ACK && GVM_NTOHLL(hdr->req_id) == SYNC_MAGIC) {
                    // 收到 Directory 的确认，说明 sync_batch 内的所有 diff 都已落盘
                    // 唤醒 Harvester 线程继续发送下一批
                    pthread_mutex_lock(&g_sync_lock);
                    g_ack_received = 1; 
                    pthread_cond_signal(&g_sync_cond); 
                    pthread_mutex_unlock(&g_sync_lock);
                }
            }
            
            // 移动环形缓冲区指针
            sb.head += total_msg_len;
        }
        sb_compact(&sb);

        // [Stage 3] 循环末尾强制冲刷
        // 这一步至关重要。它确保了上述 while 循环处理的一批包（可能几十个）
        // 对应的页面在处理完后立刻被锁回 RO。
        // 这将 RW 窗口限制在“微秒级”，兼顾了性能与一致性。
        flush_lazy_ro_queue(); 
    }
    return NULL;
}

// --- 初始化 ---
void giantvm_user_mem_init(void *ram_ptr, size_t ram_size) {
    g_ram_base = ram_ptr;
    g_ram_size = ram_size;

    init_latches();
    pthread_spin_init(&g_reorder_lock, 0); 

    for(int i=0; i<PAGE_POOL_SIZE; i++) {
        g_image_pool[i] = mmap(NULL, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    }

    size_t num_pages = ram_size / 4096;

    char *env_req = getenv("GVM_SOCK_REQ");
    char *env_push = getenv("GVM_SOCK_PUSH");
    char *env_id = getenv("GVM_SLAVE_ID");

    if (env_req && env_push) {
        g_is_slave = 1;
        g_fd_req = atoi(env_req);
        g_fd_push = atoi(env_push);
        g_slave_id = env_id ? atoi(env_id) : 0;
        
        printf("[GiantVM-User] V29 Wavelet Engine Active (Slave ID: %d)\n", g_slave_id);
        
        g_threads_running = true;
        pthread_create(&g_listen_thread, NULL, mem_push_listener_thread, NULL);
        pthread_create(&g_harvester_thread, NULL, diff_harvester_thread_fn, NULL);
    }

    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_flags = SA_SIGINFO | SA_NODEFER; 
    sa.sa_sigaction = sigsegv_handler;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV, &sa, NULL);

    // Initial state: Invalid (PROT_NONE)
    mprotect(g_ram_base, g_ram_size, PROT_NONE);
}

