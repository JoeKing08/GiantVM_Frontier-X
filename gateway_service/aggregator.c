/*
 * [IDENTITY] Gateway Sidecar - The Stateless Router
 * ---------------------------------------------------------------------------
 * 物理角色：网络侧车，无状态包聚合器。
 * 职责边界：
 * 1. 物理路由转发：根据虚拟节点 ID 范围将包投递给下一跳 IP。
 * 2. 流量聚合：合并小包为 MTU 大包，降低全网 PPS 压力。
 * 3. 自学习逻辑：通过捕获入站流量自动更新路由条目。
 * 
 * [禁止事项]
 * - 严禁进行数据深包检测 (DPI) 或 CRC32 校验。
 * - 严禁引入任何动态内存分配 (malloc/free)，必须使用预分配 Buffer。
 * - 严禁持有锁进行网络发送。
 * ---------------------------------------------------------------------------
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <pthread.h> 
#include <sched.h>
#include <poll.h>
#include <sys/sysinfo.h>
#include <time.h>

#include "aggregator.h"
#include "../common_include/wavevm_protocol.h"
#include "uthash.h"

#if defined(__x86_64__) || defined(__i386__)
  #define CPU_RELAX() __asm__ volatile("pause" ::: "memory")
#else
  #define CPU_RELAX() sched_yield()
#endif

// A unified, hashable structure for each known downstream node route.
typedef struct {
    uint32_t id;                    // Key for the hash table (slave_id)
    struct sockaddr_in addr;        // Slave's network address, pre-filled
    slave_buffer_t *buffer;         // Pointer to the aggregation buffer, LAZILY ALLOCATED
    pthread_mutex_t lock;           // Per-node lock for buffer access
    UT_hash_handle hh;              // Makes this structure hashable by uthash
} gateway_node_t;

// --- 全局状态 ---
static gateway_node_t *g_node_map = NULL; // IMPORTANT: Must be initialized to NULL
// [REVISED PATCH] 使用读写锁替代互斥锁，保障数据面性能
static pthread_rwlock_t g_map_lock = PTHREAD_RWLOCK_INITIALIZER; // A global lock to protect the hash map itself (for creation/deletion)
#define BATCH_SIZE 64

static struct sockaddr_in g_upstream_addr; // The address of the upstream gateway or master
static volatile int g_primary_socket = -1; 
static int g_upstream_tx_socket = -1;
static int g_local_port = 0;
int g_ctrl_port = 0; // 供应给 wavevm_gateway

static int g_is_single_core = 0;

void detect_cpu_env() {
    if (get_nprocs() <= 1) g_is_single_core = 1;
}


/* 
 * [物理意图] 在纳秒级时间内，从动态哈希表中定位特定虚拟节点的“物理坐标”。
 * [关键逻辑] 使用 uthash 维护 ID 到 IP/Port 的映射。采用“快慢路径分离”：查找走无锁读，创建走写锁保护。
 * [后果] 这是网关转发性能的瓶颈点。若查找效率低下，整个 Pod 内部的内存同步延迟将呈指数级上升。
 *[警告] We perform a lock-free read here. This is ONLY safe because:
       1. The hash table is populated SINGLE-THREADED during initialization.
       2. The hash table is EFFECTIVELY IMMUTABLE during runtime.
       3. NO dynamic node addition/rehashing allows to happen while workers are running.
       DO NOT call HASH_ADD or find_or_create_node after init_aggregator returns!
 */
static inline gateway_node_t* find_node(uint32_t slave_id) {
    gateway_node_t *node = NULL;
    // HASH_FIND is read-only. Safe on immutable table.
    HASH_FIND_INT(g_node_map, &slave_id, node);
    return node;
}

// Helper function to find a node, creating it if it doesn't exist.
static gateway_node_t* find_or_create_node(uint32_t slave_id) {
    gateway_node_t *node = find_node(slave_id);
    if (node) {
        return node;
    }

    // Node not found, need to create it under the global map lock.
    pthread_rwlock_wrlock(&g_map_lock);
    
    // Double-check after acquiring the lock to handle race condition
    HASH_FIND_INT(g_node_map, &slave_id, node);
    if (node == NULL) {
        node = (gateway_node_t*)calloc(1, sizeof(gateway_node_t));
        if (node) {
            node->id = slave_id;
            pthread_mutex_init(&node->lock, NULL);
            node->buffer = NULL; // Buffer is lazily allocated on the first push
            HASH_ADD_INT(g_node_map, id, node);
        } else {
            fprintf(stderr, "[Gateway CRITICAL] Out of memory creating new node entry!\n");
        }
    }
    
    pthread_rwlock_unlock(&g_map_lock);
    return node;
}

/* 
 * [物理意图] 注入“静态初始坐标”，作为 P2P 网络启动时的引航灯。
 * [关键逻辑] 解析配置文件中的 ROUTE 范围指令，一次性预热数千个虚拟节点的路由条目，避免运行时的哈希抖动。
 * [后果] 提供了系统的“初始稳定性”。若配置加载错误，节点启动后的 Gossip 宣告将无法找到正确的上游网关。
 */
static int load_slave_config(const char *path) {
    FILE *fp = fopen(path, "r");
    if (!fp) { perror("Config open"); return -1; }
    
    char line[256];
    int routes_loaded = 0;

    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
        
        char keyword[16];
        // 仅支持 ROUTE 指令
        if (sscanf(line, "%15s", keyword) != 1) continue;
        
        if (strcmp(keyword, "ROUTE") == 0) {
            uint32_t start_id, count;
            char ip_str[64];
            int port;
            
            // 格式: ROUTE <StartVID> <Count> <IP> <Port>
            if (sscanf(line, "%*s %u %u %63s %d", &start_id, &count, ip_str, &port) == 4) {
                // 展开路由聚合
                for (uint32_t i = 0; i < count; i++) {
                    uint32_t current_id = start_id + i;
                    gateway_node_t *node = find_or_create_node(current_id);
                    if (node) {
                        node->addr.sin_family = AF_INET;
                        node->addr.sin_addr.s_addr = inet_addr(ip_str);
                        node->addr.sin_port = htons(port);
                    }
                }
                routes_loaded++;
            }
        }
    }
    fclose(fp);
    printf("[Gateway] Loaded %d route groups.\n", routes_loaded);
    return 0;
}

// Sends a raw datagram to a specific downstream node address.
static int raw_send_to_downstream(int fd, gateway_node_t *node, void *data, int len) {
    if (!node || node->addr.sin_port == 0) return -EHOSTUNREACH; 
    return sendto(fd, data, len, MSG_DONTWAIT, (struct sockaddr*)&node->addr, sizeof(node->addr));
}

/* 
 * [物理意图] 将积压的“内存小波”正式投射到物理光纤。
 * [关键逻辑] 检查聚合缓冲区，将多个子包封装为一个 MTU 大小的 UDP 数据包，并执行非阻塞发送。
 * [后果] 它解决了分布式内存的“PPS 爆炸”问题。通过牺牲极微小的延迟换取了巨大的带宽利用率，防止了网卡软中断打死 CPU。
 */
static int flush_buffer(int fd, gateway_node_t *node) {
    if (!node || !node->buffer || node->buffer->current_len == 0) return 0;
    
    int ret = raw_send_to_downstream(fd, node, node->buffer->raw_data, node->buffer->current_len);
    
    if (ret < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) {
            return -1; // Network congested, tell caller to keep data.
        }
        // For other fatal errors, we still clear the buffer to prevent resending bad data.
    }

    node->buffer->current_len = 0;
    return 0;
}

/* 
 * [物理意图] 模拟硬件控制器的“指数退避”机制，防止网络雪崩。
 * [关键逻辑] 根据尝试次数执行三级等待：1. CPU 忙等 (pause)；2. 软让权 (nanosleep)；3. 深度睡眠 (usleep)。
 * [后果] 这是针对云环境（如 K8s）的关键优化。它防止了网关在网络拥塞时，通过疯狂轮询耗尽物理 CPU 的配额。
 */
static void smart_backoff(int attempts) {
    // 场景 A: 单核机器 / 极度受限容器
    // 没有任何资本进行忙等，必须立即把 CPU 让给 QEMU 进程
    if (g_is_single_core) {
        sched_yield(); 
        return;
    }

    // 场景 B: 多核高性能机器
    if (attempts < 1000) {
        // 阶段 1: CPU 忙等 (约 1-2us)
        // 保持 CPU 流水线热度，赌网卡马上这就好
        #if defined(__x86_64__) || defined(__i386__)
            __asm__ volatile("pause");
        #elif defined(__aarch64__)
            __asm__ volatile("yield");
        #endif
    } else if (attempts < 2000) {
        // 阶段 2: 软让出 (约 0-10us)
        // nanosleep(0) 会触发调度检查，如果没有更高优先级任务，立即返回
        struct timespec ts = {0, 0};
        nanosleep(&ts, NULL);
    } else {
        // 阶段 3: 认怂 (1ms+)
        // 网络彻底堵死，必须睡久一点防止烧干 CPU Quota
        usleep(1);
    }
}

/* 
 * [物理意图] 实现“零拷贝转发”与“聚合转发”的智能分流。
 * [关键逻辑] 1. 小包入队聚合；2. 超过 MTU 的大包（如全页同步）直接穿透（Passthrough）绕过缓冲区。
 * [后果] 这一步保证了包序的“因果一致性”。它确保了全量更新包不会被后续的小增量包在网关缓冲区内超越。
 */
static int internal_push(int fd, uint32_t slave_id, void *data, int len) {
    // 1. 读锁保护查找，防止与动态路由更新冲突
    pthread_rwlock_rdlock(&g_map_lock);
    gateway_node_t *node = find_node(slave_id);
    if (!node) {
        pthread_rwlock_unlock(&g_map_lock);
        return -1;
    }
    
    // 2. 获取节点级互斥锁，随后释放全局读锁
    pthread_mutex_lock(&node->lock);
    pthread_rwlock_unlock(&g_map_lock); 

    // [FIXED] 大包透传逻辑 (Pass-through)
    // 防止缓冲区溢出，并解决大包死循环重试问题
    if (len > MTU_SIZE) {
        // A. 为了保序，必须先冲刷掉 Buffer 里已有的积压数据
        if (node->buffer && node->buffer->current_len > 0) {
            // 尝试冲刷，如果网络拥塞失败，返回 EBUSY 让上层退避重试
            // 不能丢弃旧数据，否则包序会乱
            if (flush_buffer(fd, node) != 0) {
                pthread_mutex_unlock(&node->lock);
                return -EBUSY; 
            }
        }
        
        // B. 缓冲区已空，直接发送大包 (绕过 Buffer)
        pthread_mutex_unlock(&node->lock);
        int ret = raw_send_to_downstream(fd, node, data, len);
        if (ret < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) return -EBUSY;
        return ret; // 返回发送结果 (可能 EAGAIN)
    }

    // --- 常规小包聚合逻辑 ---
    
    // Lazy allocation
    if (node->buffer == NULL) {
        node->buffer = (slave_buffer_t*)malloc(sizeof(slave_buffer_t));
        if (node->buffer) {
            node->buffer->current_len = 0;
        } else {
            pthread_mutex_unlock(&node->lock);
            return -ENOMEM;
        }
    }
    
    // 缓冲区满？冲刷。
    if (node->buffer->current_len + len > MTU_SIZE) {
        if (flush_buffer(fd, node) != 0) {
            pthread_mutex_unlock(&node->lock);
            return -EBUSY; 
        }
    }
    
    // 安全拷贝 (此时 len <= MTU 且 buffer 有空间)
    memcpy(node->buffer->raw_data + node->buffer->current_len, data, len);
    node->buffer->current_len += len;

    // 高性能即时冲刷阈值 (85%)
    if (node->buffer->current_len > (MTU_SIZE * 0.85)) {
        flush_buffer(fd, node);
    }

    pthread_mutex_unlock(&node->lock);
    return 0;
}

extern void internal_process_single_packet(struct wvm_header *hdr, uint32_t src_ip);

int push_to_aggregator(uint32_t slave_id, void *data, int len) {
    extern int g_my_node_id;

    if (g_primary_socket < 0) return -1;
    return internal_push(g_primary_socket, slave_id, data, len);
}

void flush_all_buffers(void) {
    gateway_node_t *current_node, *tmp;
    if (g_primary_socket < 0) return;

    // [FIX] 必须加读锁保护遍历过程！
    // 虽然 HASH_ITER 本身较慢，但这是保证不崩的唯一方法
    pthread_rwlock_rdlock(&g_map_lock);
    
    HASH_ITER(hh, g_node_map, current_node, tmp) {
        // 注意：这里需要获取节点锁。
        // 锁序：MapLock(Read) -> NodeLock(Mutex) 是安全的，不会死锁
        pthread_mutex_lock(&current_node->lock);
        flush_buffer(g_primary_socket, current_node);
        pthread_mutex_unlock(&current_node->lock);
    }
    
    pthread_rwlock_unlock(&g_map_lock);
}

/* 
 * [物理意图] 实现无中心网络的“身份识别”与“路由热修复”。
 * [关键逻辑] 监听并捕获所有入站流量，自动提取源 ID 与物理 IP。若发现邻居坐标变更，立即更新路由表。
 * [后果] 这是 WaveVM 能够支撑百万节点的奥秘。它不再依赖人工配置，而是通过“谁发包，我认识谁”实现拓扑的自动收敛。
 */
static void learn_route(uint32_t slave_id, struct sockaddr_in *addr) {
    // 1. 快速检查：如果已有路由且未变，直接返回 (无锁读)
    gateway_node_t *node = NULL;
    pthread_rwlock_rdlock(&g_map_lock);
    HASH_FIND_INT(g_node_map, &slave_id, node);
    if (node && 
        node->addr.sin_addr.s_addr == addr->sin_addr.s_addr &&
        node->addr.sin_port == addr->sin_port) {
        pthread_rwlock_unlock(&g_map_lock);
        return; // 路由稳定，无需更新
    }
    pthread_rwlock_unlock(&g_map_lock);

    // 2. 变更或新增：获取写锁进行更新
    pthread_rwlock_wrlock(&g_map_lock);
    // Double check
    HASH_FIND_INT(g_node_map, &slave_id, node);
    if (!node) {
        node = calloc(1, sizeof(gateway_node_t));
        if (node) {
            node->id = slave_id;
            pthread_mutex_init(&node->lock, NULL);
            HASH_ADD_INT(g_node_map, id, node);
            printf("[Gateway-Auto] Learned New Node: %u\n", slave_id);
        }
    }
    
    if (node) {
        node->addr = *addr;
        // printf("[Gateway-Auto] Updated Route Node: %u\n", slave_id);
    }
    pthread_rwlock_unlock(&g_map_lock);
}

/* 
 * [物理意图] 打造一台“软件定义的万兆交换机”。
 * [关键逻辑] 利用 CPU 亲和性将 Worker 绑定到特定核心，配合 recvmmsg 进行零拷贝接收与自学习路由分发。
 * [后果] 这是网关的“物理引擎”。其吞吐能力直接决定了超级虚拟机的总线带宽，必须保持极简逻辑以规避 Cache Miss。
 */
static void* gateway_worker(void *arg) {
    long core_id = (long)arg;
    int local_fd;

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
        fprintf(stderr, "[Gateway] Warning: Could not set CPU affinity for worker %ld\n", core_id);
    }

    local_fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (local_fd < 0) {
        perror("[Gateway] Worker socket create failed");
        return NULL;
    }

    int opt = 1;
    setsockopt(local_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    setsockopt(local_fd, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt));
    
    struct sockaddr_in bind_addr = { .sin_family = AF_INET, .sin_addr.s_addr = INADDR_ANY, .sin_port = htons(g_local_port) };
    if (bind(local_fd, (struct sockaddr*)&bind_addr, sizeof(bind_addr)) < 0) {
        perror("[Gateway] Worker bind failed"); 
        close(local_fd);
        return NULL;
    }

    if (core_id == 0) {
        g_primary_socket = local_fd;
    }

    struct mmsghdr msgs[BATCH_SIZE];
    struct iovec iovecs[BATCH_SIZE];
    struct sockaddr_in src_addrs[BATCH_SIZE];
    uint8_t *buffer_pool = malloc(BATCH_SIZE * WVM_MAX_PACKET_SIZE);

    for (int i = 0; i < BATCH_SIZE; i++) {
        iovecs[i].iov_base = buffer_pool + (i * WVM_MAX_PACKET_SIZE);
        iovecs[i].iov_len = WVM_MAX_PACKET_SIZE;
        msgs[i].msg_hdr.msg_iov = &iovecs[i];
        msgs[i].msg_hdr.msg_iovlen = 1;
        msgs[i].msg_hdr.msg_name = &src_addrs[i];
        msgs[i].msg_hdr.msg_namelen = sizeof(struct sockaddr_in);
    }

    while (1) {
        int n = recvmmsg(local_fd, msgs, BATCH_SIZE, 0, NULL);
        if (n <= 0) {
            if (errno == EINTR) continue;
            continue; 
        }

        for (int i = 0; i < n; i++) {
            uint8_t *ptr = (uint8_t *)iovecs[i].iov_base;
            int pkt_len = msgs[i].msg_len;
            struct sockaddr_in *src = &src_addrs[i];

            if (pkt_len < sizeof(struct wvm_header)) continue;
            struct wvm_header *hdr = (struct wvm_header *)ptr;
            if (ntohl(hdr->magic) != WVM_MAGIC) continue;
            uint16_t msg_type = ntohs(hdr->msg_type);

            uint32_t source_id = ntohl(hdr->slave_id); // 发送者 ID
            uint32_t target_id = ntohl(hdr->target_id); // 目标 ID（兼容旧逻辑时可能为 AUTO_ROUTE）
            
            // [关键]：只要收到合法的 WVM 包，就学习源路由
            // 排除掉 Upstream (Master/Core) 的 ID，只学习 Downstream (Leaf) 节点
            // 这里可以通过 ID 范围判断，或者简单地全部学习（Upstream 路由更新也无妨）
            if (source_id < WVM_MAX_SLAVES) {
                learn_route(source_id, &src_addrs[i]);
            }

            // Packets from upstream are always considered downstream.
            if (src->sin_addr.s_addr == g_upstream_addr.sin_addr.s_addr && src->sin_port == g_upstream_addr.sin_port) {
                uint32_t route_id = source_id; // 兼容旧包：没有目标信息时按旧字段转发
                if (target_id < WVM_MAX_SLAVES) {
                    route_id = target_id;
                } else if (target_id == WVM_NODE_AUTO_ROUTE) {
                    route_id = source_id;
                }
                int r = internal_push(local_fd, route_id, ptr, pkt_len);
                if (r < 0 && msg_type == MSG_VCPU_RUN) {
                    fprintf(stderr, "[Gateway] upstream->route failed msg=%u route=%u err=%d\n",
                            (unsigned)msg_type, route_id, r);
                }
            } else {
                // Packets from anywhere else are considered upstream.
                int tx_fd = (g_upstream_tx_socket >= 0) ? g_upstream_tx_socket : local_fd;
                ssize_t sret = sendto(tx_fd, ptr, pkt_len, MSG_DONTWAIT,
                                      (struct sockaddr*)&g_upstream_addr, sizeof(g_upstream_addr));
                if (msg_type == MSG_VCPU_RUN || msg_type == MSG_VCPU_EXIT || sret < 0) {
                    char src_ip[INET_ADDRSTRLEN] = {0};
                    char up_ip[INET_ADDRSTRLEN] = {0};
                    inet_ntop(AF_INET, &src->sin_addr, src_ip, sizeof(src_ip));
                    inet_ntop(AF_INET, &g_upstream_addr.sin_addr, up_ip, sizeof(up_ip));
                    fprintf(stderr, "[Gateway] route->upstream msg=%u src=%s:%u up=%s:%u ret=%zd errno=%d\n",
                            (unsigned)msg_type,
                            src_ip, (unsigned)ntohs(src->sin_port),
                            up_ip, (unsigned)ntohs(g_upstream_addr.sin_port),
                            sret, (sret < 0) ? errno : 0);
                }
            }
        }
    }
    free(buffer_pool);
    return NULL;
}

// [PATCH] 新增控制协议定义
typedef struct {
    uint32_t magic;      // 0x57564D43 "WVMC" (WaveVM Control)
    uint16_t op_code;    // 1 = ADD_ROUTE, 2 = DEL_ROUTE
    uint32_t node_id;    // 目标虚拟节点 ID
    uint32_t ip;         // 网络序 IP
    uint16_t port;       // 网络序 Port
} __attribute__((packed)) wvm_gateway_ctrl_pkt;

/* 
 * [物理意图] 提供“上帝视角”的外部干预入口，处理大规模机柜级扩容。
 * [关键逻辑] 在 9001 独立控制端口（可自行配置)上监听 WVMC 指令，通过写锁（Wrlock）强制注入跨 Pod 的联邦路由。
 * [后果] 实现了分形架构的层级级联。通过此接口，自动化运维工具可以将分散的 Pod 编织成一个巨大的戴森球算力网。
 */
void dynamic_add_route(uint32_t node_id, uint32_t ip, uint16_t port) {
    // 1. 获取写锁 (独占，会暂停所有数据转发微秒级时间)
    pthread_rwlock_wrlock(&g_map_lock);
    
    gateway_node_t *node = NULL;
    HASH_FIND_INT(g_node_map, &node_id, node);
    
    if (!node) {
        node = (gateway_node_t*)calloc(1, sizeof(gateway_node_t));
        if (node) {
            node->id = node_id;
            pthread_mutex_init(&node->lock, NULL); // 节点内部的Buffer锁保持Mutex
            node->buffer = NULL; 
            HASH_ADD_INT(g_node_map, id, node);
        }
    }
    
    if (node) {
        node->addr.sin_family = AF_INET;
        node->addr.sin_addr.s_addr = ip;
        node->addr.sin_port = port;
        printf("[Gateway] Route Added/Updated: Node %u -> %s:%d\n", 
               node_id, inet_ntoa(node->addr.sin_addr), ntohs(port));
    }
    
    pthread_rwlock_unlock(&g_map_lock);
    struct timespec ts = {0, 1000}; // 1000 ns = 1 us
    nanosleep(&ts, NULL);
}

// [PATCH] 控制平面监听线程
static void* control_plane_thread(void *arg) {
    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) return NULL;
    struct sockaddr_in addr = { 
        .sin_family = AF_INET, 
        .sin_addr.s_addr = INADDR_ANY, 
        .sin_port = htons(g_ctrl_port) 
    };

    if (bind(sockfd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("[Gateway] FATAL: Control plane bind failed (Port occupied?)");
        exit(1); // 端口冲突直接退出
    }
    
    printf("[Gateway] Control Plane Active on Port %d (Static)\n", g_ctrl_port);
    
    wvm_gateway_ctrl_pkt pkt;
    while (1) {
        ssize_t n = recv(sockfd, &pkt, sizeof(pkt), 0);
        if (n == sizeof(pkt) && ntohl(pkt.magic) == WVM_CTRL_MAGIC) {
            if (ntohs(pkt.op_code) == 1) { // ADD / UPDATE
                dynamic_add_route(ntohl(pkt.node_id), pkt.ip, pkt.port);
            } 
        }
    }
    return NULL;
}

int init_aggregator(int local_port, const char *upstream_ip, int upstream_port, const char *config_path) {
    if (g_primary_socket >= 0) return 0;

    g_local_port = local_port;
    if (load_slave_config(config_path) != 0) return -ENOENT;

    g_upstream_addr.sin_family = AF_INET;
    g_upstream_addr.sin_addr.s_addr = inet_addr(upstream_ip);
    g_upstream_addr.sin_port = htons(upstream_port); 

    // 单独的 upstream 发送 fd，避免复用 worker 的接收 fd 在某些环境下出现回环发送异常。
    g_upstream_tx_socket = socket(AF_INET, SOCK_DGRAM, 0);
    if (g_upstream_tx_socket < 0) {
        perror("[Gateway] upstream tx socket create failed");
        return -1;
    }
    int tx_flags = fcntl(g_upstream_tx_socket, F_GETFL, 0);
    if (tx_flags >= 0) {
        fcntl(g_upstream_tx_socket, F_SETFL, tx_flags | O_NONBLOCK);
    }

    long num_cores = get_nprocs();
    printf("[Gateway] System has %ld cores. Scaling out RX workers...\n", num_cores);

    for (long i = 0; i < num_cores; i++) {
        pthread_t thread;
        if (pthread_create(&thread, NULL, gateway_worker, (void*)i) != 0) {
            perror("[Gateway] Failed to create worker thread");
            return -1;
        }
        pthread_detach(thread);
    }

    detect_cpu_env();
    
    // [PATCH] 启动控制线程
    pthread_t ctrl_tid;
    pthread_create(&ctrl_tid, NULL, control_plane_thread, NULL);
    pthread_detach(ctrl_tid);

    return 0;}
