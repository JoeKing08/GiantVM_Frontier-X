
/*
 * [IDENTITY] Protocol Stack - The Wavelet Law
 * ---------------------------------------------------------------------------
 * 物理角色：系统的"通用语言"和逻辑时钟定义。
 * 职责边界：
 * 1. 定义 Wavelet (MSG_COMMIT/PUSH) 与 V28 MESI (MSG_INVALIDATE) 兼容指令集。
 * 2. 规定 gvm_header 结构，集成 QoS 分级与端到端 CRC32。
 * 3. 提供 is_next_version 版本判定算法，解决 UDP 乱序真理冲突。
 * 
 * [禁止事项]
 * - 严禁删除 Header 中的 epoch 字段 (它是解决网络分区脑裂的唯一钥匙)。
 * - 严禁在 Payload 结构体中添加非对齐字段。
 * ---------------------------------------------------------------------------
 */
#ifndef GIANTVM_PROTOCOL_H
#define GIANTVM_PROTOCOL_H

#include "giantvm_config.h"
#include "platform_defs.h"

/*
 * GiantVM V29.5 "Wavelet" Protocol Definition (FINAL FIXED)
 * Includes V28 MESI Fallback & V29 Prophet Extensions
 */

// --- 1. 网络消息类型 (Network Message Types) ---
enum {
    // [Core] 基础通信
    MSG_PING           = 0,
    MSG_MEM_READ       = 1, // Pull Request
    MSG_MEM_WRITE      = 2, // Push / Direct Write
    MSG_MEM_ACK        = 3, // Pull Response / Sync ACK
    
    // [VCPU] 远程调度
    MSG_VCPU_RUN       = 5,
    MSG_VCPU_EXIT      = 6,
    MSG_VFIO_IRQ       = 7,

    // [V28 MESI] 一致性协议 (之前漏掉的部分!)
    MSG_INVALIDATE           = 10, // Directory -> Owner: Revoke permission
    MSG_DOWNGRADE            = 11, // Directory -> Owner: M -> S
    MSG_FETCH_AND_INVALIDATE = 12, // Directory -> Owner: Fetch data & Invalid
    MSG_WRITE_BACK           = 13, // Owner -> Directory: Data writeback

    // [V29 Wavelet] 主动推送与版本控制
    MSG_DECLARE_INTEREST   = 25, // Client -> Directory: Subscribe
    MSG_PAGE_PUSH_FULL     = 26, // Directory -> Client: Full Page Update
    MSG_PAGE_PUSH_DIFF     = 27, // Directory -> Client: Diff Update
    MSG_COMMIT_DIFF        = 28, // Client -> Directory: Diff Commit
    MSG_FORCE_SYNC         = 29, // Directory -> Client: Version Conflict

    // [V29 Prophet] 语义透传
    MSG_RPC_BATCH_MEMSET   = 31, // Scatter-Gather Batch Command

    // [自治集群扩展]
    MSG_HEARTBEAT      = 40, // 周期性存活宣告与 Epoch Gossip
    MSG_VIEW_PULL      = 41, // 获取邻居的局部视图
    MSG_VIEW_ACK       = 42, // 返回局部视图数据
    MSG_NODE_ANNOUNCE  = 43  // 新节点上线宣告 (Warm-plug)
};

// --- 2. 通用包头 (Header) ---
struct gvm_header {
    uint32_t magic;
    uint16_t msg_type;
    uint16_t payload_len; 
    uint32_t slave_id;      // Source Node ID
    uint64_t req_id;        // Request ID / GPA (in some legacy cases)
    uint8_t  qos_level;     // 1=Fast, 0=Slow
    uint8_t  flags;
    uint8_t  mode_tcg;
    uint8_t  node_state;    // 发送者当前的生命周期状态
    uint32_t epoch;         // 发送者所处的逻辑周期
    uint8_t  padding;
    uint32_t crc32;         // End-to-End Integrity Check
} __attribute__((packed));

#define GVM_FLAG_ZERO 0x01

// --- 3. Payload 结构体 (Data Structures) ---

// [V29] 增量更新日志
struct gvm_diff_log {
    uint64_t gpa;
    uint64_t version;     
    uint16_t offset;
    uint16_t size;
    uint8_t  data[0];     // Variable length data
} __attribute__((packed));

// [V29] 全页推送
struct gvm_full_page_push {
    uint64_t gpa;
    uint64_t version;
    uint8_t  data[4096];
} __attribute__((packed));

// [V29] 带版本的读响应
struct gvm_mem_ack_payload {
    uint64_t gpa;
    uint64_t version;
    uint8_t  data[4096];
} __attribute__((packed));

// [V29] 物理段描述符
struct gvm_rpc_region {
    uint64_t gpa;
    uint64_t len;
} __attribute__((packed));

// [V29] 批量操作指令
struct gvm_rpc_batch_memset {
    uint32_t val;
    uint32_t count;
    // Followed by: struct gvm_rpc_region regions[];
} __attribute__((packed));

// 节点生命周期状态语义
enum {
    NODE_STATE_SHADOW   = 0, // 刚启动，对拓扑不可见
    NODE_STATE_WARMING  = 1, // 预热中，同步元数据与热点页
    NODE_STATE_ACTIVE   = 2, // 活跃，承载 Owner 权限
    NODE_STATE_DRAINING = 3, // 准备下线，只读不写
    NODE_STATE_OFFLINE  = 4  // 已失效 (本地判定)
};

// 心跳 Payload：用于实现无中心的 Epoch 共识
struct gvm_heartbeat_payload {
    uint32_t local_epoch;
    uint32_t active_node_count; // 本地观察到的活跃节点数
    uint16_t load_factor;       // 负载情况
    uint32_t peer_epoch_sum;    // 用于快速计算均值或直方图的特征值
} __attribute__((packed));

// 视图交换结构：用于发现新邻居
struct gvm_view_entry {
    uint32_t node_id;
    uint32_t ip_addr;
    uint16_t port;
    uint8_t  state;
} __attribute__((packed));

struct gvm_view_payload {
    uint32_t entry_count;
    struct gvm_view_entry entries[0];
} __attribute__((packed));

// --- 4. IPC 消息定义 (Local QEMU <-> Daemon) ---

struct gvm_ipc_header_t {
    uint32_t type;
    uint32_t len;
};

#define GVM_IPC_TYPE_MEM_FAULT       1
#define GVM_IPC_TYPE_MEM_WRITE       2
#define GVM_IPC_TYPE_CPU_RUN         3
#define GVM_IPC_TYPE_IRQ             4
#define GVM_IPC_TYPE_COMMIT_DIFF     5
#define GVM_IPC_TYPE_INVALIDATE      6 

// [V29 Fix] 正式定义 RPC 透传类型
#define GVM_IPC_TYPE_RPC_PASSTHROUGH 99

// 前后端分离路由占位符
#define GVM_NODE_AUTO_ROUTE 0x3FFFFFFF

#define GVM_CTRL_MAGIC 0x47564D43

#define SYNC_MAGIC 0x53594E43 

struct gvm_ipc_fault_req {
    uint64_t gpa;
    uint32_t len;
    uint32_t vcpu_id;
};

struct gvm_ipc_fault_ack {
    int status;
};

struct gvm_ipc_write_req {
    uint64_t gpa;
    uint32_t len;
};

typedef struct {
    uint64_t rax, rbx, rcx, rdx, rsi, rdi, rsp, rbp;
    uint64_t r8, r9, r10, r11, r12, r13, r14, r15;
    uint64_t rip, rflags;
    uint8_t sregs_data[512];
    uint32_t exit_reason;
    struct {
        uint8_t direction;
        uint8_t size;
        uint16_t port;
        uint32_t count;
        uint8_t data[8];
    } io;
    struct {
        uint64_t phys_addr;
        uint32_t len;
        uint8_t is_write;
        uint8_t data[8];
    } mmio;
} gvm_kvm_context_t;

typedef struct {
    uint64_t regs[16];
    uint64_t eip;
    uint64_t eflags;
    uint64_t cr[5];
    uint64_t xmm_regs[32];
    uint32_t mxcsr;
    uint32_t exit_reason;
    uint64_t fs_base, gs_base;
    uint64_t gdt_base, gdt_limit;
    uint64_t idt_base, idt_limit;
} gvm_tcg_context_t;

struct gvm_ipc_cpu_run_req {
    uint32_t mode_tcg;
    uint32_t slave_id;   // 如果设为 GVM_NODE_AUTO_ROUTE (0x3FFFFFFF)，由后端决定
    uint32_t vcpu_index; // 传递 vCPU 序号用于查表路由
    union {
        gvm_kvm_context_t kvm;
        gvm_tcg_context_t tcg;
    } ctx;
};

struct gvm_ipc_cpu_run_ack {
    int status;
    uint32_t mode_tcg;
    union {
        gvm_kvm_context_t kvm;
        gvm_tcg_context_t tcg;
    } ctx;
};

static inline uint64_t gvm_get_u64_unaligned(const void *ptr) {
    uint64_t val;
    memcpy(&val, ptr, 8);
    return GVM_NTOHLL(val);
}

// 版本判定
static inline int is_next_version(uint64_t local, uint64_t push) {
    uint32_t l_epoch = (uint32_t)(local >> 32);
    uint32_t l_cnt   = (uint32_t)(local & 0xFFFFFFFF);
    uint32_t p_epoch = (uint32_t)(push >> 32);
    uint32_t p_cnt   = (uint32_t)(push & 0xFFFFFFFF);

    if (l_epoch == p_epoch) return p_cnt == l_cnt + 1;
    if (p_epoch == l_epoch + 1) return p_cnt == 1; // 跨纪元判定
    return 0;
}

static inline int is_newer_version(uint64_t local, uint64_t push) {
    uint32_t l_epoch = (uint32_t)(local >> 32);
    uint32_t p_epoch = (uint32_t)(push >> 32);
    if (p_epoch > l_epoch) return 1;
    if (p_epoch < l_epoch) return 0;
    return (uint32_t)(push & 0xFFFFFFFF) > (uint32_t)(local & 0xFFFFFFFF);
}

#include "crc32.h"

#endif // GIANTVM_PROTOCOL_H

