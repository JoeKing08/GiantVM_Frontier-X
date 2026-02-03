
#ifndef GIANTVM_IOCTL_H
#define GIANTVM_IOCTL_H

#include <linux/ioctl.h>
#include "../common_include/giantvm_protocol.h"

struct gvm_ioctl_gateway {
    uint32_t gw_id;
    uint32_t ip;   // Network byte order
    uint16_t port; // Network byte order
};

// Control Plane Injection
#define IOCTL_SET_GATEWAY _IOW('G', 1, struct gvm_ioctl_gateway)

#define IOCTL_GVM_REMOTE_RUN _IOWR('G', 2, struct gvm_ipc_cpu_run_req)

// 路由表更新结构体
struct gvm_ioctl_route_update {
    uint32_t start_index;
    uint32_t count;
    // 柔性数组，用户态需分配足够的空间
    // 对于 CPU 表是 uint32_t，对于 MEM 表是 uint16_t，这里统一用 u32 传输方便对齐
    uint32_t entries[0]; 
};

#define IOCTL_UPDATE_CPU_ROUTE _IOW('G', 3, struct gvm_ioctl_route_update)
#define IOCTL_UPDATE_MEM_ROUTE _IOW('G', 4, struct gvm_ioctl_route_update)
#define IOCTL_WAIT_IRQ _IOR('G', 5, uint32_t) 
// 返回值是触发中断的 IRQ 号 (简化起见，返回 1 表示有中断)

#include <linux/types.h>

struct gvm_mem_range {
    uint64_t start;
    uint64_t size;
};

// 动态内存布局结构体：支持最多 32 个不连续的 RAM 槽位
struct gvm_ioctl_mem_layout {
    uint32_t count;
    struct gvm_mem_range slots[32];
};

#define IOCTL_SET_MEM_LAYOUT   _IOW('G', 10, struct gvm_ioctl_mem_layout)
#define IOCTL_RPC_SYNC_ACK     _IOW('G', 11, uint8_t)
#define IOCTL_UPDATE_EPOCH _IOW('G', 20, uint32_t)

#endif // GIANTVM_IOCTL_H

