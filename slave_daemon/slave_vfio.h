
#ifndef SLAVE_VFIO_H
#define SLAVE_VFIO_H

#include <stdint.h>
#include <time.h>
#include <linux/vfio.h>

// 最大支持的透传设备数
#define MAX_VFIO_DEVICES 8
// 每个设备最多支持的 BAR 数量 (PCI 标准为 6)
#define MAX_BARS 6

typedef struct {
    int active;
    uint32_t region_index; // VFIO_PCI_BAR0_REGION_INDEX ...
    uint64_t gpa_start;    // 配置文件中定义的 Guest 物理起始地址
    uint64_t size;         // Region 大小
    uint64_t offset;       // 真实硬件在 device_fd 中的偏移量 (由内核告知)
} gvm_vfio_region_t;

typedef struct {
    char pci_id[32];       // e.g., "0000:01:00.0"
    char group_path[64];   // e.g., "/dev/vfio/12"
    int group_fd;
    int device_fd;
    
    // 中断支持
    int irq_fd;
    
    // [FIX] 防风暴限流器：记录上次发送中断的时间
    // 使用 CLOCK_MONOTONIC 保证不受 NTP 影响
    struct timespec last_irq_time; 
    
    gvm_vfio_region_t regions[MAX_BARS];
} gvm_vfio_device_t;

// 初始化 VFIO 子系统
int gvm_vfio_init(const char *config_file);

// 核心拦截接口：检查 GPA 是否命中，如果命中则执行硬件操作
// 返回 1 表示拦截处理成功，0 表示未命中 (需转发 Master)
int gvm_vfio_intercept_mmio(uint64_t gpa, void *data, int len, int is_write);

// 轮询所有设备的中断 (需要在独立线程调用)
void gvm_vfio_poll_irqs(int master_sock, struct sockaddr_in *master_addr);

#endif

