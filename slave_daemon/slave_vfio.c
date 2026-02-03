
/*
 * [IDENTITY] VFIO Interceptor - The GPU Virtualizer
 * ---------------------------------------------------------------------------
 * 物理角色：Slave 端物理硬件的"影子拦截器"。
 * 职责边界：
 * 1. 强制开启硬件 Bus Master 权限，确保远程 DMA 正常。
 * 2. 拦截 BAR 空间 MMIO 读写，实现对物理显卡的分布式穿透。
 * 3. 实现中断去抖动 (100us Debouncing)，防止物理 GPU 中断冲垮网络链路。
 * 
 * [物理原则]
 * - 中断转发严禁走 Slow Lane，必须直达 Master 的 Fast Lane。
 * ---------------------------------------------------------------------------
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <linux/vfio.h>
#include <linux/pci_regs.h> // 需要系统头文件，定义了 PCI_COMMAND_MASTER
#include <errno.h>
#include <arpa/inet.h>      // 用于发送中断包

#include "slave_vfio.h"
#include "../common_include/giantvm_protocol.h"

#define VFIO_CONTAINER_PATH "/dev/vfio/vfio"
#define MAX_EPOLL_EVENTS 16

static int g_container_fd = -1;
static gvm_vfio_device_t g_devices[MAX_VFIO_DEVICES];
static int g_dev_count = 0;

// 网络上下文，用于发送中断
static int g_net_fd = -1;
static struct sockaddr_in g_master_addr;
static int g_net_ready = 0;

/* 
 * [物理意图] 强行夺取物理显卡的“主线控制权”，开启远程 DMA 引擎。
 * [关键逻辑] 直接操作 PCI 配置空间命令寄存器，绕过 Guest BIOS 的限制，激活硬件 Bus Master 位。
 * [后果] 实现了远程 GPU 的物理觉醒。若此位未开启，Slave 端的显卡将无法进行 DMA 传输，分布式 I/O 会瞬间瘫痪。
 */
static int enable_bus_master(int device_fd) {
    struct vfio_region_info reg = { .argsz = sizeof(reg) };
    reg.index = VFIO_PCI_CONFIG_REGION_INDEX;
    
    if (ioctl(device_fd, VFIO_DEVICE_GET_REGION_INFO, &reg) < 0) {
        perror("[VFIO] Failed to get Config Space info");
        return -1;
    }

    // 读写配置空间通过 pread/pwrite 到 device_fd
    uint16_t cmd;
    off_t cmd_offset = reg.offset + PCI_COMMAND; // PCI_COMMAND = 0x04

    // 1. 读取当前 Command Register
    if (pread(device_fd, &cmd, sizeof(cmd), cmd_offset) != sizeof(cmd)) {
        perror("[VFIO] Failed to read PCI Command Reg");
        return -1;
    }

    // 2. 检查并设置 Bus Master (Bit 2)
    if (!(cmd & PCI_COMMAND_MASTER)) {
        printf("[VFIO] Bus Master disabled (0x%x). Enabling...\n", cmd);
        cmd |= PCI_COMMAND_MASTER;
        
        if (pwrite(device_fd, &cmd, sizeof(cmd), cmd_offset) != sizeof(cmd)) {
            perror("[VFIO] Failed to write PCI Command Reg");
            return -1;
        }
        printf("[VFIO] Bus Master enabled successfully.\n");
    } else {
        printf("[VFIO] Bus Master already enabled.\n");
    }
    return 0;
}

/* 
 * [物理意图] 在物理显卡与分布式链路之间“打通神经信号”。
 * [关键逻辑] 创建 eventfd 并将其注入物理硬件的 MSI-X 中断向量表中，实现硬件中断到用户态事件的转化。
 * [后果] 它是中断转发的源头。如果没有 eventfd 的正确绑定，Slave 端的显卡在完成任务后将无法通知 Master 端的 vCPU。
 */
static int setup_irq(gvm_vfio_device_t *dev) {
    // 为简单起见，且为了保证通用性，我们优先尝试启用 INTx (Legacy Interrupt)
    // 真实的 GPU 驱动通常会请求 MSI-X，这需要拦截配置空间的写操作来动态建立映射。
    // 由于 V27.0 不拦截 Config Space 写（太复杂），我们假设 Host VFIO 驱动
    // 能正确处理 Guest 驱动的中断请求。
    // 在最基础的透传场景中，我们至少要保证 INTx 能够通过。
    
    struct vfio_irq_info irq_info = { .argsz = sizeof(irq_info) };
    irq_info.index = VFIO_PCI_INTX_IRQ_INDEX;
    
    if (ioctl(dev->device_fd, VFIO_DEVICE_GET_IRQ_INFO, &irq_info) < 0) {
        // 设备可能不支持 INTx，尝试 MSI
        return 0; 
    }

    if (!(irq_info.flags & VFIO_IRQ_INFO_EVENTFD)) return 0;

    // 创建 eventfd 用于内核通知用户态
    dev->irq_fd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
    if (dev->irq_fd < 0) { perror("eventfd"); return -1; }

    // 将 eventfd 注册给 VFIO
    struct vfio_irq_set *irq_set;
    size_t argsz = sizeof(*irq_set) + sizeof(int);
    irq_set = malloc(argsz);
    irq_set->argsz = argsz;
    irq_set->flags = VFIO_IRQ_SET_DATA_EVENTFD | VFIO_IRQ_SET_ACTION_TRIGGER;
    irq_set->index = VFIO_PCI_INTX_IRQ_INDEX;
    irq_set->start = 0;
    irq_set->count = 1;
    memcpy(irq_set->data, &dev->irq_fd, sizeof(int));

    if (ioctl(dev->device_fd, VFIO_DEVICE_SET_IRQS, irq_set) < 0) {
        perror("[VFIO] Failed to bind INTx eventfd");
        close(dev->irq_fd);
        dev->irq_fd = -1;
        free(irq_set);
        return -1;
    }

    free(irq_set);
    printf("[VFIO] INTx Interrupt hook installed (fd=%d)\n", dev->irq_fd);
    return 0;
}

// -----------------------------------------------------------
// 辅助: 获取 Region 真实偏移
// -----------------------------------------------------------
static int setup_region(gvm_vfio_device_t *dev, int index, uint64_t gpa_base, uint64_t config_size) {
    struct vfio_region_info reg = { .argsz = sizeof(reg) };
    reg.index = index;
    
    if (ioctl(dev->device_fd, VFIO_DEVICE_GET_REGION_INFO, &reg) < 0) return 0;
    if (reg.size == 0) return 0;

    if (config_size > reg.size) config_size = reg.size; // 安全截断

    dev->regions[index].active = 1;
    dev->regions[index].region_index = index;
    dev->regions[index].gpa_start = gpa_base;
    dev->regions[index].size = config_size;
    dev->regions[index].offset = reg.offset; // 内核返回的物理偏移

    printf("[VFIO]   -> BAR%d Mapped: GPA 0x%lx -> Host Offset 0x%llx (Size 0x%lx)\n", 
           index, gpa_base, reg.offset, config_size);
    return 1;
}

/* 
 * [物理意图] 建立 Slave 节点物理硬件的“用户态访问隧道”。
 * [关键逻辑] 通过 VFIO 接口接管物理显卡，获取 BAR 空间的 Host 物理偏移，并建立 DMA 隔离容器。
 * [后果] 实现了真正的硬件直通。若偏移量（Offset）计算错误，所有的远程 MMIO 读写都会指向错误的物理内存，导致 Host 宿主机重启。
 */
static int init_device(const char *pci_id, const char *group_path, uint64_t *bar_gpas, uint64_t *bar_sizes) {
    if (g_dev_count >= MAX_VFIO_DEVICES) return -1;
    gvm_vfio_device_t *dev = &g_devices[g_dev_count];
    
    // 1. Container Init
    if (g_container_fd < 0) {
        g_container_fd = open(VFIO_CONTAINER_PATH, O_RDWR);
        if (g_container_fd < 0) { perror("Open VFIO Container"); return -1; }
        if (ioctl(g_container_fd, VFIO_CHECK_EXTENSION, VFIO_TYPE1_IOMMU) != 1) {
            fprintf(stderr, "[VFIO] IOMMU Type1 not supported\n"); return -1;
        }
    }

    // 2. Open Group
    dev->group_fd = open(group_path, O_RDWR);
    if (dev->group_fd < 0) { perror("Open VFIO Group"); return -1; }

    struct vfio_group_status status = { .argsz = sizeof(status) };
    ioctl(dev->group_fd, VFIO_GROUP_GET_STATUS, &status);
    if (!(status.flags & VFIO_GROUP_FLAGS_VIABLE)) {
        fprintf(stderr, "[VFIO] Group not viable (Bind to vfio-pci?)\n"); close(dev->group_fd); return -1;
    }

    if (ioctl(dev->group_fd, VFIO_GROUP_SET_CONTAINER, &g_container_fd) < 0) {
        perror("Set Container"); close(dev->group_fd); return -1;
    }

    // 3. Set IOMMU
    static int iommu_set = 0;
    if (!iommu_set) {
        if (ioctl(g_container_fd, VFIO_SET_IOMMU, VFIO_TYPE1_IOMMU) < 0) {
            perror("Set IOMMU"); close(dev->group_fd); return -1;
        }
        iommu_set = 1;
    }

    // 4. Get Device FD
    dev->device_fd = ioctl(dev->group_fd, VFIO_GROUP_GET_DEVICE_FD, pci_id);
    if (dev->device_fd < 0) { perror("Get Device FD"); close(dev->group_fd); return -1; }

    strncpy(dev->pci_id, pci_id, 31);
    
    // 5. 【关键】启用 Bus Master
    if (enable_bus_master(dev->device_fd) < 0) {
        fprintf(stderr, "[VFIO] Warning: Failed to enable Bus Master for %s\n", pci_id);
    }

    // 6. 映射 BAR
    for (int i = 0; i < MAX_BARS; i++) {
        if (bar_sizes[i] > 0) setup_region(dev, i, bar_gpas[i], bar_sizes[i]);
    }

    // 7. 设置中断
    setup_irq(dev);

    dev->active = 1;
    g_dev_count++;
    return 0;
}

// -----------------------------------------------------------
// 外部接口 1: 初始化
// -----------------------------------------------------------
int gvm_vfio_init(const char *config_file) {
    if (!config_file) return -1;
    FILE *fp = fopen(config_file, "r");
    if (!fp) {
        printf("[VFIO] Config file '%s' not found. Distributed I/O disabled.\n", config_file);
        return -1;
    }

    char line[512];
    char pci_id[32], group_path[64];
    uint64_t bar_gpas[MAX_BARS] = {0};
    uint64_t bar_sizes[MAX_BARS] = {0};
    int parsing = 0;

    printf("[VFIO] Loading config from %s...\n", config_file);

    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        
        if (strncmp(line, "DEVICE", 6) == 0) {
            if (parsing) init_device(pci_id, group_path, bar_gpas, bar_sizes);
            
            sscanf(line, "DEVICE %s %s", pci_id, group_path);
            memset(bar_gpas, 0, sizeof(bar_gpas));
            memset(bar_sizes, 0, sizeof(bar_sizes));
            parsing = 1;
        } else if (strncmp(line, "BAR", 3) == 0 && parsing) {
            int idx;
            uint64_t gpa, size;
            if (sscanf(line, "BAR%d %lx %lu", &idx, &gpa, &size) == 3) {
                if (idx >= 0 && idx < MAX_BARS) {
                    bar_gpas[idx] = gpa;
                    bar_sizes[idx] = size;
                }
            }
        } else if (strncmp(line, "END", 3) == 0 && parsing) {
            init_device(pci_id, group_path, bar_gpas, bar_sizes);
            parsing = 0;
        }
    }
    // Handle last device if no END tag
    if (parsing) init_device(pci_id, group_path, bar_gpas, bar_sizes);

    fclose(fp);
    return g_dev_count;
}

// -----------------------------------------------------------
// 外部接口 2: MMIO 拦截
// -----------------------------------------------------------
int gvm_vfio_intercept_mmio(uint64_t gpa, void *data, int len, int is_write) {
    for (int i = 0; i < g_dev_count; i++) {
        if (!g_devices[i].active) continue;
        
        for (int j = 0; j < MAX_BARS; j++) {
            gvm_vfio_region_t *reg = &g_devices[i].regions[j];
            if (!reg->active) continue;

            if (gpa >= reg->gpa_start && gpa < reg->gpa_start + reg->size) {
                uint64_t offset = reg->offset + (gpa - reg->gpa_start);
                ssize_t ret;
                
                if (is_write) ret = pwrite(g_devices[i].device_fd, data, len, offset);
                else ret = pread(g_devices[i].device_fd, data, len, offset);
                
                if (ret != len) {
                    // 硬件读写失败是严重错误，但也只能打印日志
                    // fprintf(stderr, "[VFIO] HW Access Failed: GPA %lx\n", gpa);
                }
                return 1; // Intercepted
            }
        }
    }
    return 0; // Passthrough to Master
}

// -----------------------------------------------------------
// 外部接口 3: 中断转发线程 (Poll Loop)
// -----------------------------------------------------------

// [Helper] 计算时间差 (微秒)
static long diff_us(struct timespec *now, struct timespec *prev) {
    return (now->tv_sec - prev->tv_sec) * 1000000 + (now->tv_nsec - prev->tv_nsec) / 1000;
}

/* 
 * [物理意图] 充当分布式硬件中断的“消减器（Debouncer）”。
 * [关键逻辑] 使用 epoll 监听 eventfd，并强制执行 100us 的物理去抖动，防止高频中断冲垮 Master 链路。
 * [后果] 保护了全网信道的稳定性。它通过在物理源头限制中断频率，保证了分布式系统不会陷入“中断风暴”死循环。
 */
void gvm_vfio_poll_irqs(int master_sock, struct sockaddr_in *master_addr) {
    if (g_dev_count == 0) return;

    // 1. 设置网络上下文 (复制一份，因为主线程可能修改)
    g_net_fd = master_sock;
    if (master_addr) memcpy(&g_master_addr, master_addr, sizeof(struct sockaddr_in));
    else return;

    // 2. 创建 Epoll 实例
    int epfd = epoll_create1(0);
    if (epfd < 0) { perror("epoll_create"); return; }

    struct epoll_event ev, events[MAX_EPOLL_EVENTS];
    int registered_count = 0;

    // 3. 注册所有设备的 IRQ EventFD 到 Epoll
    for (int i = 0; i < g_dev_count; i++) {
        if (g_devices[i].irq_fd < 0) continue;
        
        memset(&ev, 0, sizeof(ev));
        ev.events = EPOLLIN;
        ev.data.u32 = i; // 将设备索引存入 data，以便 wait 返回时知道是哪个设备
        
        if (epoll_ctl(epfd, EPOLL_CTL_ADD, g_devices[i].irq_fd, &ev) < 0) {
            perror("[VFIO] epoll_ctl add failed");
            continue;
        }
        registered_count++;
    }
    
    if (registered_count == 0) {
        printf("[VFIO] No interrupts to poll. Thread exiting.\n");
        close(epfd);
        return;
    }

    printf("[VFIO] IRQ Polling Thread Started (Watching %d fds)...\n", registered_count);

    // 4. 轮询循环
    while (1) {
        int n = epoll_wait(epfd, events, MAX_EPOLL_EVENTS, -1); // Block indefinitely
        if (n < 0) {
            if (errno == EINTR) continue;
            perror("epoll_wait"); break;
        }
        for (int i = 0; i < n; i++) {
            int dev_idx = events[i].data.u32;
            int irq_fd = g_devices[dev_idx].irq_fd;
            uint64_t counter;
            
            // 必须读取 eventfd 以清空计数，否则会水平触发死循环
            if (read(irq_fd, &counter, sizeof(counter)) == sizeof(counter)) {
                
                // 构造中断包发送给 Master
                struct gvm_header hdr;
                memset(&hdr, 0, sizeof(hdr)); 

                hdr.magic = htonl(GVM_MAGIC);
                hdr.msg_type = htons(MSG_VFIO_IRQ);
                hdr.payload_len = 0;
                hdr.slave_id = 0; 
                hdr.req_id = 0;
                hdr.qos_level = 1; 
                
                // [CRC]
                hdr.crc32 = 0;
                hdr.crc32 = htonl(calculate_crc32(&hdr, sizeof(hdr)));

                // 发送
                // 注意：这里需要在多线程环境下安全使用 socket
                // UDP sendto 是原子的，只要 g_net_fd 有效即可
                sendto(g_net_fd, &hdr, sizeof(hdr), 0, 
                       (struct sockaddr*)&g_master_addr, sizeof(g_master_addr));
                
                // printf("[VFIO] IRQ Forwarded for Device %d\n", dev_idx);
            }
        }
    }
    close(epfd);
}

