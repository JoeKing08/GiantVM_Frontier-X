
/*
 * [IDENTITY] GPU Placeholder & Shadow Proxy - The Hardware Impersonator
 * ---------------------------------------------------------------------------
 * 物理角色：远程 GPU 在 Master 节点的"本地替身"与"资源占位符"。
 * 
 * 核心存在意义：
 * 1. 启动欺骗（Boot-time Deception）：物理 GPU 位于 Slave 节点，Master 侧的 QEMU 
 *    在初始化 PCI 总线时由于看不见真实硬件，会报错退出或拒绝分配资源。
 *    Stub 通过注册相同的 VendorID/DeviceID，让 QEMU "以为" 显卡就在本地。
 * 2. 资源锚定（BAR Anchoring）：在 QEMU 启动阶段，强制在物理地址空间（GPA）中
 *    为显存（BAR1）和寄存器（BAR0）“圈地”。
 *    如果没有这个 Stub 提前占位，Guest OS 的 BIOS 和内核将无法为远程 GPU 分配
 *    正确的地址窗口，导致 Slave 端的物理 GPU 即使在线也无法被映射。
 * 3. 运行控制（Prophet Hook）：在骗过启动器并成功运行后，利用预留的控制寄存器
 *    作为入口，为 Guest 提供超光速内存同步（MSG_RPC_BATCH_MEMSET）能力。
 * 
 * [禁止事项]
 * - 严禁随意更改 BAR1 的 Size (必须与 Slave 端物理显存完全对齐)，否则会导致
 *   Master/Slave 之间的地址映射断层。
 * - 严禁移除 PCI 配置空间的伪装代码，否则分布式显卡将无法通过 Guest 的驱动自检。
 * ---------------------------------------------------------------------------
 */
#include "qemu/osdep.h"
#include "hw/pci/pci.h"
#include "hw/qdev-properties.h"
#include "exec/cpu-common.h"
#include "exec/memory.h"
#include "cpu.h"
#include "giantvm_protocol.h" 

// 寄存器偏移
#define REG_ADDR_LOW   0x00
#define REG_ADDR_HIGH  0x04
#define REG_SIZE_LOW   0x08
#define REG_SIZE_HIGH  0x0C
#define REG_VAL        0x10
#define REG_COMMAND    0x14 
#define CMD_OP_MEMSET  1

#define MAX_BATCH_REGIONS 512

typedef struct GvmGpuStubState {
    PCIDevice pdev;
    MemoryRegion bar0, bar1, bar2;
    
    // 状态寄存器
    uint64_t reg_addr;
    uint64_t reg_size;
    uint32_t reg_val;
    
    // QOM 属性
    uint32_t vendor_id;
    uint32_t device_id;
    uint32_t subsystem_vendor_id;
    uint32_t subsystem_id;
    uint32_t class_id;
    uint64_t bar0_size;
    uint64_t bar1_size;
    uint64_t bar2_size;
} GvmGpuStubState;

// 引用外部函数
extern int gvm_send_rpc_sync(uint16_t msg_type, void *payload, size_t len);

/* 
 * [物理意图] 执行“GVA 到 GPA 的跨时空翻译”，将 Guest 的软件意图转化为全网硬件动作。
 * [关键逻辑] 1. 强制预读触发缺页（Safety Map）；2. 合并连续物理段；3. 发起同步 RPC 广播。
 * [后果] 这是 Prophet 引擎的引信。它通过物理段合并，将原本需要数百万次网络交互的 memset 压缩为一次同步操作。
 */
static void handle_memset_command(GvmGpuStubState *s) {
    uint64_t gva = s->reg_addr;
    uint64_t remain = s->reg_size;
    uint32_t val = s->reg_val;
    
    // 分配 Batch 缓冲区
    size_t batch_alloc_size = sizeof(struct gvm_rpc_batch_memset) + 
                              MAX_BATCH_REGIONS * sizeof(struct gvm_rpc_region);
    struct gvm_rpc_batch_memset *batch = g_malloc0(batch_alloc_size);
    struct gvm_rpc_region *regions = (struct gvm_rpc_region *)(batch + 1);
    
    batch->val = htonl(val);
    int count = 0;
    CPUState *cpu = current_cpu;

    // --- 核心循环：GVA -> GPA 翻译与打包 ---
    while (remain > 0) {
        // [Safety 1] 强制预读：触发缺页异常
        // 这一步至关重要，防止 cpu_get_phys_page_debug 返回 -1
        uint8_t dummy;
        if (cpu_memory_rw_debug(cpu, gva, &dummy, 1, 0) != 0) {
            fprintf(stderr, "[GVM-Stub] Invalid GVA access or SegFault: %lx\n", gva);
            break; // 停止优化
        }

        // [Safety 2] 查表翻译
        hwaddr gpa = cpu_get_phys_page_debug(cpu, gva & TARGET_PAGE_MASK);
        if (gpa == -1) break; 

        gpa += (gva & ~TARGET_PAGE_MASK); // 加上页内偏移

        uint64_t page_remain = TARGET_PAGE_SIZE - (gpa & ~TARGET_PAGE_MASK);
        uint64_t chunk = (remain < page_remain) ? remain : page_remain;

        // [Optimization] 物理段合并
        if (count > 0 && regions[count-1].gpa + regions[count-1].len == gpa) {
            regions[count-1].len += chunk;
        } else {
            // 缓冲区满？发送并重置
            if (count >= MAX_BATCH_REGIONS) {
                batch->count = htonl(count);
                // 转换字节序 (Host -> Network)
                for(int i=0; i<count; i++) {
                    regions[i].gpa = GVM_HTONLL(regions[i].gpa);
                    regions[i].len = GVM_HTONLL(regions[i].len);
                }
                
                // [Blocking] 同步发送
                gvm_send_rpc_sync(MSG_RPC_BATCH_MEMSET, batch, 
                    sizeof(struct gvm_rpc_batch_memset) + count * sizeof(struct gvm_rpc_region));
                
                count = 0; // 重置计数器
            }
            
            // 记录新段
            regions[count].gpa = gpa;
            regions[count].len = chunk;
            count++;
        }

        gva += chunk;
        remain -= chunk;
    }

    // 发送剩余尾包
    if (count > 0) {
        batch->count = htonl(count);
        for(int i=0; i<count; i++) {
            regions[i].gpa = GVM_HTONLL(regions[i].gpa);
            regions[i].len = GVM_HTONLL(regions[i].len);
        }
        gvm_send_rpc_sync(MSG_RPC_BATCH_MEMSET, batch, 
            sizeof(struct gvm_rpc_batch_memset) + count * sizeof(struct gvm_rpc_region));
    }

    g_free(batch);
    
    // [Consistency] 刷新 TLB，确保 CPU 感知到内存变化
    tlb_flush_all_cpus_synced(cpu);
}

/* 
 * [物理意图] 拦截 Guest 驱动发出的“上帝指令（Control Plane Commands）”。
 * [关键逻辑] 监听对 BAR2 寄存器的写操作，根据寄存器地址（ADDR/SIZE/CMD）拼凑出 Prophet 引擎的执行上下文。
 * [后果] 它是软件语义到物理动作的“跨界桥梁”。通过此函数，Guest 的一个赋值语句就能转化为全网 PB 级内存的同步。
 */
static void gvm_stub_bar2_write(void *opaque, hwaddr addr, uint64_t val, unsigned size) {
    GvmGpuStubState *s = opaque;
    
    switch (addr) {
        case REG_ADDR_LOW:  s->reg_addr = (s->reg_addr & 0xFFFFFFFF00000000) | (val & 0xFFFFFFFF); break;
        case REG_ADDR_HIGH: s->reg_addr = (s->reg_addr & 0x00000000FFFFFFFF) | (val << 32); break;
        case REG_SIZE_LOW:  s->reg_size = (s->reg_size & 0xFFFFFFFF00000000) | (val & 0xFFFFFFFF); break;
        case REG_SIZE_HIGH: s->reg_size = (s->reg_size & 0x00000000FFFFFFFF) | (val << 32); break;
        case REG_VAL:       s->reg_val  = (uint32_t)val; break;
        case REG_COMMAND:   
            if (val == CMD_OP_MEMSET) handle_memset_command(s); 
            break;
    }
}

static const MemoryRegionOps gvm_stub_bar2_ops = {
    .write = gvm_stub_bar2_write,
    .endianness = DEVICE_NATIVE_ENDIAN,
    .valid = { .min_access_size = 4, .max_access_size = 4 }
};

/* 
 * [物理意图] 在 QEMU 内部“无中生有”地创造出一个符合 PCI 规范的显卡分身。
 * [关键逻辑] 伪造 VendorID (0x10de) 和 DeviceID，并向 QEMU 内存系统注册 32GB 级的虚拟 BAR 空间。
 * [后果] 解决了启动时的“存在性自洽”。如果没有 realize 过程，QEMU 会因为检测不到硬件而拒绝加载相关的分布式显卡驱动。
 */
static void gvm_gpu_stub_realize(PCIDevice *pci_dev, Error **errp) {
    GvmGpuStubState *s = GVM_GPU_STUB(pci_dev);

    pci_config_set_vendor_id(pci_dev->config, s->vendor_id);
    pci_config_set_device_id(pci_dev->config, s->device_id);
    pci_config_set_class(pci_dev->config, s->class_id);
    if (s->subsystem_id) {
        pci_set_word(pci_dev->config + PCI_SUBSYSTEM_VENDOR_ID, s->subsystem_vendor_id);
        pci_set_word(pci_dev->config + PCI_SUBSYSTEM_ID, s->subsystem_id);
    }
    pci_config_set_interrupt_pin(pci_dev->config, 1);

    if (s->bar0_size > 0) {
        memory_region_init(&s->bar0, OBJECT(s), "gvm-gpu-bar0", s->bar0_size);
        pci_register_bar(pci_dev, 0, PCI_BASE_ADDRESS_SPACE_MEMORY, &s->bar0);
    }
    if (s->bar1_size > 0) {
        memory_region_init(&s->bar1, OBJECT(s), "gvm-gpu-bar1", s->bar1_size);
        pci_register_bar(pci_dev, 1, 
                         PCI_BASE_ADDRESS_SPACE_MEMORY | PCI_BASE_ADDRESS_MEM_PREFETCH | PCI_BASE_ADDRESS_MEM_TYPE_64, 
                         &s->bar1);
    }

    // [V29] BAR2 初始化为 IO 回调
    if (s->bar2_size > 0) {
        memory_region_init_io(&s->bar2, OBJECT(s), &gvm_stub_bar2_ops, s, "gvm-gpu-bar2", s->bar2_size);
        pci_register_bar(pci_dev, 2, PCI_BASE_ADDRESS_SPACE_MEMORY, &s->bar2);
    }
    
    printf("[GVM-Stub] V29.5 Prophet Ready: BAR2 IO Intercept Active.\n");
}

static Property gvm_gpu_stub_properties[] = {
    DEFINE_PROP_UINT32("vendor_id", GvmGpuStubState, vendor_id, 0x10de), 
    DEFINE_PROP_UINT32("device_id", GvmGpuStubState, device_id, 0x1eb8), 
    DEFINE_PROP_UINT32("sub_vid", GvmGpuStubState, subsystem_vendor_id, 0x0), 
    DEFINE_PROP_UINT32("sub_did", GvmGpuStubState, subsystem_id, 0x0), 
    DEFINE_PROP_UINT32("class_id", GvmGpuStubState, class_id, 0x030000), 
    DEFINE_PROP_UINT64("bar0_size", GvmGpuStubState, bar0_size, 16 * 1024 * 1024),
    DEFINE_PROP_UINT64("bar1_size", GvmGpuStubState, bar1_size, 12UL * 1024 * 1024 * 1024),
    // 默认 BAR2 大小 4KB，足够映射寄存器
    DEFINE_PROP_UINT64("bar2_size", GvmGpuStubState, bar2_size, 4096), 
    DEFINE_PROP_END_OF_LIST(),
};

static void gvm_gpu_stub_class_init(ObjectClass *class, void *data) {
    DeviceClass *dc = DEVICE_CLASS(class);
    PCIDeviceClass *k = PCI_DEVICE_CLASS(class);
    k->realize = gvm_gpu_stub_realize;
    k->vendor_id = PCI_ANY_ID;
    k->device_id = PCI_ANY_ID;
    k->class_id = PCI_CLASS_DISPLAY_VGA;
    dc->hotpluggable = true;
    device_class_set_props(dc, gvm_gpu_stub_properties);
    set_bit(DEVICE_CATEGORY_DISPLAY, dc->categories);
}

static const TypeInfo gvm_gpu_stub_info = {
    .name          = "giantvm-gpu-stub",
    .parent        = TYPE_PCI_DEVICE,
    .instance_size = sizeof(GvmGpuStubState),
    .class_init    = gvm_gpu_stub_class_init,
    .interfaces = (InterfaceInfo[]) { { INTERFACE_PCIE_DEVICE }, { } },
};

static void gvm_gpu_stub_register_types(void) {
    type_register_static(&gvm_gpu_stub_info);
}
type_init(gvm_gpu_stub_register_types)
