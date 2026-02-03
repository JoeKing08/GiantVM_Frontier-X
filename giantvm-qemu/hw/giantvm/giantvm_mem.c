
#include "qemu/osdep.h"
#include "exec/memory.h"
#include "qemu/mmap-alloc.h"
#include "sysemu/kvm.h"

/*
 * Memory Interception for Infinite Scale (V18 - Dirty Log Enabled)
 */

void giantvm_setup_memory_region(MemoryRegion *mr, uint64_t size, int fd) {
    void *ptr;

    // Mode A: fd is /dev/giantvm
    // Mode B: fd is /dev/shm/giantvm_ram
    ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    
    if (ptr == MAP_FAILED) {
        fprintf(stderr, "GiantVM: Failed to mmap guest memory from fd=%d. Error: %s\n", 
                fd, strerror(errno));
        exit(1);
    }

    // Register with QEMU
    memory_region_init_ram_ptr(mr, NULL, "giantvm-ram", size, ptr);
    
    // 启用脏页日志 (Dirty Logging)
    // 这是 Mode B 在 Linux 5.15 上实现写同步的唯一标准方法。
    // 它告诉 KVM：请追踪这块内存的写入情况。
    memory_region_set_log(mr, true, DIRTY_MEMORY_MIGRATION);

    fprintf(stderr, "GiantVM: Mapped %lu bytes (Dirty Logging ON).\n", size);
}

