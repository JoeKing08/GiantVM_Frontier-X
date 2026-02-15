#pragma once

#include "qemu/osdep.h"
#include "qom/object.h"
#include "sysemu/accel.h"
#include "qemu/thread.h"

/*
 * Shared WaveVM accelerator definitions (used across wavevm-*.c files).
 * Keep this header minimal: it only defines the accelerator state and
 * identifiers needed by other compilation units.
 */

#define TYPE_WAVEVM_ACCEL "wavevm-accel"
#define WAVEVM_ACCEL(obj) OBJECT_CHECK(WaveVMAccelState, (obj), TYPE_WAVEVM_ACCEL)

typedef enum {
    WVM_MODE_KERNEL,
    WVM_MODE_USER,
} WvmMode;

typedef struct WaveVMAccelState {
    AccelState parent_obj;
    int dev_fd;
    int sync_sock;
    int ipc_sock;
    WvmMode mode;
    QemuThread sync_thread;
    QemuThread ipc_thread;
    QemuThread irq_thread;
    bool sync_thread_running;
    QemuThread net_thread;
    int master_sock;
} WaveVMAccelState;

extern int g_wvm_local_split;

