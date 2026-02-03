
#ifndef UNIFIED_DRIVER_H
#define UNIFIED_DRIVER_H
#include "../common_include/platform_defs.h"

struct dsm_driver_ops {
    void*    (*alloc_large_table)(size_t size);
    void     (*free_large_table)(void *ptr);
    void*    (*alloc_packet)(size_t size, int atomic);
    void     (*free_packet)(void *ptr);

    void     (*set_gateway_ip)(uint32_t gw_id, uint32_t ip, uint16_t port);
    int      (*send_packet)(void *data, int len, uint32_t target_id);

    void     (*fetch_page)(uint64_t gpa, void *buf); 
    void     (*invalidate_local)(uint64_t gpa);
    // handle_page_fault is now part of the logic_core, not the driver ops
    void     (*log)(const char *fmt, ...) GVM_PRINTF_LIKE(1, 2);
    int      (*is_atomic_context)(void);
    void     (*touch_watchdog)(void);

    uint64_t (*alloc_req_id)(void *rx_buffer); 
    void     (*free_req_id)(uint64_t id);
    uint64_t (*get_time_us)(void);
    uint64_t (*time_diff_us)(uint64_t start);
    int      (*check_req_status)(uint64_t id); 
    void     (*cpu_relax)(void);

    // [V29 Phase 0 New] Ops for robust retry logic
    void     (*get_random)(uint32_t *val);         // For Jitter calculation
    void     (*yield_cpu_short_time)(void);     // For sleeping in retry loop

    void   (*send_packet_async)(uint16_t msg_type, void* payload, int len, uint32_t target_id, uint8_t qos);
};

extern struct dsm_driver_ops *g_ops;
#endif