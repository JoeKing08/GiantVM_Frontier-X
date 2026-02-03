
#ifndef LOGIC_CORE_H
#define LOGIC_CORE_H

#include "unified_driver.h"
#include "../common_include/giantvm_protocol.h"
#include <stdint.h>

// --- 初始化与配置 ---
int gvm_core_init(struct dsm_driver_ops *ops, int total_nodes_hint);
void gvm_set_total_nodes(int count);
void gvm_set_my_node_id(int id);

// --- 核心处理逻辑 (被 User/Kernel Backend 调用) ---
// 处理收到的网络包
void gvm_logic_process_packet(struct gvm_header *hdr, void *payload, uint32_t source_node_id);

// --- 缺页处理逻辑 (被 Fault Handler 调用) ---
// V28 兜底：拉取全页
// 返回 0 成功 (page_buffer 已填充), <0 失败
// version_out: 用于回传版本号给 V29 Wavelet 引擎
int gvm_handle_page_fault_logic(uint64_t gpa, void *page_buffer, uint64_t *version_out);

// [V29] 本地缺页快速路径 (内核态专用)
int gvm_handle_local_fault_fastpath(uint64_t gpa, void* page_buffer, uint64_t *version_out);

// [V29] 宣告兴趣 (异步)
void gvm_declare_interest_in_neighborhood(uint64_t gpa);

// --- RPC 接口 ---
int gvm_rpc_call(uint16_t msg_type, void *payload, int len, uint32_t target_id, void *rx_buffer, int rx_len);

// --- 路由接口 ---
uint32_t gvm_get_directory_node_id(uint64_t gpa);

void update_local_topology_view(uint32_t src_id, uint32_t src_epoch, uint8_t src_state, struct sockaddr_in *src_addr);

// 计算任务路由 (V27 遗留，用于 RPC 调度)
uint32_t gvm_get_compute_slave_id(int vcpu_index);

void gvm_set_mem_mapping(int slot, uint32_t value);

void gvm_set_cpu_mapping(int vcpu_index, uint32_t slave_id);

#endif // LOGIC_CORE_H

