
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <arpa/inet.h>
#include "../common_include/giantvm_ioctl.h"
#include "../common_include/giantvm_config.h"

/*
 * GiantVM V29 Control Tool (Heterogeneous-Aware)
 * 
 * 职责：
 * 1. 解析 V27 风格的异构配置 (Cores/RAM)。
 * 2. [关键] 根据 RAM 大小自动展开 Virtual Nodes (虚拟节点)，实现 DHT 内存加权负载均衡。
 * 3. [关键] 根据 Cores 数量填充 CPU 路由表，实现算力加权调度。
 */

// 虚拟节点粒度：每 4GB RAM 对应 1 个 DHT 槽位 (Virtual Node)
#define GVM_RAM_UNIT_GB 4 

static uint32_t local_cpu_table[GVM_CPU_ROUTE_TABLE_SIZE];

typedef struct {
    int phys_id;    // 配置文件里的 BaseID (物理ID)
    char ip[64];
    int port;
    int cores;
    int ram_gb;
    
    // 计算属性
    int vnode_start; // 在 DHT 环上的起始虚拟 ID
    int vnode_count; // 拥有的虚拟节点数量 (权重)
} NodeInfo;

// 辅助：注入 CPU 路由表 (带错误熔断)
void inject_cpu_route(int dev_fd) {
    uint32_t chunk_size = 1024;
    // 动态分配，避免栈溢出
    size_t buf_size = sizeof(struct gvm_ioctl_route_update) + chunk_size * sizeof(uint32_t);
    struct gvm_ioctl_route_update *payload = malloc(buf_size);
    if (!payload) { perror("malloc"); exit(1); }

    printf("[*] Injecting CPU Topology (%d vCPUs)...\n", GVM_CPU_ROUTE_TABLE_SIZE);

    for (uint32_t i = 0; i < GVM_CPU_ROUTE_TABLE_SIZE; i += chunk_size) {
        uint32_t current_count = chunk_size;
        if (i + current_count > GVM_CPU_ROUTE_TABLE_SIZE) 
            current_count = GVM_CPU_ROUTE_TABLE_SIZE - i;

        payload->start_index = i;
        payload->count = current_count;
        memcpy(payload->entries, &local_cpu_table[i], current_count * sizeof(uint32_t));

        // [FIX] 严格检查返回值，实现 Fail-Fast
        // 如果中间某块注入失败，说明内核状态已不一致，必须立即停止部署
        if (ioctl(dev_fd, IOCTL_UPDATE_CPU_ROUTE, payload) < 0) {
            fprintf(stderr, "[FATAL] CPU Route Injection Failed at index %d (errno=%d). State is inconsistent!\n", i, errno);
            free(payload);
            close(dev_fd);
            exit(1); // 返回非零状态码，中断外部的 Shell 脚本
        }
    }
    free(payload);
    printf("[+] CPU Routing Table Injected Successfully.\n");
}

// 辅助：注入全局参数
void inject_global_param(int dev_fd, int slot, int value) {
    size_t buf_size = sizeof(struct gvm_ioctl_route_update) + sizeof(uint32_t);
    struct gvm_ioctl_route_update *payload = malloc(buf_size);
    
    payload->start_index = slot; 
    payload->count = 1;
    payload->entries[0] = (uint32_t)value;

    if (ioctl(dev_fd, IOCTL_UPDATE_MEM_ROUTE, payload) < 0) {
        perror("[-] Failed to inject global param");
    }
    free(payload);
}

// 辅助：注入单个网关条目
void inject_gateway(int dev_fd, int id, const char* ip, int port) {
    struct gvm_ioctl_gateway gw_cmd;
    gw_cmd.gw_id = id;
    gw_cmd.ip = inet_addr(ip);
    gw_cmd.port = htons(port);
    if (ioctl(dev_fd, IOCTL_SET_GATEWAY, &gw_cmd) < 0) {
        perror("[-] Warning: Failed to set gateway IP");
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <HETERO_CONFIG_FILE> <MY_PHYS_ID>\n", argv[0]);
        return 1;
    }

    const char *config_file = argv[1];
    int my_phys_id = atoi(argv[2]); // 这里的 ID 对应配置文件里的 BaseID

    int dev_fd = open("/dev/giantvm", O_RDWR);
    if (dev_fd < 0) {
        perror("[-] Failed to open /dev/giantvm");
        return 1;
    }

    FILE *fp = fopen(config_file, "r");
    if (!fp) { perror("[-] Config open failed"); return 1; }

    printf("[*] GiantVM V29.5 Control Tool (Heterogeneous Engine)\n");

    // 1. 解析配置并计算权重
    NodeInfo *nodes = malloc(sizeof(NodeInfo) * GVM_MAX_SLAVES);
    if (!nodes) { perror("malloc nodes"); return 1; }
    int node_count = 0;
    char line[256];
    
    int total_vnodes = 0; // DHT 环总大小

    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
        
        char keyword[16];
        int bid, port, cores, ram;
        char ip[64];
        
        if (sscanf(line, "%15s", keyword) != 1) continue;
        if (strcmp(keyword, "NODE") != 0) continue;

        if (sscanf(line, "%*s %d %63s %d %d %d", &bid, ip, &port, &cores, &ram) == 5) {
            
            // [FIX 1] 边界检查：防止数组越界
            if (node_count >= GVM_MAX_SLAVES) {
                fprintf(stderr, "[Warn] Max node limit (%lu) reached. Ignoring remaining config.\n", GVM_MAX_SLAVES);
                break; // 停止解析，保护内存
            }

            nodes[node_count].phys_id = bid;
            strncpy(nodes[node_count].ip, ip, 63);
            nodes[node_count].port = port;
            nodes[node_count].cores = cores;
            nodes[node_count].ram_gb = ram;
            
            int v_count = ram / GVM_RAM_UNIT_GB;
            if (v_count < 1) v_count = 1;
            
            nodes[node_count].vnode_start = total_vnodes;
            nodes[node_count].vnode_count = v_count;
            
            total_vnodes += v_count;
            node_count++;
        }
    }
    fclose(fp);
    printf("[+] Topology: %d Physical Nodes -> %d Virtual DHT Nodes (Weighted).\n", node_count, total_vnodes);

    // 2. 注入 Gateway 表 (基于虚拟节点 ID)
    // DHT 算法算出的是 0..total_vnodes-1 之间的虚拟 ID
    // 内核拿到虚拟 ID 后查 Gateway 表，必须能查到对应的物理 IP
    for (int i = 0; i < node_count; i++) {
        for (int v = 0; v < nodes[i].vnode_count; v++) {
            int v_id = nodes[i].vnode_start + v;
            // 将所有属于该物理机的虚拟 ID 都指向同一个 IP
            inject_gateway(dev_fd, v_id, nodes[i].ip, nodes[i].port);
        }
    }
    printf("[+] Gateway Table Expanded & Injected.\n");

    // 3. 构建 CPU 路由表 (基于物理核心数)
    // CPU 调度通常走 RPC (MSG_VCPU_RUN)，目标 ID 应该是该物理机的主 ID (通常是 vnode_start)
    int current_vcpu = 0;
    
    // 策略：按顺序分配 vCPU 到物理节点
    for (int i = 0; i < node_count; i++) {
        // 分配该节点拥有的 Cores 数量的 vCPU
        for (int c = 0; c < nodes[i].cores; c++) {
            if (current_vcpu < GVM_CPU_ROUTE_TABLE_SIZE) {
                // 指向该物理机的第一个虚拟 ID (Primary ID)
                local_cpu_table[current_vcpu++] = nodes[i].vnode_start;
            }
        }
    }
    // 填补剩余 vCPU (Round-Robin)
    int node_cursor = 0;
    while (current_vcpu < GVM_CPU_ROUTE_TABLE_SIZE) {
        local_cpu_table[current_vcpu++] = nodes[node_cursor].vnode_start;
        node_cursor = (node_cursor + 1) % node_count;
    }
    
    inject_cpu_route(dev_fd);

    // 4. 注入全局参数
    // Slot 0: Total Nodes (这里指 Total Virtual Nodes，用于 DHT 取模)
    inject_global_param(dev_fd, 0, total_vnodes);
    
    // Slot 1: My Node ID
    // 传入的 my_phys_id 是配置文件里的 BaseID。
    // 我们需要找到它对应的 vnode_start (Primary Virtual ID)，告诉内核“我是谁”
    int my_virtual_id = -1;
    for (int i = 0; i < node_count; i++) {
        if (nodes[i].phys_id == my_phys_id) {
            my_virtual_id = nodes[i].vnode_start;
            break;
        }
    }
    
    if (my_virtual_id == -1) {
        fprintf(stderr, "[-] Error: My Phys ID %d not found in config!\n", my_phys_id);
        close(dev_fd);
        return 1;
    }
    
    inject_global_param(dev_fd, 1, my_virtual_id);
    
    printf("[+] V29.5 Configured: Total V-Nodes=%d, My Primary V-ID=%d\n", total_vnodes, my_virtual_id);

    free(nodes);

    close(dev_fd);
    return 0;
}
