/*
 * [IDENTITY] Infrastructure - The Physical Axioms
 * ---------------------------------------------------------------------------
 * 物理角色：定义整个系统的"物理常数"和寻址边界。
 * 职责边界：
 * 1. 设定 WVM_SLAVE_BITS (20位ID空间)，支撑百万级虚拟节点。
 * 2. 规定内存粒度 (1GB 路由 / 2MB 订阅 / 4KB 一致性)。
 * 3. 设定物理 MTU (1400)，规避 Overlay 网络的 IP 分片。
 * 
 * [禁止事项]
 * - 严禁在不修改 Gateway 内存映射逻辑的情况下增大 WVM_SLAVE_BITS。
 * - 严禁将 MTU_SIZE 设为 1500 以上，否则公网环境 PPS 性能将下降 90%。
 * ---------------------------------------------------------------------------
 */
#ifndef WAVEVM_CONFIG_H
#define WAVEVM_CONFIG_H

/* 
 * [集群规模上限] 定义节点 ID 的位宽。
 * 设定为 20 位，意味着支持 2^20 = 1,048,576 个逻辑节点 ID。
 * 在分形架构中，这对应于全网唯一的虚拟节点 ID (Virtual ID)。
 * 注意：如果将其调大，需要同步调整网关的路由表内存分配大小。
 */
#ifndef WVM_SLAVE_BITS
#define WVM_SLAVE_BITS 20
#endif

// --- 网络字节序转换宏 ---
/*
 * [跨架构兼容性] 64位整数的主机序(Host)到网络序(Network)转换。
 * 物理意义：
 * 1. WaveVM 协议强制使用大端序 (Big-Endian) 在网络传输。
 * 2. 这里的实现兼容了 x86 (Little-Endian) 和其他架构。
 * 3. 若不转换，跨架构通信时 GPA 地址会解析错误，导致读写错误的物理内存。
 */
#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
    #define WVM_HTONLL(x) __builtin_bswap64((unsigned long long)(x))
    #define WVM_NTOHLL(x) __builtin_bswap64((unsigned long long)(x))
#else
    #define WVM_HTONLL(x) (x)
    #define WVM_NTOHLL(x) (x)
#endif

// --- 核心常量定义 ---
#define WVM_MAX_SLAVES (1UL << WVM_SLAVE_BITS)
#define WVM_MAX_GATEWAYS (WVM_MAX_SLAVES)
/*
 * [协议特征与物理限制]
 * WVM_MAGIC: 数据包特征码 "WVMX"，用于在接收端快速识别并丢弃非法/垃圾流量。
 * MTU_SIZE: 设定为 1400 字节（低于标准以太网 1500）。
 *           原因：预留 100 字节给 Overlay 网络（如 VXLAN/Geneve）的隧道头。
 *           后果：防止底层网络进行 IP 分片。IP 分片会引发 CPU 软中断飙升，是高性能网络的杀手。
 */
#define WVM_MAGIC 0x57564D58 // "WVMX"
#define WVM_SERVICE_PORT 9000
#define MTU_SIZE  1400       
#define WVM_MAX_PACKET_SIZE 65536
#define MAX_VCPUS 4096
#define WVM_CPU_ROUTE_TABLE_SIZE MAX_VCPUS

// --- 内存粒度定义 ---
/*
 * [一级：DHT 路由粒度] 1GB (2^30)。
 * 物理意义：决定哪个节点是 Directory Owner。
 * 作用：我们不对每个 4KB 页面做哈希路由，而是每 1GB 数据块归属同一个 Owner。
 * 效果：极大减少了哈希计算频率，利用了程序的空间局部性。
 */
#define WVM_ROUTING_SHIFT 30 
/*
 * [二级：兴趣订阅粒度] 2MB (2^21)。
 * 物理意义：HugePage 对齐的订阅单元。
 * 作用：当节点发送 DECLARE_INTEREST 时，是一次性订阅 2MB 范围内的所有页面。
 * 效果：减少了 512 倍的订阅信令交互 (相比 4KB)。
 */
#define WVM_AFFINITY_SHIFT 21
/*
 * [三级：MESI 一致性粒度] 4KB (2^12)。
 * 物理意义：最小的原子数据同步单元。
 * 作用：Diff 计算、版本号校验、写冲突检测都精确到 4KB。
 */
#define WVM_PAGE_SHIFT 12

#define WVM_DEFAULT_SHM_PATH "/wavevm_ram"

#endif // WAVEVM_CONFIG_H
