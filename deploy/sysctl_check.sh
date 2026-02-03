
#!/bin/bash
# GiantVM V29 Environment Check & Hardening Script

echo "[*] GiantVM V29: Tuning Kernel Parameters for Production..."

# 1. 基础资源限制 (Basic Resource Limits)
# 确保系统能打开足够多的文件句柄
sysctl -w fs.file-max=200000 > /dev/null
# V29的精细化脏区捕获依赖于大量的mprotect操作，必须极大增加VMA限制
sysctl -w vm.max_map_count=2000000 > /dev/null
# 预留一些大页，用于KVM后端和性能敏感的组件
sysctl -w vm.nr_hugepages=10240 > /dev/null

echo "[+] System resource limits increased."

# 2. UDP 缓冲区深井扩容 (UDP Buffer Deepening)
# 这是保证公网环境下不因突发流量（如元数据风暴或日志广播）而丢包的生命线
# 将发送和接收缓冲区均设置为50MB (默认仅约200KB)
sysctl -w net.core.rmem_max=52428800 > /dev/null
sysctl -w net.core.rmem_default=52428800 > /dev/null
sysctl -w net.core.wmem_max=52428800 > /dev/null
sysctl -w net.core.wmem_default=52428800 > /dev/null

echo "[+] UDP network buffers boosted to 50MB."

# 3. 网络设备队列长度 (Network Device Queue)
# 增加内核处理网络包的队列长度，为高PPS（每秒包数）场景提供缓冲
sysctl -w net.core.netdev_max_backlog=10000 > /dev/null

echo "[+] Network device backlog queue increased."
echo "[SUCCESS] Kernel parameters are tuned for V29 'Wavelet' deployment."
