# WaveVM: GitHub Actions + Tailscale 测试落地步骤

本文目标：
1. 用 GitHub Actions 跑基础构建与 Mode B 多实例冒烟。
2. 用 Tailscale 把“临时实例”接入同一内网，做后续 Mode A / 多机链路验证。

## 1. 先启用仓库工作流

仓库已加入工作流文件：
- `.github/workflows/wavevm-ci.yml`

触发方式：
1. 打开 GitHub 仓库页面 -> `Actions`。
2. 选择 `wavevm-ci`。
3. 点击 `Run workflow`。
4. `run_modeb_smoke=true`，`run_modea_probe=true`。

输出说明：
- `build`：编译 `wavevm_gateway` / `wavevm_node_slave` / `wvm_ctl` / `wavevm_node_master`。
- `modeb-smoke`：在同一 runner 上启动双实例（多进程）冒烟。
- `modea-probe`：只检测 `/dev/kvm`、kvm 模块、内核头是否存在（不阻塞流程）。

## 2. 你需要在 GitHub 配置的 Secrets

如果后续要让 Actions 直接进 Tailscale 网络，建议先准备：
- `TS_AUTHKEY`：Tailscale auth key（建议 reusable + 短有效期）。
- `TS_TAILNET`：可选，tailnet 名称（例如 `example.ts.net`）。

> 当前 `wavevm-ci` 还没有默认启用 Tailscale 登录步骤，先保证基础 CI 稳定，再加远程链路步骤更安全。

## 3. 远程实例创建建议（你执行）

建议你先准备两台 Ubuntu 22.04/24.04 实例（可来自云厂商或其他平台）：
1. CPU 至少 2 vCPU，内存至少 4GB。
2. 开放 UDP 出站，允许安装 Tailscale。
3. 每台实例执行：
   - 安装 Tailscale
   - `tailscale up --authkey <你的key> --ssh`
4. 记录两台机器的 Tailscale IP（如 `100.x.y.z`）。

## 4. 我这边下一步（等你给 key）

你给我 `TS_AUTHKEY` 后，我会在当前 codespace 做：
1. 安装/启动 Tailscale 客户端。
2. 用 authkey 加入 tailnet。
3. 验证到你实例的连通性（`tailscale ping` / `ssh`）。
4. 给你一版“可直接复制”的远程多机测试命令（Mode B 先通，再看 Mode A）。

## 5. 建议的测试推进顺序

1. 先跑 `wavevm-ci`，确保 GitHub runner 上构建与本地多实例冒烟稳定。
2. 然后接入两台 Tailscale 实例，做跨机 Mode B 链路。
3. 最后在具备 `/dev/kvm` 与内核构建条件的机器上推进 Mode A。
