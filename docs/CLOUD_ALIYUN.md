# 阿里云 PAI-DSW A10 环境记录

记录日期：2026-11-15。该环境用于验证 Jittor CUDA、真实数据训练、完整论文通道配置和无机械臂仿真。

## 实例与持久化目录

| 项目 | 实际配置 |
| --- | --- |
| GPU | NVIDIA A10，23028 MiB |
| 驱动 / CUDA | 570.211.01 / 12.1.105 |
| CPU / 内存 | 8 vCPU / 30 GiB |
| 系统盘 | 100 GiB |
| 项目 | `/mnt/workspace/projects/flowpolicy-jittor` |
| Jittor venv | `/mnt/workspace/venvs/flowpolicy` |
| 仿真 venv | `/mnt/workspace/venvs/flowpolicy-sim` |
| JIT 缓存 | `/mnt/workspace/.cache/jittor` |
| 日志 | `/mnt/workspace/logs` |
| 最终停机状态 | 2026-11-15 16:23 已手动停止，控制台记录累计服务时长约 2 小时 39 分钟 |

训练环境使用 Jittor 1.3.11.0、NumPy 1.26.4、Zarr 2.18.7；仿真环境使用 MuJoCo 3.3.7、Meta-World 3.0.0 和 Gymnasium 1.4.0。两个环境通过 JSON 行和 NumPy 数组交互，仿真依赖没有进入 Jittor 训练图。

## 启动方式

```bash
source /mnt/workspace/activate_flowpolicy.sh
```

该脚本激活训练 venv、设置 `JITTOR_HOME` 和 CUDA 路径，并进入项目目录。仿真命令显式使用：

```bash
MUJOCO_GL=egl /mnt/workspace/venvs/flowpolicy-sim/bin/python scripts/sim_server.py
```

## 已验证的五层证据

1. 严格环境检查：

   ```bash
   python scripts/check_environment.py --strict-train --probe-timeout 1200
   ```

   结果为 `strict_check.passed=true`，并实际同步执行 Jittor CUDA kernel。

2. 真实数据短训练：

   ```bash
   python scripts/train_smoke.py \
     --data data/reach-v3.zarr --epochs 1 --batch-size 32 \
     --num-points 1024 --device cuda \
     --out outputs/cloud_reach_gpu_smoke --overwrite
   ```

   共 1089 个训练窗口、34 次参数更新；89.53 秒，2.633 秒/step；生成 `policy.ckpt`。这是小通道工程验证，不是论文训练预算。

3. 完整论文通道单步：

   ```bash
   python scripts/cuda_smoke.py \
     --out outputs/cloud_full_channels_a10 --device cuda \
     --batch-size 1 --num-points 1024 --steps 1 --overwrite
   ```

   `down_dims=[512,1024,2048]`，255,138,308 个可训练参数；一次 forward/backward/update 为 14.86 秒，监控峰值显存 8307 MiB，结果有限。它只证明 A10 能运行该配置，不等于完成训练。

4. 无机械臂仿真与跨环境闭环：

   - EGL 无界面渲染得到 `[128,128,3]` RGB、`[1024,3]` 点云和 9 维机器人状态；
   - `reach-v3` 脚本专家第 51 步首次成功；
   - 1-epoch checkpoint 完成 60 步闭环，checkpoint reload 最大误差为 0；
   - 该短训练模型没有成功，done reason 为 `step_cap`。正确结论是接口通过、训练预算不足。

5. 完整通道多 seed 有限预算试验：

   - `button-press-wall-v3`，完整 `512/1024/2048` 通道，255,138,308 个可训练参数；
   - batch 64、30 epochs、1050 updates/seed，seeds 0/1/2；
   - 每个 seed 约 688 秒，外部监控峰值 10726 MiB；
   - 三个最后 epoch mean loss 均值 `6.819e-6`，验证 loss 均值 `7.360e-6`；
   - 每个 checkpoint 20 个闭环 episode，结果为 20/20、20/20、20/20；
   - NFE=1，三个 seed 的策略推理均值为 3.811 ± 0.016 ms；
   - 三组共 60 个 episode，成功步数 107–152，存在天花板效应。

## 证据边界

- CUDA、真实数据更新和完整通道单步现在可以声称“已验证”。
- 可以声称完成“论文网络结构的三 seed 有限预算训练”，但不能声称完成论文 3000 epochs、37 任务配方；
  当前只有一个任务、30 epochs，且未启用 detached EMA target。
- 不能用一次 1-epoch rollout 计算成功率。
- Meta-World 3.0 / MuJoCo 3.3.7 是替代仿真栈，不能声称与论文的 v2/mujoco-py 像素级一致。
- 本机 RTX 5060 的 CUDA 要求与云端 A10 不同：本机 Blackwell 需要较新的 toolkit；A10 已在 CUDA 12.1 上通过，不应再写成所有 GPU 都必须 CUDA 12.8。

## 额度控制

当前 A10 公共资源每小时消耗约 6.991 计算时。本轮先做显存/吞吐探测，再执行 3 seeds × 30 epochs ×
20 episodes；训练和 rollout 串行，完成证据下载后立即停止实例。

本轮设置了定时关机作为兜底，并在 2026-11-15 16:23 确认手动停止。控制台提示：免费云盘会保存环境，
但停止后 15 天内如果没有再次启动，云盘环境将被释放。因此如需补第二任务，应在 15 天内启动，
或从本地保存的代码、JSON、日志和截图恢复。三个 checkpoint 的 SHA256 和逐回合结果已经本地保存。
