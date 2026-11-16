# 阿里云 PAI-DSW A10 环境记录

更新日期：2026-11-16。该环境用于验证 Jittor CUDA、真实数据训练、完整论文通道配置和无机械臂仿真。

## 实例与持久化目录

| 项目 | 实际配置 |
| --- | --- |
| GPU | NVIDIA A10，23028 MiB |
| 驱动 / CUDA | 570.211.01 / 12.1.105 |
| CPU / 内存 | 8 vCPU / 30 GiB |
| 系统盘 | 100 GiB |
| 项目 | `/mnt/workspace/projects/flowpolicy-jittor` |
| FPJ-033 固定 commit checkout | `/mnt/workspace/projects/flowpolicy-jittor-fpj033`（commit `929769a`） |
| Jittor venv | `/mnt/workspace/venvs/flowpolicy` |
| 仿真 venv | `/mnt/workspace/venvs/flowpolicy-sim` |
| JIT 缓存 | `/mnt/workspace/.cache/jittor` |
| 日志 | `/mnt/workspace/logs` |
| 最终停机状态 | FPJ-037 完成后于 2026-11-16 15:50（UTC+8）再次确认“已停止” |

训练环境使用 Jittor 1.3.11.0、NumPy 1.26.4、Zarr 2.18.7；仿真环境使用 MuJoCo 3.3.7、Meta-World 3.0.0 和 Gymnasium 1.4.0。两个环境通过 JSON 行和 NumPy 数组交互，仿真依赖没有进入 Jittor 训练图。

## 启动方式

```bash
source /mnt/workspace/activate_flowpolicy.sh
```

该脚本激活训练 venv、设置 `JITTOR_HOME` 和 CUDA 路径，并进入项目目录。仿真命令显式使用：

```bash
MUJOCO_GL=egl /mnt/workspace/venvs/flowpolicy-sim/bin/python scripts/sim_server.py
```

## 已验证的七层证据

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

6. 更难任务的匹配预算试验：

   - `reach-wall-v3` 先通过 10/10 专家变体门槛并采集 20 条 × 120 步的确定性示范；
   - 使用与第 5 项相同的完整通道、1024 点、batch 64、30 epochs、seeds 0/1/2 和
     每 seed 20 个闭环 episode；
   - 每个 seed 1050 updates，最后训练/验证 loss 和 checkpoint SHA256 均已保存；
   - 结果为 4/20、5/20、10/20，即 **31.67% ± 16.07%**，平均策略延迟 3.775 ms；
   - 外部 `nvidia-smi` 采样峰值 10605 MiB；60 个 episode 均保留逐回合成功步数、延迟和
     `done_reason=step_cap`；
   - 公开脱敏证据位于 `evidence/fpj033/formal/`，不含数据集、checkpoint、私有路径或密钥。

7. 完整通道 target 单 seed 配对：

   - commit `a24ae558`，`reach-wall-v3` seed 0；完整通道、1024 点、batch 64、30 epochs、
     EMA 0.999、每组 1050 updates 和 20 个 200-step episode；
   - 除 `target_mode` 与 EMA 的 loss 角色外，数据哈希、网络、优化、seed 和评测配置一致，匹配检查通过；
   - released-code 为 10/20（50%），detached 为 6/20（30%）；策略延迟为 3.562/3.502 ms；
   - 训练峰值显存为 11053/11741 MiB，detached 多 688 MiB；验证 loss 为 `4.85e-6`/`1.12e-4`；
   - checkpoint 重载最大动作差为 `2.08e-5`/`1.19e-7`，不是零；所有 checkpoint 哈希已独立复核；
   - 首次运行在 code 完成后因周期 trainer checkpoint 占满系统盘而退出；清理已关闭、已哈希的冗余
     trainer checkpoint 后，`--reuse` 明确复用 code，只从头重跑 detached，最终返回 0；
   - 公开脱敏证据位于 `evidence/fpj036/`。实例与运行时间见 `cloud_metadata.json`。

## 证据边界

- CUDA、真实数据更新和完整通道单步现在可以声称“已验证”。
- 可以声称完成“两个任务的论文网络结构三 seed 有限预算训练”，并另行完成完整通道 target 三 seed
  配对；后者仅适用于 `reach-wall-v3`、30 epochs，不能声称完成论文 3000 epochs、37 任务配方。
- 不能用一次 1-epoch rollout 计算成功率。
- Meta-World 3.0 / MuJoCo 3.3.7 是替代仿真栈，不能声称与论文的 v2/mujoco-py 像素级一致。
- 本机 RTX 5060 的 CUDA 要求与云端 A10 不同：本机 Blackwell 需要较新的 toolkit；A10 已在 CUDA 12.1 上通过，不应再写成所有 GPU 都必须 CUDA 12.8。

## FPJ-037：只补缺失的 seeds 1/2

FPJ-037 不重复 seed 0，也不改写 `evidence/fpj036/`。云端只运行 code/detached × seeds 1/2，配置与
seed 0 保持一致。四个 cell、80 个新增 episode、配对配置和 checkpoint 哈希均已独立复核；合并
FPJ-036 seed 0 后得到：

| target | seed 0 | seed 1 | seed 2 | mean ± sample std |
|---|---:|---:|---:|---:|
| code | 10/20 | 5/20 | 2/20 | **28.33% ± 20.21%** |
| detached | 6/20 | 9/20 | 1/20 | **26.67% ± 20.21%** |

两组只差一个成功回合，当前有限预算不支持稳定排名。额外使用 seed-1 trainer 做了三组无重训诊断：
code EMA/code 为 2/20、code 在线/exact 为 3/20、detached EMA/code 为 10/20；对应在线/code 主结果
为 5/20 和 9/20。诊断只用于分离权重选择与动作切片，不计入三 seed 统计。

第一次诊断启动漏设 `MUJOCO_GL=egl`，在任何 episode 产生前退出；EGL simulator hello 探测通过后，
从已导出的同一 checkpoint 继续完成全部 60 个诊断 episode。公开包 `evidence/fpj037/` 不含这次失败
启动的日志，只在 metadata 中记录重试原因和边界。

## 额度控制

免费试用包每个周期 250 计算时，共 3 个周期。阿里云资源包使用明细中，FPJ-036 开机前最近一条
扣减记录是 2026-11-16 07:06:49：扣减前 200.743、扣减 5.265、扣减后 **195.478 计算时**；
实验在 07:48:50 开机，中间没有更晚的扣减记录。控制台同时提示抵扣信息可能延迟约 6 小时，因此
195.478 是开机前的**账面余额**，不用于推算尚未出账的实际消耗。

A10 配置页显示每小时消耗约 6.991 计算时。FPJ-036 只做一个配对 seed：首次尝试 3282 秒，清理
磁盘后的成功恢复 1752 秒；已完成的 code 单元没有重复训练或 rollout。trainer checkpoint 单个约
4.08 GB，独立复核哈希后删除；两组 policy checkpoint 保留在停止的系统盘，公开仓库只保存哈希。

实例在 2026-11-16 09:53 确认停止，FPJ-036 到此封闭。随后用户授权 FPJ-037 扩展，实例于同日重新
启动，只补 seeds 1/2。FPJ-037 结果和诊断经独立哈希核验并下载后，按精确路径删除 8 个最终/派生
checkpoint，共 `20,432,488,795` 字节；清理后 FPJ-037 目录内没有 `.ckpt`，系统盘占用从约
88.29 GiB 降至 69.26 GiB。实例于 2026-11-16 15:50 再次确认“已停止”，没有删除实例或结果 JSON。
被删权重只可通过重新训练恢复；公开仓库保存结果、配置、逐回合记录和权重哈希，不保存权重本体。
控制台提示免费云盘停止后保留 15 天，因此若还需从云盘取原始日志，应在该期限内处理。
