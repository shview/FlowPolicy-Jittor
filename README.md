# FlowPolicy-Jittor

这是对 **FlowPolicy: Enabling Fast and Robust 3D Flow-based Policy via Consistency Flow Matching for Robot Manipulation**（AAAI 2025 Oral）的 Jittor 复现工程。

当前目标不是一次搬完 37 个任务，而是先完成一个可验证的全仿真闭环：

```text
MetaWorld 脚本专家
  -> reach-v3（10 条）+ button-press-wall-v3（20 条）示范（Zarr）
  -> Jittor 数据集与 FlowPolicy
  -> Jittor 训练与仿真 rollout
  -> 3 seeds 的成功率、推理时间与 NFE
```

## 复现边界

- 已完成的模型、Consistency Flow Matching 损失与动作推理使用 Jittor；后续训练循环和优化器也必须保持 Jittor，不把 PyTorch 混入训练图。
- MuJoCo、MetaWorld 和专家策略属于仿真环境，不需要改写成 Jittor。
- 仿真器与模型之间只传递 NumPy/Zarr 数据，避免 PyTorch 和 Jittor 共用同一计算图。
- 管线打通和当前低预算实验使用 `reach-v3` 与 `button-press-wall-v3`；`reach-wall-v3` 只保留为后续扩展候选，不混入现有结果（专家探测见 `env/metaworld_contract.md`）。
- 闭环评测通过进程桥完成：策略在 Jittor 环境，仿真在隔离的 sim-env（`scripts/sim_server.py` + `scripts/rollout.py`，见 D017）。

## 新机器快速开始

下面以 Linux/WSL、Python 3.10 为例。训练与仿真使用两个隔离环境，避免 MuJoCo 依赖进入
Jittor 训练环境：

```bash
git clone https://github.com/shview/FlowPolicy-Jittor.git
cd FlowPolicy-Jittor

# 1. Jittor 训练与测试环境
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[test]"
python scripts/check_environment.py --strict-train --allow-cpu
python -m pytest -q

# 2. 独立 MetaWorld/MuJoCo 仿真环境
deactivate
python3.10 -m venv .sim-env
.sim-env/bin/python -m pip install --upgrade pip setuptools wheel
.sim-env/bin/python -m pip install \
  "mujoco==3.3.7" "metaworld==3.0.0" "numpy==1.26.4" \
  "zarr>=2.12,<3" "imageio==2.37.0" "imageio-ffmpeg==0.6.0" "pillow==11.3.0"
export FLOWPOLICY_SIM_PYTHON="$PWD/.sim-env/bin/python"
```

先探测专家，再生成一条示范；训练和 rollout 回到 Jittor 环境执行：

```bash
MUJOCO_GL=egl "$FLOWPOLICY_SIM_PYTHON" scripts/task_probe.py \
  --task reach-wall-v3 --variants 10
MUJOCO_GL=egl "$FLOWPOLICY_SIM_PYTHON" scripts/record_demos.py \
  --record --task reach-wall-v3 --episodes 1 --max-steps 120 \
  --output data/reach-wall-v3.zarr

. .venv/bin/activate
python scripts/train_smoke.py --data data/reach-wall-v3.zarr \
  --epochs 2 --checkpoint-every 1 --out outputs/quick-start
python scripts/rollout.py --data data/reach-wall-v3.zarr \
  --checkpoint outputs/quick-start/policy.ckpt --task reach-wall-v3 \
  --sim-python "$FLOWPOLICY_SIM_PYTHON" --episodes 1 \
  --out outputs/quick-start-rollout
```

仓库不包含 Zarr 数据和 checkpoint。完整环境差异、EGL 配置与云端命令见
[环境说明](docs/ENVIRONMENT.md)和[阿里云记录](docs/CLOUD_ALIYUN.md)。

## 当前状态

- [x] 固定官方基线版本：FlowPolicy commit `86d10c1`。
- [x] 审计论文、官方代码、数据格式和仿真链路。
- [x] 核对本机 GPU、WSL 与 Python 环境。
- [x] 建立分阶段复现计划和答辩检查表。
- [x] 建立 Jittor CPU 开发环境；1D 基础算子和时间嵌入测试通过。
- [x] 完成 Conditional 1D U-Net 的 Jittor CPU 结构与梯度测试。
- [x] 完成 XYZ PointNet 与机器人状态编码器的 Jittor CPU 测试。
- [x] 完成官方三字段 limits 归一化、可选 identity 路径与 checkpoint 测试。
- [x] 完成公开代码口径的 Consistency Flow Matching loss CPU 数值与梯度测试。
- [x] 完成顶层 `FlowPolicy` CPU 集成、NFE=1 动作生成与带配置指纹的严格 checkpoint 回归。
- [x] 建立现代 MetaWorld/muJoCo 仿真环境（隔离 venv，`MUJOCO_GL=egl` 软件渲染），并记录与官方旧栈的差异（D016）。
- [x] 采集真实示范 `data/reach-v3.zarr`（10 条 × 120 步，同 seed 可复现）。
- [x] 采集第二个任务示范 `data/button-press-wall-v3.zarr`（20 条 × 120 步）。
- [x] 真实示范上的训练与闭环 rollout（`scripts/train_smoke.py`、`scripts/rollout.py`）。
- [x] EMA 权重与 trainer checkpoint（`src/flowpolicy_jittor/training/ema.py`，见 D018）。
- [x] 跨框架数值对照：独立 PyTorch 镜像对前向、loss 与全部梯度做 parity（`outputs/parity/parity.json`）。
- [x] 完整通道（512/1024/2048）CPU 单步前反向；本机 CUDA 探测结论为不可用并留证（`outputs/cuda/report.json`）。
- [x] 阿里云 PAI-DSW A10 环境完成真实 Jittor CUDA 检测；小通道真实数据训练和完整通道单步前反向均通过。
- [x] 低预算实验网格：2 任务 × 3 seeds × 20 episodes，一条命令复跑（`scripts/experiments.py`，见 `outputs/experiments/`）。
- [x] CPU 回归基线：`213 passed, 1 skipped`；主环境跳过的 PyTorch parity 已在独立 parity-env 完成 170 项检查。
- [x] GPU 可行性验证：A10 23 GB、CUDA 12.1、Jittor 1.3.11；`reach-v3` 真实数据完成 34 次参数更新并保存 checkpoint。
- [x] A10 论文结构有限预算实验：完整通道、3 seeds、每 seed 30 epochs + 20 个闭环 episode，配置、哈希和逐回合结果均已保存。
- [x] `reach-wall-v3` 本地门槛：10/10 专家变体合格；20 条正式示范可确定性重录；真实数据的计划停止/恢复训练和 5-episode 闭环 smoke 已通过。

### A10 论文结构有限预算结果（NFE=1）

| 任务 | 模型与预算 | 成功率（3 seeds 均值 ± 标准差） | 单次策略推理 | 峰值显存 |
| --- | --- | --- | --- | --- |
| `button-press-wall-v3` | 512/1024/2048 通道，30 epochs，batch 64 | **100% ± 0%**（20/20、20/20、20/20） | 3.811 ± 0.016 ms | 10726 MiB |

这张表验证的是 Jittor 完整网络能在 A10 上完成多 seed 训练并进入替代仿真闭环。三组共 60 个 episode，
成功步数为 107–152，结果存在明显天花板效应。它不是论文成功率：训练仅 30 epochs、只覆盖一个任务、
仿真栈不同且未启用论文式 detached EMA target。完整证据在
`artifacts/aliyun_dsw/formal_pilot_20261115/` 对应的工作区目录中。

### 目前的实验结果（CPU、小通道、10 epochs、NFE=1）

| 任务 | 示范 | 成功率（3 seeds 均值 ± 标准差） | 单次推理 |
| --- | --- | --- | --- |
| `button-press-wall-v3` | 20 条 | **83.33% ± 20.82%**（12/20、18/20、20/20） | 2.97 ms |
| `reach-v3` | 10 条 | **11.67% ± 2.89%**（2/20、3/20、2/20） | 2.93 ms |

这是**低预算 CPU 复现**的结果，不是论文指标：通道数远小于 512/1024/2048，训练只有 10 epochs，
仿真栈是文档化的替代实现（D016）。云端 CUDA 已验证，但没有用来重新跑这张 CPU 实验表；逐 episode 明细与失败类型见
`outputs/experiments/SUMMARY.md`。

当前已经完成 CPU 闭环证据链、A10 CUDA 可行性验证，以及完整 512/1024/2048 通道的三 seed
有限预算训练和 60 个闭环 episode。尚未完成论文 3000 epochs 配方、原版旧仿真栈逐项对齐、
第二个任务的完整通道训练以及成功/失败视频整理。代码已发布到
<https://github.com/shview/FlowPolicy-Jittor>；这不改变实验边界，仍不能写成“论文成功率已复现”。

阿里云环境、持久化路径、复跑命令和证据边界见 [云端环境记录](docs/CLOUD_ALIYUN.md)。
`reach-wall-v3` 的公开脱敏证据见 [FPJ-033 preliminary bundle](evidence/fpj033/preliminary/manifest.json)：
它只覆盖本地专家、数据、恢复训练和 5-episode smoke，完整通道三 seed 云端结果仍未完成。

公开代码在默认 `horizon=4 / n_obs_steps=2 / n_action_steps=4` 下从索引 1 开始切动作，因此实际只返回 3 步。本项目默认保留这一 `code-faithful` 行为，同时提供显式 `action_slice_mode="exact"` 对照；不会把修正后的 4 步结果混入基线。

后续训练器构造 Jittor AdamW 时必须使用 `policy.trainable_parameters()`，不能直接传 `policy.parameters()`；后者仍包含为上游 state 结构保留、但已冻结的 `var_est` 参数与 normalizer buffers。

策略 checkpoint 不只检查键和 shape：保存前会从当前运行对象重新生成配置指纹，拒绝构造后被篡改的 NFE、loss 或动作切片设置；加载前会验证实现版本、Jittor 版本、数值 dtype 与 NaN/Inf。文件采用同目录临时文件写入后原子替换，加载提交失败则回滚模型和 normalizer，避免留下半恢复状态。

详细步骤见 [复现计划](docs/REPRODUCTION_PLAN.md)，重要技术取舍见 [决策记录](docs/DECISIONS.md)，答辩准备见 [答辩指南](docs/DEFENSE_GUIDE.md)。

## 上游来源

- 论文：<https://arxiv.org/abs/2412.04987>
- 官方实现：<https://github.com/zql-kk/FlowPolicy>
- 审计版本：`86d10c14972b4eab083abec9026749c340f8e5ed`
- 官方实现采用 MIT License；迁移代码时保留原始版权与许可证说明。

