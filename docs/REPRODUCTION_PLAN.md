# FlowPolicy 的 Jittor 复现计划

## 1. 复现目标与验收口径

本项目要回答三个问题：

1. FlowPolicy 的核心模型与 CFM 损失能否在 Jittor 中正确实现？
2. 在不使用真实机械臂的情况下，能否完成“示范生成—训练—在线仿真评测”的闭环？
3. 单步生成（NFE=1）是否能在保持可用成功率的同时降低推理延迟？

最低完成标准：

- `reach-v3` 的 10 条和 `button-press-wall-v3` 的 20 条专家示范可重复生成并被 Jittor 读取；
- 模型可在小样本上过拟合，loss 明显下降；
- 能在 MetaWorld 中连续执行动作并输出成功率；
- 训练和推理的核心计算图中不使用 PyTorch；
- 发布代码、环境说明、随机种子、训练曲线与失败案例。

论文规模目标（当前尚未完成）：

- 至少两个 MetaWorld 任务；
- 每个任务 3 个随机种子；
- 每次正式评测 20 个 episode；
- 同时报告成功率、每步推理时间、NFE、参数量和显存峰值；
- 同时报告最终 checkpoint 和论文所用“训练期间最高 5 次评测均值”，避免只挑最好结果。

## 2. Phase 0：一手资料与代码审计（已完成）

### 资料

- FlowPolicy 论文，arXiv `2412.04987`。
- 官方仓库 commit `86d10c1`。
- Jittor 官方文档和官方仓库（具体 API 核验后写入实现注释）。

### 已确认事实

- 全部 37 个实验均为仿真任务：34 个 MetaWorld、3 个 Adroit。
- 论文主要配置：10 条示范、batch size 128、AdamW `1e-4`、3000 epochs、每 200 epochs 评测 20 episodes、3 个种子。
- 多数任务在 500–1000 epochs 已接近收敛，因此先做短训练排错，再决定是否跑满 3000 epochs。
- `button-press-wall` 有官方任务配置和 heuristic expert，动作 4 维、机器人状态 9 维，适合作为首个任务。

### 官方代码中必须显式处理的问题

- shell 脚本和任务配置存在 `/data/code/FlowPolicy/...` 绝对路径。
- `train_policy.sh` 将 GPU 参数误写为 `${0}`。
- `eval_policy.sh` 使用未定义变量。
- 生成示范的代码可能自动递归删除已有目录。
- MetaWorld 训练示范默认 1024 点，但部分 YAML/runner 使用 512 点。
- 环境 `seed()` 未实现，不能只设置模型 seed 就声称严格可复现。
- 论文描述的 EMA target 与公开代码实际 loss 不完全一致。

### 验证清单

- [x] 确认任务、示范、数据集、policy 和 runner 的调用链。
- [x] 确认输入输出张量约定。
- [x] 确认官方仓库不含训练好的 FlowPolicy checkpoint 和生成好的 Zarr 数据。
- [x] 确认本机 WSL2 能识别 NVIDIA GPU。

### 防止走偏

- 不把 `.pyc` 文件当作可复现的 baseline 源码。
- 不把“代码跑起来”直接等同于“论文结果已复现”。
- 不隐去论文与代码的差异；先定义复现口径再做实验。

## 3. Phase 1：环境与仿真最小闭环

### 实施方式

实际建立两个隔离环境：

1. `sim-env`：Python 3.10.14、MuJoCo 3.3.7、MetaWorld 3.0.0、Gymnasium 1.4.0，用于示范生成和 rollout。
2. `flowpolicy-jittor`：Python 3.10.14、Jittor、Zarr、NumPy 和训练依赖，用于模型训练和推理。

两者以 NumPy/Zarr 和 JSON 行协议为边界。现代仿真栈与上游旧栈的差异见 D016；正式长训练需要迁移到与具体 GPU 架构兼容的 Linux CUDA 环境。云端 A10 已在 CUDA 12.1 上通过，本机 Blackwell GPU 才需要更高版本 toolkit。

### 逐级验证

1. 导入现代 `mujoco`、`metaworld` 与 `gymnasium`。
2. EGL 无界面渲染 RGB 和 depth，并核对世界坐标几何。
3. 创建 `reach-v3` / `button-press-wall-v3` 并检查 observation 字段。
4. 运行脚本专家，得到成功 episode。
5. 生成 Zarr，并核对数组 shape、dtype、episode 边界和成功步数。
6. 生成 10/20 条正式示范；写入器默认拒绝覆盖已有目录。

### 验证清单

- [x] 物理 step、RGB、depth、点云均正常。（FPJ-025，D016）
- [x] 点云数量在生成、训练和评测中统一为 1024。
- [x] 专家 episode 的 `info["success"]` 可正确记录。
- [x] 相同 seed 的示范逐位一致；环境和线程等边界已写入说明。（FPJ-026、FPJ-031）

### 防止走偏

- 不在 Windows 原生安装已移除 Windows 支持的 `mujoco-py`。
- 不把现代 MuJoCo/Gymnasium/MetaWorld 静默写成原版环境；替代原因和差异必须引用 D016。
- 不直接执行带硬编码路径或自动删除行为的上游脚本。
- 不以“能 import”代替渲染和点云的完整测试。

## 4. Phase 2：Jittor 数值核心

### 实现顺序

1. 张量 shape 与配置数据类。
2. 归一化器。
3. 正弦时间嵌入。
4. PointNet XYZ encoder 与 state MLP。
5. Conditional 1D U-Net：Conv1d、GroupNorm、Mish、FiLM、上/下采样。
6. CFM 插值与 endpoint/velocity consistency loss。
7. `FlowPolicy.compute_loss()` 和 `FlowPolicy.predict_action()`。
8. EMA、checkpoint 和 optimizer state。

保留与上游一致的公开接口：

```python
policy.set_normalizer(normalizer)
loss, loss_dict = policy.compute_loss(batch)
result = policy.predict_action(obs_dict)
```

### 验证清单

- [x] 每个核心模块都有 shape test 和有限值检查。
- [x] 小配置下固定输入和同名权重的 Jittor/PyTorch 前向、loss 和 168 个梯度在容差内一致。（FPJ-028）
- [x] CFM 的时间采样、broadcast 和当前 `target_mode="code"` 梯度语义已有测试；论文式 detached EMA target 尚未实现。
- [x] 合成数据上的 10-batch 过拟合测试通过。（FPJ-023：32 个固定窗口，动作误差 5.49 → 1.42，降低 74%；loss 1.4e-4 → 1.3e-5）
- [x] policy/trainer checkpoint 保存后重载，固定输入结果一致；配置不匹配时拒绝加载。

### 防止走偏

- 不把 PyTorch Tensor 混入 Jittor 计算图。
- 不凭名称猜测 Conv1d/ConvTranspose1d 的参数和输出长度；先用官方 API 做最小 shape test。
- 不一开始缩小网络并把结果称作原论文复现；若因 8GB 显存使用缩小版，明确标记为 smoke 配置。
- 不只比较 loss 数字；同时比较中间特征 shape、输出分布和梯度是否存在。

## 5. Phase 3：数据、训练与本机 smoke test

### 数据契约

```text
point_cloud: [B, T_obs=2, N=1024, 3]
agent_pos:   [B, T_obs=2, 9]
action:      [B, horizon=4, 4]
```

### 训练阶梯

1. 读取 1 个 episode，逐字段检查。
2. 对 32 个窗口过拟合，确认模型能记住示范。
3. 缩小通道的 10-step 单元 smoke test。
4. 小通道的 100 epochs smoke training；正式通道先做单步资源探测，再在 A10 上完成 batch 64、
   30 epochs、3 seeds 的有限预算训练。
5. 先做 5 个 episode smoke 确认接口，再对每个正式 checkpoint 做 20 个闭环 episode。

### 验证清单

- [x] train/validation 按 episode 划分，避免相邻窗口泄漏。（FPJ-022，`build_window_indices` + `episode_validation_mask`）
- [x] 归一化统计量只由训练集拟合。（FPJ-022，`MetaworldZarrDataset.get_normalizer`）
- [x] loss、学习率、显存、迭代时间写入结构化日志。（FPJ-024：`Trainer` 逐步记录 `step`/`lr`/`loss` 到 JSONL；
     显存只在 CUDA 路径有意义，CPU 结果为“未测量”，迭代时间由外部计时脚本记录）
- [x] NaN/Inf、shape mismatch 和空 episode 立即失败。（FPJ-022）
- [x] 训练 checkpoint 可续训：policy 状态、AdamW 矩、`n_step`、`global_step` 逐位恢复，调度精确延续。
     （FPJ-024、FPJ-031；长进程续训轨迹容差 1e-2，见 D014）
- [x] 演示数据写入器按官方布局（`data/` + `meta/episode_ends`）落盘且默认拒绝覆盖。
     （FPJ-025：`scripts/record_demos.py --self-test`）
- [x] 数据集同时支持扁平布局与官方 `data/` + `meta/` 布局。（FPJ-025，D015）
- [x] 真实示范落到 `data/reach-v3.zarr`：10 条 × 120 步，成功步数写入 `meta/success_steps`，
      同 seed 重录逐位一致。（FPJ-026）
- [x] 真实数据上的训练 + 闭环 rollout 命令化：`scripts/train_smoke.py`（JSONL 日志、checkpoint）、
      `scripts/sim_server.py` + `scripts/rollout.py`（跨环境进程桥，D017）。（FPJ-027）

### 复跑命令（本机 CPU）

```bash
# 训练（小通道 CPU smoke，非论文配置）
OMP_NUM_THREADS=4 python scripts/train_smoke.py \
    --data data/reach-v3.zarr --epochs 100 --out outputs/smoke

# 闭环 rollout（策略在 Jittor 环境，仿真在隔离 sim-env）
OMP_NUM_THREADS=2 MUJOCO_GL=egl python scripts/rollout.py \
    --data data/reach-v3.zarr --checkpoint outputs/smoke/policy.ckpt \
    --episodes 5 --max-steps 200 --out outputs/rollout
```

### 防止走偏

- 不先跑 3000 epochs 再排查数据错位。
- 不用随机拆窗口代替按 episode 划分。
- 不只看训练 loss；至少做闭环 rollout。

## 6. Phase 4：低预算定量实验与后续对照

### 云端论文结构试验

A10 上使用完整 `512/1024/2048` 通道、255,138,308 个可训练参数、1024 点、batch 64、
30 epochs 和 seeds 0/1/2。每个 seed 为 1050 次更新，平均训练 688.14 秒，峰值显存
10726 MiB。每个 checkpoint 做 20 个闭环 episode，得到 20/20、20/20、20/20；
三个 seed 的策略推理均值为 3.811 ± 0.016 ms，NFE=1。

这是论文网络结构的有限预算结果，不是完整论文配方：训练 epoch 更少，当前只有一个任务，
仿真栈不同，且 detached EMA target 未启用。满分也提示 `button-press-wall-v3` 存在天花板效应；
下一步优先增加更难任务或代表性失败案例，而不是继续增加容易任务的 seed。

### 任务安排

- 当前低预算网格：`button-press-wall-v3` 与 `reach-v3`，因为两者已有正式示范、checkpoint 和逐 episode 结果。
- 后续扩展候选：`reach-wall-v3`（脚本专家探测 159/200），在采集正式示范并完成同口径训练前不列入现有对比。
- 暂不从 Adroit 开始，避免同时引入高维灵巧手、VRL3 checkpoint 和额外环境差异。
- 不要用 `door-open-v3`：本机专家只有 30/200 成功，示范质量不足。

### 一条命令跑完整个网格

```bash
MUJOCO_GL=egl python scripts/record_demos.py --record --task button-press-wall-v3 \
    --episodes 20 --max-steps 120 --seed 0 --output data/button-press-wall-v3.zarr

python scripts/experiments.py --tasks reach-v3 button-press-wall-v3 \
    --seeds 0 1 2 --episodes 20 --max-steps 200 --epochs 10 --parallel 3 \
    --out outputs/experiments
```

`scripts/experiments.py` 对每个 (task, seed) 真跑一次 `train_smoke.py` 和 `rollout.py`，
把命令行、checkpoint、逐 episode 结果写进 `outputs/experiments/<task>/seed-<n>/`，
并汇总成 `experiments.json` 与 `SUMMARY.md`（均值 ± 样本标准差、延迟、NFE、`done_reason` 分布）。
训练单元格可并发（`--parallel`），rollout 一律串行，避免 CPU 争用污染延迟数据。

### 对照与消融

- 上游 PyTorch FlowPolicy：同一示范、同一任务、尽量相同配置。
- Jittor FlowPolicy：正式结果。
- 可选消融：统一 1024 点 vs. 官方 runner 的 512 点。
- 可选消融：公开代码的同网络 target vs. 论文描述的 EMA/stop-gradient target。

### 验证清单

- [x] 3 seeds × 每次 20 episodes（`scripts/experiments.py` + `outputs/experiments/`）。
- [x] 记录硬件、软件、commit、配置和数据 manifest（`experiments.json` 的 `config`、`env/metaworld_contract.md`、`summary.json`）。
- [x] 报告均值和标准差，不只报最好 seed（`SUMMARY.md`）。
- [ ] 展示成功和失败 rollout 的逐帧回放（当前只有逐 episode 统计与 `done_reason`）。
- [x] 推理计时包含同步说明（rollout 报告记录 mean/p95 延迟与 NFE=1）。
- [x] 完整通道 A10 试验记录 3 seeds × 20 episodes、训练/验证 loss、checkpoint/data/config SHA256、
      峰值显存与策略延迟；证据目录含逐回合 JSON。

### 防止走偏

- 不把不同点数、不同示范或不同评测 episode 混成框架对比。
- 不用最高单次成功率代表总体性能。
- 不把本机 smoke 配置与论文完整配置直接比较。

## 7. Phase 5：开源与答辩材料

### 仓库应包含

- 安装、数据生成、训练、评测四条独立命令。
- Jittor 版本与上游版本的对应关系。
- 自动测试、配置、实验日志和结果表。
- 第三方许可证和引用。
- 已知差异、失败任务、预算和复现边界。

### 答辩证据

- 一张完整流程图。
- 一张 CFM 直观图：噪声、动作、直线路径和一致性约束。
- 一张“PyTorch 上游—Jittor 迁移模块”对照表。
- 一张定量结果表：成功率、延迟、NFE、显存。
- 一段成功视频和一段有代表性的失败视频。
- 关键模块现场可运行的短命令，而不是临时跑完整训练。

## 8. 最终验收

- [ ] 新机器按 README 能建立环境。
- [x] Jittor 主环境测试 `213 passed, 1 skipped`；跳过的 PyTorch 模块由独立 parity-env 的 170 项检查覆盖，但不计作主测试 passed。
- [x] 一条命令生成示范，一条命令训练，一条命令评测。
- [x] 模型训练和推理确认使用 Jittor。
- [x] 两个任务的低预算结果可追溯到配置、seed、checkpoint 和日志。
- [x] `button-press-wall-v3` 的完整通道 A10 三 seed 有限预算结果可追溯到配置、哈希和逐回合 JSON。
- [x] README 不夸大结果，论文/代码差异均被记录。
- [ ] GitHub 仓库无数据集、checkpoint、密钥或本机绝对路径。

