# FlowPolicy-Jittor 答辩指南

## 1. 建议的讲述主线

不要按“我先装了什么，再写了什么”流水账汇报。建议按问题展开：

1. **问题**：扩散/流策略通常需要多次网络求值，机器人在线控制受推理延迟影响。
2. **方法**：FlowPolicy 用点云和机器人状态作为条件，通过一致性流匹配学习一次网络求值即可完成的动作生成。
3. **复现难点**：官方实现是 PyTorch，公开代码与论文在点数、EMA target 等细节上有差异，旧版 MuJoCo 环境也有兼容成本。
4. **我的工作**：用 Jittor 重写训练核心；建立数据、数值和仿真三级验证；明确修正和保留了哪些上游行为。
5. **结果**：同时比较成功率和速度，并展示失败案例，而不是只展示最好视频。
6. **结论**：说明 Jittor 复现是否支持论文的核心主张，以及仍存在哪些限制。

## 2. 必须真正理解的五个问题

### Q1：为什么输入使用 3D 点云，而不是 RGB？

点云直接表示几何位置，对视角和背景纹理的依赖更小；在机械操作里，物体与末端执行器的空间关系比纹理更关键。代价是深度渲染、坐标变换和点采样会带来新的工程误差。

### Q2：FlowPolicy 为什么能做到 NFE=1？

普通 flow matching 学习随时间变化的速度场，推理时要数值积分。Consistency Flow Matching 进一步约束不同时间点的预测沿同一轨迹保持一致，使模型可直接从初始噪声跨到动作端点。NFE=1 是核心设计目标，不是简单把积分步数强行改成 1。

### Q3：为什么还要约束 velocity，而不仅是最终动作？

只约束端点可能得到相同结果，却形成不稳定的中间速度场。velocity consistency 让轨迹方向也保持一致，减少一步推理时的偏差。

当前实现先复现公开代码的实际 loss：

```text
x_t = t a + (1-t) ε
r = min(t+δ, 1)
s(t) = 当前时间段的右端点
f_t = x_t + (s-t) v_t
L = mean(||f_t-f_r||²) + α mean(mask · ||v_t-v_r||²)
```

公开 `FlowPolicy.compute_loss()` 用同一个当前模型计算 `v_t` 和 `v_r`，两个分支都保留梯度；论文公式则把后一个预测写成 EMA 参数。项目把前者标为 `code-faithful` 基线，EMA target 只能在顶层真正维护、保存并加载 EMA 模型后再称为 `paper-faithful`，不能只加一次 `stop_gradient` 就混用名称。

### Q4：没有真实机械臂，复现是否成立？

原论文的 37 个任务本身就是 Adroit/MetaWorld 仿真实验。我们可以完整复现示范生成、训练、在线 rollout 和成功率评测。需要如实说明：这验证的是仿真中的算法与实现，不等价于证明现实部署性能。

### Q5：为什么首任务选得简单？

首任务用于验证 Jittor 迁移的正确性。若直接从高维 Adroit 或困难抓取开始，失败时无法区分是模型迁移、仿真环境、示范质量还是任务难度造成。简单任务通过后再增加难度，属于控制变量。

## 3. 老师可能追问的实现问题

### “哪些代码是 Jittor，哪些不是？”

- 当前已完成的 Jittor 部分：PointNet、状态编码器、Conditional U-Net、归一化器、CFM loss、顶层 policy、Zarr 数据集、训练器、AdamW、EMA 状态、policy/trainer checkpoint 和单步动作推理。
- checkpoint 不只校验权重 shape，还固定本实现版本、Jittor 版本、网络、loss、NFE 和动作切片配置；保存前会从真实运行对象重建配置，加载失败会回滚，防止同一权重在 `code`/`exact` 模式下静默执行 3/4 步。
- 已完成：真实示范训练、checkpoint 恢复、5-episode 管线 smoke、2 任务 × 3 seeds × 20 episodes 的 CPU 低预算闭环评测，以及两个任务的完整通道 A10 三 seed 有限预算训练与评测。
- 尚未完成：论文 3000 epochs 长训练、论文式 detached EMA target，以及成功/失败视频整理。不能把 30 epochs、现代替代仿真栈的结果表述成完整论文配方复现。
- 非 Jittor：MuJoCo 物理引擎、MetaWorld 环境、官方专家策略、Zarr 文件读写。
- 点云 FPS 使用 NumPy 实现；训练与仿真链路不依赖 PyTorch。PyTorch 只在隔离的 parity-env 中作为数值参照，不能把这项对照混入 Jittor 训练图。

### “怎样证明迁移是正确的？”

按证据强弱回答：

1. 子模块固定输入/权重的数值对齐；
2. CFM loss 的公式和梯度测试；
3. 小样本过拟合；
4. 同一示范、相同配置下的闭环成功率；
5. 多随机种子的均值与方差。

只说“loss 能下降”是不够的。

### “为什么结果与论文不同？”

先检查并主动说明：

- 硬件、框架和随机数实现不同；
- 官方代码与论文存在 1024/512 点和 EMA target 差异；
- 官方数据集默认对 action、agent state 和 point cloud 三个字段都做 limits 归一化；identity 点云只是可选对照。
- 公开仓库没有作者 checkpoint 和原始示范数据；
- 训练 epoch、评测口径和任务 seed 是否完全一致。

不要把所有差异简单归因于“随机性”。

### “你做的是复现还是改进？”

第一阶段是框架复现：保持模型和数据契约，迁移至 Jittor。对官方缺陷的修复属于工程修正。只有在基线稳定后，EMA target、点数等对照实验才可以称为分析或小幅改进。

公开 policy 还有一个需要主动说明的切片细节：配置写 `horizon=4`、`n_obs_steps=2`、`n_action_steps=4`，代码却从 `n_obs_steps-1=1` 开始切，因此 Python 实际只返回 3 步。主基线保留这个行为，额外的 exact 模式才返回 4 步。答辩时把它称为“公开实现与配置的边界不一致”，不要把 exact 模式冒充原始结果。

## 4. 结果展示应避免的误区

- 只放成功视频，不放失败类型。
- 只报最好 seed 或最好一次评测。
- PyTorch 与 Jittor 使用不同示范或不同点数。
- 把缩小网络的本机 smoke 结果写成完整复现结果。
- 只比较速度，不说明 warm-up、同步、batch size 和硬件。
- 把 NFE=1 与“整个系统只计算一次”混为一谈；点云编码和环境渲染仍有开销。

## 5. 每个实验背后的理由

| 实验 | 要回答的问题 | 不能替代它的证据 |
|---|---|---|
| shape/unit test | API 与张量布局是否迁移正确 | loss 下降 |
| 小样本过拟合 | 网络和优化器是否具备学习能力 | 单次 forward |
| 专家 rollout | 仿真、渲染和成功判定是否正常 | 能 import MuJoCo |
| 闭环策略 rollout | 时序窗口、归一化、动作输出是否正确 | 离线 validation loss |
| 3 seeds | 结果是否稳定 | 最好一次成功率 |
| 延迟/NFE | 单步生成的效率主张是否成立 | 参数量 |

## 6. 当前阶段可以如实汇报的内容

**可直接报告的低预算结果**

| 任务 | 评测设置 | 成功率 | 平均推理延迟 |
|---|---:|---:|---:|
| `button-press-wall-v3` | 3 seeds × 20 episodes | 83.33% ± 20.82% | 2.97 ms |
| `reach-v3` | 3 seeds × 20 episodes | 11.67% ± 2.89% | 2.93 ms |

表格标题和口头表述都应带上“CPU、小通道、10 epochs、替代仿真栈、NFE=1”。这两项数据用于证明
Jittor 闭环可运行和任务难度差异，不能与论文成功率直接比较。

**A10 论文结构有限预算结果**

| 任务 | 评测设置 | 成功率 | 平均策略延迟 | 峰值显存 |
|---|---:|---:|---:|---:|
| `button-press-wall-v3` | 完整通道，30 epochs，3 seeds × 20 episodes | 100% ± 0% | 3.811 ± 0.016 ms | 10726 MiB |
| `reach-wall-v3` | 完整通道，30 epochs，3 seeds × 20 episodes | 31.67% ± 16.07% | 3.775 ms | 10605 MiB（采样） |

口头说明要紧跟一句：这是替代仿真栈上的匹配预算对照，证明完整 Jittor 网络可以训练和闭环运行，
不等于论文 37 个任务、3000 epochs 或真实机械臂结果。容易任务的满分说明天花板效应；困难任务
三个 seed 为 20%/25%/50%，说明少量示范和短训练下方差明显，不能只展示最好 seed。

**已经完成（可当场打开文件核对）**

- 论文与官方代码审计、任务选择与环境边界（D001–D018）；
- Jittor CPU 环境与逐模块测试：1D 算子、时间嵌入、Conditional U-Net（FiLM）、
  PointNet、状态编码器、limits 归一化、公开代码口径 CFM loss、顶层 policy、NFE=1 推理；
- 真实示范：`data/reach-v3.zarr`（10×120）、`data/button-press-wall-v3.zarr` 和
  `data/reach-wall-v3.zarr`（各 20×120）；
- 训练与闭环：`scripts/train_smoke.py` + `scripts/rollout.py`（跨环境进程桥），
  带配置指纹的 checkpoint、trainer checkpoint（AdamW 矩 + EMA）；
- EMA：`EmaModel` 逐步更新、存进 trainer checkpoint、恢复时双向校验（D018）；
- 跨框架 parity：独立 PyTorch 镜像对 action_pred / loss / 168 个梯度做对照，
  170 项检查、最差偏差 1.6e-09（`outputs/parity/parity.json`）；
- 全通道（512/1024/2048）CPU 单步前反向：2.55 亿参数、约 4 s/step、峰值 4.5 GB；
- 云端 A10 CUDA 证据：真实 `reach-v3` 数据完成 34 次参数更新并保存 checkpoint；
  完整通道模型在 1024 点下完成资源探测，随后两个任务都以 batch 64、30 epochs、3 seeds
  各训练 1050 步并分别完成 60 个闭环 episode；`reach-wall-v3` 的公开证据在
  `evidence/fpj033/formal/`，包含三组训练曲线与全部 episode；
- 无机械臂云端仿真：EGL 生成 RGB、1024 点点云和 9 维状态，脚本专家可成功，
  checkpoint 与 Meta-World 的进程桥完成 60 步闭环；
- 低预算实验网格：两个任务 × 3 seeds × 20 episodes，一条命令复跑（`scripts/experiments.py`）。

**必须如实说明的边界**

- **本机 CUDA 仍未配置，但云端 A10 已验证**：本机 RTX 5060 缺少适配 Blackwell 的 toolkit；
  云端 A10 使用 CUDA 12.1。CPU 实验表仍然是 CPU 结果，不能因为云端可用就改写其来源；
- **完整通道 GPU 训练已完成两个任务的有限预算试验，但不是论文完整配方**：每个任务只有 30 epochs，
  未跑论文 3000 epochs，也未启用 detached EMA target；
- **EMA 没有作为损失目标**：loss 里 `v_r` 仍由当前模型给出（`target_mode="code"`，与 released 代码一致）；
  论文式 detached 目标未实现，原因写在 D018；
- parity 只覆盖小配置模型，不是论文规模；
- 仿真栈是替代实现（D016），成功率不能与论文数字直接比较；
- 示范数量少（10/20 条）；CPU 小通道结果与 A10 完整通道结果预算不同，不能直接归因于框架或只比较百分比。

答辩时把“已验证事实 / 计划验证内容 / 推测”分开，并主动给出上面这份边界清单，
比提前许诺论文级结果更可信；被问“为什么不宣称复现了论文指标”，照着这份清单回答即可。


