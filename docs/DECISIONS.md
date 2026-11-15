# 技术决策记录

## D001：先做 MetaWorld，不从 Adroit 开始

**决定**：首任务使用 `button-press-wall`。

**原因**：它有官方 heuristic expert，不依赖 VRL3 checkpoint；动作仅 4 维，论文把它作为简单任务讨论。先在低复杂度任务上排除框架迁移错误，再增加任务难度。

**答辩注意**：这不是为了回避难任务，而是控制变量。首阶段验证 Jittor 迁移正确性，后续任务才验证泛化和难度。

## D002：仿真环境不改写成 Jittor

**决定**：MuJoCo/MetaWorld 保持原环境，Jittor 负责模型、损失、优化和推理。

**原因**：MuJoCo 是物理仿真器，不是深度学习框架。课程要求的关键是训练实现使用 Jittor；重写仿真器既不增加算法贡献，也会制造额外误差。

**边界**：仿真器输出 NumPy，离线数据存 Zarr，动作从 Jittor 转回 NumPy。

## D003：使用两个隔离环境

**决定**：仿真/示范和 Jittor 训练分别使用独立环境。

**原因**：官方仿真栈依赖 Python 3.8、Gym 0.21、mujoco-py 2.1 等旧依赖；将它们与 Jittor 强行装入同一环境会增加版本冲突。Zarr 是稳定、可检查的边界。

## D004：主要实验统一使用 1024 个 MetaWorld 点

**决定**：示范生成、训练和 rollout 均使用 1024 点。

**原因**：论文写 MetaWorld 使用 1024 点，官方示范生成也默认 1024；公开 runner 的 512 点与论文不一致。主结果采用一致输入，512 点只作为可选消融。

## D005：区分 code-faithful 与 paper-faithful

**决定**：先迁移公开代码的实际计算，得到可核对的 `code-faithful` 基线；再把论文描述的 EMA/stop-gradient target 作为独立实验。

**原因**：论文公式与公开 `compute_loss()` 在 target 网络和梯度处理上存在差异。如果直接混合修改，就无法判断结果来自 Jittor 迁移还是算法口径变化。

## D006：本机用于开发，云端用于正式长训练

**决定**：RTX 5060 Laptop 8GB 负责 CPU 单元测试、过拟合和短训练；完整网络、多种子与长训练迁移到与目标 GPU 架构兼容的云 CUDA 环境，并先用小 batch 探测显存，再确定实例规格和训练预算。A10 已在 CUDA 12.1 上通过；“CUDA ≥12.8”只适用于当前本机 Blackwell GPU，不是云端统一要求。

**原因**：官方 U-Net 默认通道达到 2048；完整模型约 2.55 亿参数，本机只验证过 batch 2 的 CPU 单步，不能据此断言某个固定显存规格足以承载论文 batch 128。本机先证明实现正确性，云端再逐级扩大规模，可减少空耗费用。

## D007：动作切片保留 code-faithful 基线并单列修正

**决定**：默认使用 `action_slice_mode="code"`，从 `n_obs_steps-1` 开始切；另提供显式 `exact` 模式保证返回 `n_action_steps` 步。

**原因**：公开配置为 `4/2/4`，公开代码从索引 1 开始切到 5，长度为 4 的张量实际只能返回索引 1–3，即 3 步。静默改成 4 步会改变基线；完全不指出又会误导听众。两种模式分开记录，才能判断差异来自迁移还是切片修正。

## D008：本地实现 Mish，不直接调用 Jittor 1.3.11 的 `nn.Mish`

**决定**：使用 `x * tanh(nn.softplus(x))` 实现等价的 Mish。

**原因**：实际 CPU 测试发现 Jittor 1.3.11 的 `nn.Mish` 调用了不存在的
`jt.softplus`，会在运行时报 `AttributeError`；可用的函数位于
`jittor.nn.softplus`。本地实现已用公式数值测试验证。

**答辩注意**：这是框架 API 兼容修正，不是对 FlowPolicy 算法的改进。

## D009：checkpoint 同时锁定权重与行为配置

**决定**：checkpoint 强制保存并核对 schema、本实现版本、上游 commit、Jittor 版本、网络结构、CFM 超参数、NFE 和动作切片模式；保存时从当前运行对象重建配置，配置不一致时拒绝加载。状态值还需通过 dtype 与有限值检查，文件以同目录临时文件原子替换。

**原因**：相同 shape 的权重并不代表相同策略。例如同一个 checkpoint 在 `action_slice_mode="code"` 下返回 3 步，在 `exact` 下返回 4 步；只校验张量键和 shape 会让评测行为静默改变。嵌套 U-Net 的三个 condition 开关（`use_down_condition`/`use_mid_condition`/`use_up_condition`）由 `forward` 逐次读取，不改变任何张量 shape，因此也必须从**实时模块**取值写入指纹；只记录构造时的快照会让“构造后改开关”的模型以旧配置通过保存。

**答辩注意**：这属于实验可追溯性修正，不改变模型公式。正式训练 checkpoint 还需要由训练器额外保存 optimizer、global step、scheduler 和未来的 EMA state。

## D010：数据窗口忠实复刻上游 sampler，但归一化统计只用训练 episode

**决定**：Zarr 数据集按上游 `create_indices` 的规则构造窗口（`horizon=4`、`pad_before=n_obs_steps-1=1`、`pad_after=n_action_steps-1=3`），边界处**复制该 episode 首帧/末帧**补齐，而不是丢窗或补零；训练/验证按 episode 划分；`LinearNormalizer` 只用训练 episode 的 `action`/`agent_pos`/`point_cloud` 拟合。

**原因**：窗口规则决定样本数量与动作时序边界，改变它会立刻让 loss 曲线和成功率不可比，因此保留上游行为。但上游 `MetaworldDataset.get_normalizer()` 是对**整个** replay buffer 拟合的，验证 episode 的极值会进入缩放参数，属于数据泄漏；复现项目必须能说清自己的验证集是干净的，所以这里按 episode 划分后只用训练集拟合，并在代码注释与答辩材料中把它标为“对上游缺陷的工程修正”，而不是算法改动。

**答辩注意**：若有人质疑“改动上游会让结果不可比”，回答是：归一化统计的泄漏不影响模型结构，且修正后训练集统计与全量统计在 10 条示范规模下差异很小；真要严格对照，可以另跑一次“全量拟合”作为消融，并同时报告两者。

## D011：机器人状态字段同时接受 `state` 与 `agent_pos`

**决定**：数据集优先读 `agent_pos`，找不到时回落到上游实际使用的 `state`，对外统一输出为 `obs.agent_pos`。

**原因**：上游 `metaworld_dataset.py` 读取的 Zarr 字段名是 `state`，而任务 YAML/论文口径称其为 `agent_pos`。只认一个名字会让真实示范文件无法加载，或在加载时静默错位。回落顺序写死在 `STATE_KEYS` 中，并在 `dataset.state_key` 暴露实际使用的字段，便于核对。

## D012：过拟合冒烟实验的验收指标是“固定噪声下的动作误差”，不是 loss 降幅

**决定**：32 窗口过拟合的主指标是**固定噪声、NFE=1 预测动作与示范动作的平均绝对误差**；CFM loss 只作为辅助记录。验收要求同时满足：误差相对未训练策略至少下降 50%，且低于两个常数预测器（全零、平均动作）的误差。

**原因**：CFM 的 endpoint loss 在初始化时就只有约 `1e-4`，因为 `t` 与 `s` 的间距只有 `delta=0.01`，`(s-t)` 权重把 loss 压得很小；此时“loss 下降 50%”既容易靠噪声达标的弱指标，也不能说明策略学会了动作。动作误差直接度量“模型是否记住了这 32 个窗口”，并且必须打败零预测/均值预测，才能排除“只是学了个尺度”的可能。实测三者分别为 1.418 / 2.12 / 2.11，指标确实能区分。

**答辩注意**：把这条讲成“我们主动选择了更难被满足的学习证据”，而不是“换了个更好看的指标”。

## D013：CPU 冒烟实验显式限制 OpenMP 线程数

**决定**：`scripts/overfit_32_windows.py` 在 import jittor 之前 `setdefault("OMP_NUM_THREADS", "4")`。

**原因**：本文 WSL 环境有 24 核，Jittor 默认按核数起线程，实测同一配置从 0.15 s/step 退化到 3 s/step（20 倍），且 `user` 时间远大于 `wall` 时间，是典型的线程争用。这不是算法差异，但会让同一个实验在别人的机器上超时，因此写进脚本而不是文档里。

**答辩注意**：如果被问“为什么限制线程”，回答是环境稳定性与可复现性，不是精度需要；GPU 路径不受影响。

## D014：训练器用 `(seed, step)` 派生噪声，checkpoint 区分 policy 与 trainer

**决定**：

1. `Trainer` 每步的 CFM 噪声与时间由 `np.random.default_rng([seed, step])` 生成，而不是依赖 Jittor 全局 RNG。
2. 训练器 checkpoint 内嵌**完整的 policy checkpoint payload**（经由 policy 自己的严格写入器产生），另加 optimizer 的 `m`/`values`/`n_step`、`global_step`、调度配置、`seed`、最近 50 步 history，以及显式的 `"ema": null`。
3. 两者的 `kind` 字段不同，trainer 不能当 policy 加载，policy 也不能当 trainer 加载；加载携带 EMA 的 checkpoint 会直接失败而不是静默丢弃。

**原因**：Jittor 的全局随机状态无法可靠地写进 checkpoint，若靠它采样，断点续训的“随机性”就会与不中断运行不同，续训一致性永远无法验证；把噪声绑定到 `(seed, step)` 后，续训与否不影响任何一步的输入。内嵌而非重写 policy payload，是为了复用已有的版本、配置、有限值和原子写入校验，避免两套校验逻辑漂移。`ema: null` 是防误读标记：未来的实现若加入 EMA，读到 `null` 就知道该 checkpoint 没有 EMA 权重，而不是以为“EMA 与主权重相同”。

**实测限制**：checkpoint 的状态恢复是逐位精确的（权重、AdamW 矩、`n_step`、`global_step` 全部 `array_equal`）；同一进程内两次六步运行通常也逐位相同；但“保存→恢复→再训三步”与“不中断六步”实测出现最大 `8.3e-4` 的权重差和 `6.6e-5`（相对）的第 5 步 loss 差。原因是本任务梯度极小（loss ~1e-4），AdamW 除以 `sqrt(v)` 会把 float32 舍入放大。**因此本项目不声称 bit-exact 续训**，只声称“状态恢复精确 + 调度延续精确 + 轨迹在浮点容差内一致”，并把它作为 FPJ-031 继续排查。

**FPJ-031 复测（本条目更新，`scripts/determinism_probe.py`）**：把 6 步拆成“单次 6 步”与“3 步 +
保存 + 载入 + 3 步”，逐 step 比对全部权重：

| 场景 | 结果 |
|---|---|
| 同一进程，两次独立 6 步运行 | **逐位相同**（max delta = 0） |
| 独立进程，单次 6 步 vs 3+恢复（`OMP_NUM_THREADS=1/4` × `MKL_CBWR` 有无，共 4 组） | **逐位相同**，final hash 一致 |
| 已有训练历史的长进程内做“3+恢复+3” | 漂移 `5.49e-3`（`obs_encoder` 权重）、loss 相对差 `9.5e-9`；`jt.clean_graph()`、复用同一 policy 对象都无法消除 |
| `OMP_NUM_THREADS=1` vs `4` | 轨迹不同（各自内部自洽），hash 不同 |

结论：**“恢复→续训”本身不是不确定的**，不确定的是“在一个已经跑过训练的长进程里载入 checkpoint 后继续训练”；
线程数会改变数值轨迹。因此本项目把承诺收窄为：状态恢复逐位精确、调度延续逐位精确、
同一进程内可重复、长进程内恢复续训在 `1e-2` 绝对容差内一致（`tests/test_trainer.py` 的容差已按实测设为 `1e-2`），
并保留 `scripts/determinism_probe.py` 作为可复跑的测量工具，而不是把“完全一致”写进文档。

**答辩注意**：被问“续训是否完全一致”时，回答分三层：状态恢复逐位一致、学习率调度逐位一致、长进程恢复后的参数轨迹在 1e-2 绝对容差内一致；不要含糊地说“完全一致”。

## D015：同时支持两种 Zarr 磁盘布局，并用成员测试绕开 zarr 的 `meta` 限制

**决定**：`MetaworldZarrDataset` 既接受“所有数组在根”的手写存储，也接受官方采集脚本写出的
`data/<数组>` + `meta/episode_ends` 布局；数组容器的选择与 `episode_ends` 的查找全部用**成员测试**
（`key in group`），不再调用 `group.array_keys()` 枚举。

**原因**：`third_party/Metaworld/gen_demonstration_expert.py` 写出的是
`zarr_root/create_group('data')` + `create_group('meta')`，其中 `episode_ends` 在 `meta` 下。
而 zarr 2.x 的 `Group._array_iter` 在遍历目录时会断言 `not path.startswith("meta")`，
一旦根组里存在名为 `meta` 的子组，`root.array_keys()` 直接抛 `AssertionError`。
上游自己的加载器只做 `zarr_root['meta']['episode_ends']` 取值，所以没暴露这个问题；
我们若用枚举就会在真实数据上炸。改用成员测试后，两种布局都能读，且不依赖枚举顺序。

**影响**：`_open_arrays()` 只按需取 `action`、`point_cloud`、状态键与 `episode_ends`，
不再把整组数组塞进字典；`img`/`depth`/`full_state` 等额外数组被安全忽略。
新增 3 个测试：官方布局与扁平布局逐窗口一致、状态放在 `data/` 下可读、缺 `episode_ends` 时报错信息列出三个查找位置。

**答辩注意**：如果被问“为什么不用 `array_keys()`”，回答是 zarr 2.x 对 `meta` 前缀的断言，
这是用真实布局跑通后才暴露的兼容性问题，不是风格选择。

## D016：仿真环境用现代 MetaWorld/muJoCo 替代旧栈，裁剪框在世界坐标下重新推导

**决定**：

1. 采集环境放在独立 venv `/home/s/.local/share/flowpolicy/sim-env`（Python 3.10.14，
   `mujoco 3.3.7` + `metaworld 3.0.0` + `gymnasium 1.4.0` + `zarr 2.12.0`），
   **不往 Jittor 环境里装任何仿真依赖**；渲染用 `MUJOCO_GL=egl`（Mesa llvmpipe 软件渲染，无需显示器）。
2. 保留论文的观测契约：相机 `corner2`、`cam_pos=[0.6,0.295,0.8]`、`znear=0.1`/`zfar=1.5`、
   128×128 深度、固定旋转 `Rx(61.4°) @ Ry(-7°)`、FPS 采样到 1024 点、
   9 维 `agent_pos`（eef + 左右指）、脚本专家策略、只保留成功步数 ≥5 的 episode、官方 Zarr 布局。
3. 裁剪框改为在**世界坐标**下定义：`(-0.35, 0.25, -0.05) ~ (0.35, 0.95, 0.55)`，
   并且先在世界上裁剪、再施加固定旋转；官方那组界限是在 metaworld 2.0.0 + mujoco-py 的
   场景坐标系里整定的，直接搬到现代栈上 16384 个像素里只剩约 0~100 个点，等于把场景裁空。

**原因**：官方 vendored 的 `mujoco-py 2.1.2.14` + `gym 0.21` + `Metaworld` 需要 Python≤3.8 和
一堆旧构建工具，而现代 `mujoco` 自带预编译轮子、`metaworld` 3.x 自带 50 个脚本专家策略，
在本机一次装通。渲染约定也不同：`mujoco.Renderer` 的深度**已经是米制、已经上下翻正**，
不能再套官方 `near/(1-d*(1-near/far))` 和 `verticalFlip`，否则点云会静默错位。
裁剪框重推的依据不是猜：把相机点云按 `cam_xmat/cam_xpos` 变换到世界后，
有 1173 个点落在 z=-0.913 的地面板上、5233 个点落在 z=0 的桌面上，几何对得上，
再取桌面上方的工作空间作为框，保留 28.5% 的点。

**影响与必须披露的差异**：

- 字段、形状、维度、成功筛选、磁盘布局与论文一致；**逐像素点云不可能与论文一致**
  （不同 MuJoCo 版本、不同深度约定、重推的裁剪框）。
- 我们只落盘训练真正用到的 `action`、`state`、`point_cloud` 和 `meta/episode_ends`；
  官方采集脚本还会写 `img`、`depth`、`full_state`。这些字段 FlowPolicy 训练不读
  （shape_meta 只有点云、状态、动作），但属于省略项，必须披露。
- `action` 记录的是脚本专家的**原始输出**（与官方采集脚本一致）。MetaWorld 在 `step()` 内部
  把动作裁剪到 [-1,1]，因此落盘值可能略超出该范围：本次 `reach-v3` 10 条 episode 中只有第 1 维
  有 1.96% 的条目超过 1，最大 1.48。训练时学的是原始输出，推理时同样会经过环境裁剪，闭环语义一致。
- 因此所有训练/评测结果只能表述为“在替代仿真栈上复现”，不能表述为“与论文同一数据”。
- 现代 API 细节：`step()` 的 `info["success"]` 是 float（按 `>0.5` 判定），`reset()` 的 info 为空字典，
  专家类名是 `Sawyer*V3Policy`，任务名形如 `reach-v3`。

**答辩注意**：被问“为什么不用原版 mujoco-py”，回答：原栈锁定 Python 3.8 且构建脆弱，
现代栈能在本机稳定跑通 EGL 软件渲染并自带脚本专家；为了不误导，裁剪框重推、
深度约定差异和“不能宣称同分布”都写进了文档与 README。

## D017：rollout 用“进程内 JSON 行桥”跨环境，不合并依赖

**决定**：闭环评测拆成两个进程：

- `scripts/sim_server.py` 在 sim-env 里跑仿真，逐行读 JSON 请求、回 JSON 响应，
  数组以 base64 编码的 `.npy` 载荷传输（不丢精度、不需要安装额外编解码库）；
- `scripts/rollout.py` 在 Jittor 环境里加载 checkpoint、算动作，通过
  `subprocess.PIPE` 与前者对话。

协议：`hello` / `reset {"index": n}` / `step {"action": [...]}` / `close`；
`step` 的响应区分 `done`（环境自然终止）与 `step_cap`（达到步数上限），
所以“跑满 N 步”不会被误记为“环境结束”。

**原因**：课程要求训练实现用 Jittor，而仿真栈不能装进 Jittor 环境（D003/D016）。
把 jittor 装进 sim-env 或把 mujoco 装进 Jittor 环境都会破坏隔离，也会让“哪个环境出了问题”
变得不可判定。进程桥的代价是每步一次进程间往返（实测可忽略：40 步端到端 6.7 秒）。

**约定**：

1. `--action-slice-mode` 必须显式给出（`code` 或 `exact`），并写进 rollout 报告；
   `code` 取 `[n_obs_steps-1, n_obs_steps-1+n_action_steps)`，`exact` 取
   `[min(n_obs_steps-1, horizon-n_action_steps), +n_action_steps)`。
2. 噪声由 `(noise_seed, episode, chunk_index)` 决定，保证同一 checkpoint 的 rollout 可重放。
3. 报告必须同时列出成功与失败 episode、每 episode 的步数与成功步数、以及 NFE 和单次推理延迟；
   5 条 smoke episode 不构成成功率。
4. rollout 开始前同时记录重复推理、同对象重载和新对象重载三种动作差异。CPU 结果可以逐位一致；
   CUDA 后端不能预设差异必须为 0，应把重载差与“不重载的重复推理差”一起判断。A10 实测重复推理
   漂移与重载漂移同为 `1e-4` 量级，因此这组数据用于排查 checkpoint 损坏，不用于声称位级确定性。

**影响**：`outputs/smoke`（100 epochs 小通道 CPU 训练）与 `outputs/rollout`（5 条 episode）都是
可复跑的证据，但它们只是管线证据，不是论文指标。


## D018：EMA 目标、CUDA 结论与跨框架 parity 的边界

**决定**：FPJ-028 把“论文级的训练目标”拆成三件可独立验证的事，并分别给出结论：

1. **EMA 权重**：`src/flowpolicy_jittor/training/ema.py` 实现 `EmaModel`，
   排除 `normalizer` 与 `var_est`（前者是输入统计量、后者在论文前向里是死支路），
   衰减按论文的 `min(decay, (1+step)/(10+step))` 起步，再在 `warmup_steps` 内线性过渡到 `decay`。
   `Trainer(ema_decay=...)` 每步更新影子权重，并把 `ema` 段写进 trainer checkpoint；
   恢复时“checkpoint 有而 trainer 没有”和“trainer 有而 checkpoint 没有”两种情况都直接报错，
   不允许静默丢权重。
2. **CUDA**：`scripts/cuda_smoke.py` 真去设 `jt.flags.use_cuda=1` 并跑一个 matmul。
   本机结论是**不可用**，报告里保存了原始报错 `Check failed: value==0  No CUDA found.`
   （WSL 有驱动和 `/dev/dxg`，但没有 CUDA toolkit：Blackwell 需要 ≥ 12.8，本机未安装）。
   云端 A10 后来在 CUDA 12.1 上通过真实 Jittor kernel、反向更新与完整通道训练；因此每个数字必须
   标明“本机 CPU”或“云端 A10”，不能用云端结果改写本机环境结论。
3. **跨框架 parity**：`scripts/parity_check.py`（Jittor 侧导出权重/输入/输出/梯度）
   加 `scripts/parity_torch_model.py`（独立写的 PyTorch 镜像，跑在单独的 `parity-env` 里，
   按同名权重严格加载）。实测：action_pred 最大绝对误差 9.5e-07、loss 7.3e-10、
   168 个梯度张量最大绝对误差 1.6e-09，170 项检查全过。

**为什么不是“实现了 EMA 就用它做损失目标”**：论文的 `target_mode="detached"` 需要在前向图里
放第二份模型（EMA 副本），会改动 checkpoint 指纹与显存占用，属于训练配方层面的改动。
当前实现把 EMA 权重存进 trainer checkpoint 并可用于评测/回放，但 `compute_loss` 仍用当前模型
算 `v_r`（`target_mode="code"`，与上游 released 配置一致）；这条差异写在 summary 的
`ema.note` 与 `prohibited_claims` 里，不靠读者猜。

**影响**：
- `outputs/cuda/report.json`：CUDA 探测 + 全通道（512/1024/2048）单步前反向证据：
  2.55 亿参数、CPU 约 3.7–4.3 s/step、峰值内存约 4.5 GB、动作形状 `[2,3,4]`。
- `outputs/parity/parity.json`：跨框架 parity 的逐项数值与容差判定。
- D018 当时全通道只有一步资源探测；后续 D019 已补上完整通道三 seed 有限预算训练。它仍不是
  论文 3000 epochs、37 任务配方，不能回写成“论文成功率已复现”。

## D019：A10 多 seed 试验采用 batch 64、30 epochs 和 20 回合评测

**决定**：完整 `512/1024/2048` 通道模型在 A10 上先做 batch 探测，再用 batch 64、30 epochs、
3 个独立训练 seed（0、1、2）和每 seed 20 个闭环 episode 完成论文结构试验。训练与 rollout 串行，
不让训练负载污染约 4 ms 的策略推理延迟。

**原因**：batch 1–64 均能运行，batch 128 在生成报告前异常退出；batch 64 在吞吐与显存余量之间更稳妥。
3 个训练 seed 与论文的随机种子数量一致，也是课程复现中计算均值和样本标准差的最低合理配置。
只有三组结果方差明显或出现失败时才扩到 5 个 seed；若全部达到天花板，优先增加更难任务或失败案例，
比重复跑第 4、5 个容易任务 seed 更有信息量。

**边界**：这仍是“论文网络结构的多 seed 试验”，不是完整论文配方。训练只有 30 epochs，数据来自
现代替代仿真栈，当前只覆盖 `button-press-wall-v3`，且未启用论文式 detached EMA target。

