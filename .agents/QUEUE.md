# 串行任务队列

只有 gate 角色可以把任务标为完成并激活下一项。技术细节仍以 `docs/REPRODUCTION_PLAN.md` 为准。

## 已完成

### [x] FPJ-021：锁定 U-Net 嵌套行为配置

Gate `FPJ021-GATE-01` 通过。`_runtime_checkpoint_config()` 现在从实时 `self.model` 读取三个 condition flag，
任一 flag 构造后被修改都会阻止保存与加载；基线由 `128 passed, 1 skipped` 更新为 `131 passed, 1 skipped`。

### [x] FPJ-022：Jittor Zarr Dataset

Gate `FPJ022-GATE-01` 通过。`src/flowpolicy_jittor/data/dataset.py` 复刻上游 `create_indices` 窗口语义，
边界复制首末帧补齐，按 episode 划分训练/验证，normalizer 只用训练 episode 拟合；
新增 18 个测试，基线更新为 `149 passed, 1 skipped`。

### [x] FPJ-024：最小训练器与训练 checkpoint

Gate `FPJ024-GATE-01` 通过。新增 `src/flowpolicy_jittor/training/trainer.py`：AdamW 只作用于
`trainable_parameters()`，warmup+cosine 调度为 step 的纯函数，噪声由 `(seed, step)` 派生，
trainer checkpoint 内嵌 policy payload 并保存 AdamW 矩、`n_step`、`global_step` 与 `ema: null`；
新增 20 个测试，基线更新为 `174 passed, 1 skipped`。
已知限制（D014）：CPU 后端续训轨迹最大漂移 `8.3e-4`，不声称 bit-exact。

### [x] FPJ-025：MetaWorld / MuJoCo / EGL 示范环境

Gate `FPJ025-GATE-01` 通过。现代栈一次装通（`mujoco 3.3.7` + `metaworld 3.0.0` + `gymnasium 1.4.0`，
独立 venv，`MUJOCO_GL=egl` 软件渲染），真实 `reach-v3` 示范已录制并通过 `--check-store` 契约校验；
数据集新增官方 `data/` + `meta/` 布局支持（D015），仿真栈替代与重推裁剪框写入 D016。
基线 `174 passed, 1 skipped` → `177 passed, 1 skipped`。

### [x] FPJ-026：专家示范（1 条临时 + 10 条正式）

Gate `FPJ026-GATE-01` 通过。`data/reach-v3.zarr` 含 10 条 × 120 步真实示范，成功步数
`[70, 77, 87, 81, 67, 78, 85, 73, 63, 76]`，`metaworld` 全局 RNG 播种后同 seed 重录逐位一致；
修复前的未播种数据保留在 `data/archive/`。基线保持 `177 passed, 1 skipped`。

### [x] FPJ-027：训练与 5-episode rollout smoke
Gate `FPJ027-GATE-01` 通过。真实数据 100 epochs（3400 步）训练完成，loss `1.365e-4 → 2.229e-7`，
验证 loss `2.20e-5`；闭环 rollout 5 条 episode（成功步数 111/0/172/163/0）经跨环境进程桥完成，
NFE=1、单次推理 2.886 ms、checkpoint 重载动作差 0.0。过程中发现并修复了 `Trainer` 时间采样越界缺陷。
基线 `177 → 185 passed, 1 skipped`。

### [x] FPJ-028：CUDA、完整通道、parity 与 EMA

Gate `FPJ028-GATE-01` 通过。CUDA 真探测后记录为**不可用**（`Check failed: value==0  No CUDA found.`，
驱动有、toolkit 无），没有把 CPU 结果写成 GPU 结论；`512/1024/2048` 全通道在 CPU 上跑通一步真实前反向
（2.55 亿参数、约 3.7–4.3 s/step、峰值 4.5 GB、动作 `[2,3,4]`）；独立的 PyTorch 镜像（torch 2.14.1+cpu）
对 action_pred、loss 与 168 个梯度做跨运行时对照，170 项检查全过，最差偏差 `1.644e-09`；
`EmaModel` + trainer checkpoint 落地（衰减 `0.1 → 0.999` 单调、重载后 120 个张量与在线权重不同）。
反例角色抓到两个真缺陷：`var_est` 前缀过滤失效、parity 容差过宽会漏判 24% 的梯度偏差，均已修复并加回归测试。
基线 `185 → 203 passed, 1 skipped`。

### [x] FPJ-029：低预算定量实验网格

Gate `FPJ029-GATE-01` 通过。新增 `scripts/experiments.py`（一条命令跑完 (task, seed) 网格，
训练并发、rollout 串行以免 CPU 争用污染延迟），并录制第二个任务示范
`data/button-press-wall-v3.zarr`（20 条 × 120 步，成功步数 51–62）。
实跑 2 任务 × 3 seeds × 20 episodes：
`button-press-wall-v3` **83.33% ± 20.82%**（12/20、18/20、20/20）、
`reach-v3` **11.67% ± 2.89%**（2/20、3/20、2/20），latency 2.93–2.97 ms、NFE=1、
全部 episode 以 `step_cap` 结束、checkpoint 重载动作差 0。
独立复核脚本重算 6 个 cell 的逐 episode 结果与均值/标准差，`FAILURES: none`。
基线 `203 → 208 passed, 1 skipped`。
注意：这是低预算 CPU 复现（小通道、10 epochs、替代仿真栈），不是论文指标。

### [x] FPJ-030：消除 `test_policy` 的未复现间发失败

Gate `FPJ030-GATE-01` 通过。`test_top_level_adamw_updates_active_but_not_frozen_parameters`
显式 `jt.set_global_seed(0)`，断言由“任一参数变化”改为“最大变化量 > 1e-6”并带上
loss/参数数诊断信息。修复后单测连续 20 次全过、整文件 `tests/test_policy.py` 3 次全过，
仍未复现原始失败，故保留“未定根因”的说明而不是声称已定位。

### [x] FPJ-031：CPU 后端浮点不确定性

Gate `FPJ031-GATE-01` 通过。新增 `scripts/determinism_probe.py`（逐步权重快照 + 跨运行比对）
与 `tests/test_determinism.py`（5 个测试）。实测结论：同一进程两次 6 步运行**逐位相同**；
独立进程内“单次 6 步 vs 3+恢复+3”在 `OMP_NUM_THREADS=1/4` × `MKL_CBWR` 有无共 4 组下**全部逐位相同**；
只有在“已跑过训练的长进程里载入 checkpoint 再续训”时出现 `5.49e-3` 权重漂移、`9.5e-9` loss 相对差，
`jt.clean_graph()` 与复用 policy 对象都无效；线程数变化会改变轨迹（各自自洽）。
因此把承诺收窄为“状态恢复与调度逐位精确、长进程内续训在 `1e-2` 容差内”，
`tests/test_trainer.py` 的容差按实测从 `2e-3` 改为 `1e-2` 并写明依据。基线 `208 → 213 passed, 1 skipped`。

### [x] FPJ-032：云端 GPU 可行性、完整通道有限预算训练与首次发布

Gate `FPJ032-GATE-01` 通过。A10/CUDA 12.1 上完成真实 Jittor CUDA、真实 Zarr 更新、
完整 `512/1024/2048` 通道单步与 EGL 闭环验证；`button-press-wall-v3` 以 batch 64、
30 epochs、3 seeds 各 1050 updates，并完成 3 × 20 rollout，结果为 20/20、20/20、20/20，
策略延迟 `3.811 ± 0.016 ms`、峰值显存 10726 MiB。三个 checkpoint/data/config 哈希、训练曲线和
逐 episode JSON 已保存。代码已发布到 `https://github.com/shview/FlowPolicy-Jittor`，仓库未跟踪
数据集、输出、checkpoint、密钥或大文件。该结果是单任务完整通道有限预算证据，不是论文
3000 epochs / 37 任务配方；基线保持 `213 passed, 1 skipped`。

### [x] FPJ-033：更难 MetaWorld 任务与完整通道有限预算实验

Gate `FPJ033-GATE-01` 通过。`reach-wall-v3` 的 10 个专家 variant 全部成功后，正式数据、
本地 smoke、周期 checkpoint / CLI resume 与 fresh-clone 检查均通过；A10 上完成完整
`512/1024/2048` 通道、30 epochs、3 seeds × 20 episodes，结果为 4/20、5/20、10/20，
即 **31.67% ± 16.07%**，策略平均延迟 3.775 ms，外部采样峰值显存 10605 MiB。
15 个脱敏公开证据文件的 manifest/source 哈希、全部 60 条 episode 与 Git 候选内容经独立复核；
不含原始数据集/checkpoint，也不是论文 3000-epoch/37-task 结果。基线更新为
`227 passed, 1 skipped`。

## 当前任务

### [x] FPJ-034：论文式 detached EMA target

- 为 `v_r` 增加与在线 policy 分离、不可训练且显式 detached 的 EMA target；不能只在在线输出上
  加 `stop_gradient`，也不能改变现有 `target_mode="code"` 的默认行为；
- 明确定义 EMA target 的初始化与更新时序，并用小配置测试证明 target 不接收 optimizer 梯度、
  不被在线 optimizer 更新，而是只按 EMA 规则变化；
- trainer checkpoint 必须保存、校验并恢复在线权重与 target/EMA 双方状态，覆盖继续训练；旧 checkpoint
  要么保持兼容，要么以清晰错误拒绝，不能静默改变语义；
- 先完成有限值、梯度隔离、保存/恢复、CPU 小配置资源开销与全量回归测试；本任务不授权云端开机或
  昂贵正式实验，未通过 gate 前不得称为 paper-faithful target。

## 后续顺序

### [x] FPJ-035：target 语义对照

在相同任务、示范、seed、点数和训练预算下比较 released-code target 与 detached EMA target；
先报告损失、稳定性与资源开销，再决定是否值得扩到完整通道。

### [x] FPJ-036：A10 完整通道 target 配对验证

在 FPJ-035 本地三 seed 对照封闭后，使用阿里云 PAI-DSW A10 对完整 512/1024/2048 通道做
code/detached 单 seed 严格配对。先回答 CUDA 可行性、显存、耗时与闭环方向是否一致，不把单 seed
验证写成正式三 seed 结论；证据完成后立即停止实例。

### [>] FPJ-037：三 seed target 扩展、证据整理与发布（Gate 已通过，等待下一任务）

用已授权的 A10 只补跑 seeds 1/2 的 code/detached 四个单元，不重跑或改写 FPJ-036 seed 0。
云端训练期间同步整理 FPJ-032 至 FPJ-036 已有证据，增加统一索引和本地校验器，固定公开证据的
LF 行尾。新结果在四个单元完成、匹配检查和独立重算前保持留空；每个单元闭合后先哈希再清理大型
trainer checkpoint，最终生成独立的 `evidence/fpj037/` 脱敏包并停机。

### [x] FPJ-038：视频材料已取消

用户于 2026-11-16 明确不再需要制作视频材料，因此不再整理或生成视频素材。FPJ-037 已完成的
技术说明、公开证据和答辩讲解草稿保留；后续若需要答辩更新，将作为新的独立任务处理。

### [ ] FPJ-039：已并入 FPJ-037（不再单独执行）

用户于 2026-11-16 明确同意材料整理与云端训练同步进行，原 FPJ-039 的 seeds 1/2 扩展已吸收到
FPJ-037，不得在队列中再次启动造成重复训练。

## 测试可靠性

（FPJ-030 已在上面“已完成”段落中关闭。原始失败未复现，修复方式与 20 次重复运行结果记录在那里。）
