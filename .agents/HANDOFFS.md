# 串行交接记录

本文件只允许在末尾追加。禁止修改、删除或重排既有记录。

## INIT-2026-11-13

- Task: 初始化多-agent串行协作协议
- Role completed: bootstrap
- Result: passed
- Files changed:
  - `AGENTS.md`
  - `agent.md`
  - `.agents/README.md`
  - `.agents/STATE.yaml`
  - `.agents/QUEUE.md`
  - `.agents/HANDOFFS.md`
  - `.agents/HANDOFF_TEMPLATE.md`
- Commands run:
  - `git status --short`
  - `rg` 检查当前 checkpoint 指纹、测试基线和现有文档
- Exact evidence:
  - 仓库已初始化，但没有首次 commit；当前工程文件均未跟踪。
  - 最新已复核 CPU 基线为 `128 passed, 1 skipped`。
  - 唯一 skip 是未安装 PyTorch 的独立 parity 模块。
  - 当前 P1：运行时 checkpoint 指纹没有重新读取 U-Net 三个 condition flag。
- Remaining risks:
  - CUDA、完整通道、PyTorch parity、EMA、Dataset、训练和仿真均未完成。
  - policy checkpoint 不包含 optimizer、global step、scheduler 或 EMA。
- Prohibited claims:
  - 不得声称已复现论文成功率、CUDA、EMA、端到端训练或 MetaWorld rollout。
- Recommended next role: implementation
- Next agent should read:
  - `AGENTS.md`
  - `.agents/STATE.yaml`
  - `src/flowpolicy_jittor/model/policy.py`
  - `tests/test_policy.py`
  - `docs/DECISIONS.md`

## MAINT-2026-11-13-01

- Task: 加固多-agent串行协议
- Role completed: protocol_maintenance
- Result: passed
- Owner: root orchestrator
- Files changed:
  - `AGENTS.md`
  - `.agents/README.md`
  - `.agents/STATE.yaml`
  - `.agents/HANDOFFS.md`
  - `.agents/HANDOFF_TEMPLATE.md`
  - `scripts/validate_agent_state.py`
- Commands run:
  - `scripts/validate_agent_state.py`
  - `python -m compileall -q scripts/validate_agent_state.py`
- Exact outputs:
  - 初版 validator 返回 `task=FPJ-021 stage=implementation status=ready`。
  - 复审发现角色切换和 owner 互斥检查不足，已按审查意见加固。
- Acceptance status:
  - [x] 角色完成时一次性切换到下一 stage/ready/owner null。
  - [x] 明确 orchestrator 同时只启动一个当前角色 agent。
  - [x] validator 检查 owner/status、唯一当前队列项、前置 role handoff 和阶段顺序。
  - [x] 增加 stale owner recovery 与 protocol maintenance 规则。
- Findings:
  - P1: none after fixes
  - P2: 文件级协议不能提供操作系统锁；互斥由 orchestrator 串行启动和 owner 状态共同保证。
  - P3: none
- Remaining risks:
  - 私自绕过 orchestrator 的外部进程仍可能并发写文件；当前课程项目不引入额外锁服务。
- Prohibited claims:
  - 协议加固不代表 FPJ-021 产品代码已经完成。
- Recommended next role: implementation
- Next agent should read:
  - `AGENTS.md`
  - `.agents/STATE.yaml`
  - `.agents/QUEUE.md`
  - 本记录

## MAINT-2026-11-13-GATE

- Task: 多-agent串行协议最终 gate
- Role completed: protocol_maintenance_gate
- Result: passed
- Owner: root/protocol_final_gate
- Files changed:
  - `none`（只读 gate）
- Commands run:
  - `scripts/validate_agent_state.py`
  - `python -m compileall -q scripts/validate_agent_state.py`
  - validator negative checks for owner and prerequisite handoffs
- Exact outputs:
  - `Agent state OK: task=FPJ-021 stage=implementation status=ready`
  - validator 和 compileall 均为 exit 0。
  - `in_progress` 无 owner、verification 无 implementation handoff、gate 缺前置 handoff 均被正确拒绝。
- Acceptance status:
  - [x] 无角色切换死锁。
  - [x] owner/status 组合受校验。
  - [x] 阶段顺序和唯一当前任务受校验。
  - [x] stale owner 和协议维护流程有明确规则。
- Findings:
  - P1: none
  - P2: none
  - P3: 文件协议不等同于操作系统文件锁，已在规则中披露。
- Remaining risks:
  - 故意绕过 orchestrator 的外部进程仍可能并发写文件。
- Prohibited claims:
  - 协议 gate 通过不代表 FPJ-021 产品代码完成。
- Recommended next role: implementation
- Next agent should read:
  - `AGENTS.md`
  - `.agents/STATE.yaml`
  - `.agents/QUEUE.md`
  - 本记录

## FPJ021-IMPL-01

- Task: FPJ-021 Lock nested U-Net behavior into the checkpoint runtime fingerprint
- Role completed: implementation
- Result: passed
- Owner: orchestrator/implementation
- Files changed:
  - `src/flowpolicy_jittor/model/policy.py`
  - `tests/test_policy.py`
  - `docs/DECISIONS.md`
- Commands run:
  - `wsl.exe -d Ubuntu -- /home/s/.local/share/flowpolicy/miniforge-py310/bin/python scripts/validate_agent_state.py`
  - `wsl.exe -d Ubuntu -- /home/s/.local/share/flowpolicy/miniforge-py310/bin/python -m pytest -q tests/test_policy.py`
  - `wsl.exe -d Ubuntu -- /home/s/.local/share/flowpolicy/miniforge-py310/bin/python -m pytest -q`
  - `git status --short`
- Exact outputs:
  - claim 时 validator：`Agent state OK: task=FPJ-021 stage=implementation status=in_progress`
  - `pytest -q tests/test_policy.py` → `27 passed in 12.72s`，exit 0
  - `pytest -q` → `131 passed, 1 skipped in 20.28s`，exit 0；skip 仍仅为 `tests/test_conv1d_parity.py`（无 torch），不计入 passed
  - `git status --short` → 全部文件仍为 untracked，未做任何 commit
- Change summary:
  - `_runtime_checkpoint_config()` 现在从实时 `self.model` 读取 `use_down_condition`/`use_mid_condition`/`use_up_condition`，不再只沿用构造时的 `_construction_config` 快照。
  - 新增参数化测试 `test_unet_condition_flag_mutation_blocks_save`，对三个 flag 分别断言：指纹随实时模块变化、保存抛 `RuntimeError` 且错误信息含该 flag、目标文件与临时文件都不落地。
- Acceptance status:
  - [x] The runtime checkpoint config reads all three live U-Net condition flags.
  - [x] Mutating any one of the three flags after construction prevents checkpoint saving.
  - [x] The checkpoint still saves and restores when the runtime config is unchanged.（`test_checkpoint_roundtrip_rebuilds_empty_normalizer` 仍通过）
  - [x] Policy checkpoint tests pass.（27 passed）
  - [x] The full CPU test suite does not regress from the recorded baseline.（128 → 131 passed，1 skipped 不变）
  - [x] No CUDA, PyTorch parity, EMA, Dataset, or simulator result is claimed.
- Findings:
  - P1: none
  - P2: 指纹仍只比较扁平键；`shape_meta` 等嵌套字典依赖整体相等，没有字段级诊断信息，属可读性问题，不削弱阻断能力。
  - P3: `bool()` 强制转换意味着把 flag 改成等价真值（如 `1`）不会阻断，但 `forward` 同样按真值判断，行为未变，不构成绕过。
- Remaining risks:
  - 仅 CPU 小通道配置；CUDA、完整 `512/1024/2048` 通道、PyTorch parity、EMA、Zarr 数据、训练循环与仿真 rollout 仍未验证。
  - 本轮实现与验证由同一个 orchestrator 串行执行，属于自审而非独立审计。
- Prohibited claims:
  - 不得声称已复现论文成功率、CUDA 已验证、PyTorch/Jittor 精确对齐、EMA 已完成、端到端训练或 MetaWorld rollout 已完成。
  - policy checkpoint 仍不等于完整 trainer checkpoint。
- Recommended next role: verification
- Next agent should read:
  - `src/flowpolicy_jittor/model/policy.py`（`_runtime_checkpoint_config`、`_assert_runtime_configuration`、`save_checkpoint`）
  - `tests/test_policy.py`（`test_unet_condition_flag_mutation_blocks_save`）
  - `docs/DECISIONS.md`（D009）
  - `AGENTS.md`

## FPJ021-VERIFY-01

- Task: FPJ-021 Lock nested U-Net behavior into the checkpoint runtime fingerprint
- Role completed: verification
- Result: passed
- Owner: orchestrator/verification
- Files changed:
  - none（只读；独立脚本写在仓库外的 `/mnt/f/workspace/2/verify_fpj021.py`）
- Commands run:
  - `wsl.exe -d Ubuntu -- /home/s/.local/share/flowpolicy/miniforge-py310/bin/python scripts/validate_agent_state.py`
  - `wsl.exe -d Ubuntu -- /home/s/.local/share/flowpolicy/miniforge-py310/bin/python -m pytest -q tests/test_policy.py`
  - `wsl.exe -d Ubuntu -- /home/s/.local/share/flowpolicy/miniforge-py310/bin/python -m pytest -q`
  - `wsl.exe -d Ubuntu -- /home/s/.local/share/flowpolicy/miniforge-py310/bin/python /mnt/f/workspace/2/verify_fpj021.py`
- Exact outputs:
  - validator：`Agent state OK: task=FPJ-021 stage=verification status=in_progress`
  - `pytest -q tests/test_policy.py` → `27 passed in 11.06s`，exit 0
  - `pytest -q` → `131 passed, 1 skipped in 20.25s`，exit 0；skip 仍为 `tests/test_conv1d_parity.py`（`No module named 'torch'`）
  - 独立脚本输出：
    - `[1] use_down_condition: output_changed=True`
    - `[1] use_mid_condition: output_changed=True`
    - `[1] use_up_condition: output_changed=True`
    - `[2] pre_fix_saved_after_mutation=True`
    - `[3] rejected=True names_field=True`
    - `[3] file_written=False`
    - `FPJ-021 independent verification: PASS`
- Independent evidence（实现者没有提供过的检查）:
  - 证明三个 flag 都是真实行为开关：逐个置 False 后 `predict_action` 输出均改变（不是“只有指纹变化、行为不变”）。
  - 证明新测试非空转：把 `_runtime_checkpoint_config` 换回“只 deepcopy 构造快照”的旧实现后，同样篡改能**成功保存**，即修复前的漏洞可复现。
  - 证明修复有效：当前实现对同一篡改抛 `RuntimeError`，错误信息含被改字段名，且 checkpoint 文件与临时文件都未产生。
- Acceptance status:
  - [x] The runtime checkpoint config reads all three live U-Net condition flags.
  - [x] Mutating any one of the three flags after construction prevents checkpoint saving.
  - [x] The checkpoint still saves and restores when the runtime config is unchanged.
  - [x] Policy checkpoint tests pass.（27 passed）
  - [x] The full CPU test suite does not regress from the recorded baseline.（131 passed / 1 skipped）
  - [x] No CUDA, PyTorch parity, EMA, Dataset, or simulator result is claimed.
- Findings:
  - P1: none
  - P2: 独立脚本只覆盖小通道 CPU 配置；完整通道下的相同断言依赖 FPJ-028 的 CUDA 阶段。
  - P3: verification 与 implementation 由同一 orchestrator 串行执行，独立性来自“换回旧实现复现缺陷 + 行为差异实测”，不是来自独立 agent。
- Remaining risks:
  - 未验证 CUDA、完整通道、训练循环与仿真；policy checkpoint 仍不是 trainer checkpoint。
- Prohibited claims:
  - 不得把本次 CPU 验证表述为论文结果复现或 CUDA 通过。
- Recommended next role: anti_pattern
- Next agent should read:
  - `src/flowpolicy_jittor/model/policy.py:305-355`
  - `tests/test_policy.py`（新增用例）
  - 本记录

## FPJ021-AP-01

- Task: FPJ-021 Lock nested U-Net behavior into the checkpoint runtime fingerprint
- Role completed: anti_pattern
- Result: passed
- Owner: orchestrator/anti_pattern
- Files changed:
  - none（只读审计；脚本在仓库外 `/mnt/f/workspace/2/audit_fpj021.py`）
- Commands run:
  - `wsl.exe -d Ubuntu -- /home/s/.local/share/flowpolicy/miniforge-py310/bin/python /mnt/f/workspace/2/audit_fpj021.py`
  - `rg -n "use_down_condition|use_mid_condition|use_up_condition|condition_type" src/flowpolicy_jittor/model`
  - `git status --short`
- Exact outputs:
  - 审计表（fingerprint=保存是否会阻断；behavior=逐次 `predict_action` 输出是否真变）：

    ```text
    policy.nfe                       yes  yes
    policy.action_slice_mode         yes  no
    policy.horizon                   yes  error:ValueError
    policy.n_action_steps            yes  no
    policy.n_obs_steps               yes  error:ValueError
    policy.obs_as_global_cond        yes  no
    model.use_down_condition         yes  yes
    model.use_mid_condition          yes  yes
    model.use_up_condition           yes  yes
    model.condition_type             no   no
    model.n_groups                   no   no
    model.downsample_factor          no   no
    loss_fn.eps                      yes  yes
    loss_fn.delta                    yes  no
    loss_fn.alpha                    yes  no
    loss_fn.num_segments             yes  no
    loss_fn.boundary                 yes  no
    loss_fn.time_scale               yes  yes
    loss_fn.validate_finite          yes  no
    ```

  - `No fingerprint gap: every behavior-changing attribute is caught.`
  - 首轮审计把 `num_segments` 误设为默认值 2 得到 `no/no`，改为 4 后为 `yes`；该误设已在仓库外脚本中修正，不影响仓库文件。
  - `git status --short` → 仍只有原有 untracked 目录，`../flowpolicy-upstream/` 未被修改。
- Findings:
  - P1: none。没有发现“会改变推理行为但指纹不拦截”的第三类字段。
  - P2: `model.condition_type`、`model.n_groups`、`model.downsample_factor` 不在指纹内；实测改这三个值不影响 `predict_action` 输出（只在 `__init__` 中被消费），因此不构成本任务同类漏洞；若未来实现发生变化让它们在 `forward` 中被读取，必须同步加入指纹。
  - P3: `loss_fn.num_segments`/`delta`/`alpha`/`boundary`/`validate_finite` 已进指纹，但只影响训练 loss 而不影响推理输出；对 trainer checkpoint 而言它们同样必须固定，已转记到 FPJ-024 的关注点。
- Remaining risks:
  - 仓库级禁区检查成立（上游只读、未递归删除、未覆盖 checkpoint、未 commit）；但仓库尚无首次 commit，无法用 diff 证明“只改了 3 个文件”。
- Prohibited claims:
  - 不得据此声称 EMA、CUDA、完整通道或训练 loop 已验证。
- Recommended next role: quality
- Next agent should read:
  - `src/flowpolicy_jittor/model/conditional_unet1d.py:206-216, 435-460`
  - `src/flowpolicy_jittor/model/pointnet.py:64-95`
  - 本记录

## FPJ021-QUAL-01

- Task: FPJ-021 Lock nested U-Net behavior into the checkpoint runtime fingerprint
- Role completed: quality
- Result: passed
- Owner: orchestrator/quality
- Files changed:
  - none（只读；对抗脚本在仓库外 `/mnt/f/workspace/2/quality_fpj021.py`）
- Commands run:
  - `wsl.exe -d Ubuntu -- /home/s/.local/share/flowpolicy/miniforge-py310/bin/python /mnt/f/workspace/2/quality_fpj021.py`
  - `wsl.exe -d Ubuntu -- /home/s/.local/share/flowpolicy/miniforge-py310/bin/python scripts/validate_agent_state.py`
- Exact outputs:
  - `[1] single_flag_blocked=True file=False`
  - `[2] two_flag_blocked=True both_named=True`
  - `[3] save_after_restore=True`
  - `[4] load_rejected=True type=RuntimeError names_field=True`
  - `[4] state_untouched=True`
  - `[5] construction_rejects_disabled_path=True`
  - `FPJ-021 quality counterexamples: PASS`
- Counterexamples constructed:
  1. 单 flag 篡改后保存：阻断，且磁盘上没有文件；
  2. 双 flag 同时篡改：阻断，错误信息同时列出两个字段；
  3. 篡改后改回原值：保存恢复成功（无粘滞状态）；
  4. 篡改后 `load_checkpoint`：被拒绝，且失败加载没有改动任何张量；
  5. 构造期传 `use_down_condition=False`：被 `NotImplementedError` 拒绝。
- Findings:
  - P1: none
  - P2: 同类“配置不符”在 `load_checkpoint` 有两种异常类型：构造差异抛 `ValueError`，运行期篡改抛 `RuntimeError`。行为都正确，但训练器若只捕获 `ValueError` 会漏掉后者；FPJ-024 实现恢复逻辑时必须同时处理两者。
  - P3: `ConditionalUnet1D.__init__` 拒绝任何非“三开关全开”的配置（`conditional_unet1d.py:183-188`），因此这三个指纹项只会被运行期篡改触发；若 FPJ-028 的单路径消融要构造关闭分支的网络，必须先同步放宽该校验并扩展指纹与测试，不能只改一处。
- Remaining risks:
  - 未覆盖 CUDA 与完整通道下的相同对抗用例。
- Prohibited claims:
  - 不得把本轮的五个对抗用例说成“checkpoint 机制已完全形式化验证”。
- Recommended next role: gate
- Next agent should read:
  - `.agents/HANDOFFS.md` 中 FPJ021-IMPL-01 / FPJ021-VERIFY-01 / FPJ021-AP-01 / 本记录
  - `.agents/STATE.yaml`
  - `.agents/QUEUE.md`

## FPJ021-GATE-01

- Task: FPJ-021 Lock nested U-Net behavior into the checkpoint runtime fingerprint
- Role completed: gate
- Result: passed
- Owner: orchestrator/gate
- Files changed:
  - `.agents/STATE.yaml`
  - `.agents/HANDOFFS.md`
  - `.agents/QUEUE.md`
- Commands run:
  - `wsl.exe -d Ubuntu -- /home/s/.local/share/flowpolicy/miniforge-py310/bin/python scripts/validate_agent_state.py`
  - `wsl.exe -d Ubuntu -- /home/s/.local/share/flowpolicy/miniforge-py310/bin/python -m pytest -q`
  - `wsl.exe -d Ubuntu -- /home/s/.local/share/flowpolicy/miniforge-py310/bin/python -m pytest -q tests/test_policy.py`（重复 8 次 + 与全量交替 3 轮）
- Exact outputs:
  - `pytest -q` → `131 passed, 1 skipped`（多次运行均为该结果：11.37s / 12.36s / 10.84s / 14.88s / 20.28s 等）
  - `pytest -q tests/test_policy.py` → `27 passed`，连续 8 次全部通过；与全量交替 3 轮也全部通过
  - 唯一 skip 仍是 `tests/test_conv1d_parity.py`（无 torch），不计入 passed
  - 基线更新：`128 passed, 1 skipped` → `131 passed, 1 skipped`
- 五角色证据链:
  - implementation: FPJ021-IMPL-01
  - verification: FPJ021-VERIFY-01
  - anti_pattern: FPJ021-AP-01
  - quality: FPJ021-QUAL-01
  - gate: 本记录
- Acceptance status:
  - [x] The runtime checkpoint config reads all three live U-Net condition flags.
  - [x] Mutating any one of the three flags after construction prevents checkpoint saving.
  - [x] The checkpoint still saves and restores when the runtime config is unchanged.
  - [x] Policy checkpoint tests pass.（27 passed）
  - [x] The full CPU test suite does not regress from the recorded baseline.（131 > 128，skip 不变）
  - [x] No CUDA, PyTorch parity, EMA, Dataset, or simulator result is claimed.
- Findings:
  - P1: none
  - P2: 观察到一次不可复现的间发失败——在同一 shell 中“全量 pytest 之后立即运行 `tests/test_policy.py`”时，`test_top_level_adamw_updates_active_but_not_frozen_parameters` 失败过一次（`1 failed, 26 passed in 7.83s`）。随后 19 次运行（单测 6 次、整文件 8 次、全量+整文件交替 3 轮等）全部通过，无法复现；该用例本身没有显式设置随机种子。已在队列新增 `FPJ-030` 跟踪，不得据此宣称测试完全确定。
  - P2: `load_checkpoint` 的“配置不符”异常类型不统一（构造差异 `ValueError` / 运行期篡改 `RuntimeError`），转交 FPJ-024 处理。
  - P3: 三个指纹项只能由运行期篡改触发（构造期禁止关闭分支），FPJ-028 若做单路径消融需同步放宽校验。
- Remaining risks:
  - 基线数字虽稳定复现多次，但存在上述未解释的间发失败，属于已知不确定性。
  - 仅 CPU 小通道；policy checkpoint 仍不是 trainer checkpoint。
- Prohibited claims:
  - 不得声称已复现论文成功率、CUDA 已验证、PyTorch/Jittor 精确对齐、EMA 已完成、端到端训练或 MetaWorld rollout 已完成。
- Recommended next role: implementation（FPJ-022）
- Next agent should read:
  - `.agents/STATE.yaml`
  - `.agents/QUEUE.md`
  - `docs/REPRODUCTION_PLAN.md`（数据契约）
  - `src/flowpolicy_jittor/model/policy.py`（输入 shape 约定）

## FPJ022-IMPL-01

- Task: FPJ-022 Jittor Zarr dataset with episode-level splits and edge padding
- Role completed: implementation
- Result: passed
- Owner: orchestrator/implementation
- Files changed:
  - `src/flowpolicy_jittor/data/__init__.py`（新增）
  - `src/flowpolicy_jittor/data/dataset.py`（新增）
  - `tests/test_dataset.py`（新增）
  - `docs/DECISIONS.md`（D010、D011）
- Commands run:
  - `wsl.exe -d Ubuntu -- /home/s/.local/share/flowpolicy/miniforge-py310/bin/python scripts/validate_agent_state.py`
  - `wsl.exe -d Ubuntu -- /home/s/.local/share/flowpolicy/miniforge-py310/bin/python -m pytest -q tests/test_dataset.py`
  - `wsl.exe -d Ubuntu -- /home/s/.local/share/flowpolicy/miniforge-py310/bin/python -m pytest -q`
- Exact outputs:
  - claim：`Agent state OK: task=FPJ-022 stage=implementation status=in_progress`
  - `pytest -q tests/test_dataset.py` → `18 passed in 3.03s`，exit 0
  - `pytest -q` → `149 passed, 1 skipped in 21.00s`，exit 0；skip 仍仅为 `tests/test_conv1d_parity.py`
- Implementation notes:
  - `build_window_indices()` 是上游 `flow_policy_3d.common.sampler.create_indices` 的 NumPy 移植；窗口 `(buffer_start, buffer_end, sample_start, sample_end)` 与上游逐行一致，边界复制首帧/末帧补齐（上游 `sample_sequence` 的同款行为）。
  - `MetaworldZarrDataset` 继承 `jittor.dataset.Dataset`，`__getitem__` 返回 `{"obs": {...}, "action": ...}` 嵌套字典；Jittor 默认 collate 会把它转成嵌套 `jt.Var`（已由 DataLoader 测试覆盖）。
  - 字段名兼容：优先 `agent_pos`，回落上游实际使用的 `state`；对外统一输出 `obs.agent_pos`。
  - `get_normalizer()` 只用训练 episode（并尊重 `max_train_episodes` 下采样）拟合三字段 limits normalizer。
- Acceptance status:
  - [x] 从 Zarr 读取四个官方字段且不重写 store（Q3 证据见 quality 记录）。
  - [x] 每个样本 obs horizon 2、action horizon 4，按 episode 构造。
  - [x] 序列边界采用 padding 而不是丢窗，行为已写入 D010。
  - [x] 训练/验证按 episode 划分，窗口不跨 episode（测试 + Q1）。
  - [x] 点云 `[T,1024,3]`；尾维错误与点数不符都快速失败。
  - [x] 缺字段、空 episode、`episode_ends` 与帧数不符都快速失败。
  - [x] limits normalizer 只用训练 episode 拟合。
  - [x] 合成 Zarr fixture 覆盖 episode 边界、padding、拆分无泄漏、错误 shape、空 episode。
  - [x] 全量 CPU 测试不低于基线（131 → 149 passed，skip 不变）。
  - [x] 未声称任何仿真、训练、CUDA、EMA 或论文结果。
- Findings:
  - P1: none
  - P2: `len(dataset)` 是 Jittor 的 batch 语义（`total_len // batch_size`），不是样本数；已在 docstring 写明并提供 `num_windows`，训练器必须用后者算每 epoch 步数。
  - P2: `get_normalizer()` 会把训练 episode 的原始数组 `concatenate` 进内存；10 条示范规模没问题，但真正的大 buffer 需要改成分块统计。
  - P3: 读入后统一 `float32`；若 store 是 `float64`，精度会下降，属上游同款行为，未额外断言。
- Remaining risks:
  - 只用合成数据验证；真实 MetaWorld 示范（fpj-026）尚未生成，字段 dtype/顺序仍需现场核对。
  - 默认 `num_points=1024` 来自论文/D004，而上游任务 YAML 写的是 512；两者差异已在 D004 记录，真实数据会用 1024 生成。
- Prohibited claims:
  - 不得声称已完成真实数据训练或 rollout；本任务只交付数据读取层。
- Recommended next role: verification
- Next agent should read:
  - `src/flowpolicy_jittor/data/dataset.py`
  - `tests/test_dataset.py`
  - `docs/DECISIONS.md`（D010、D011）
  - 上游 `../flowpolicy-upstream/FlowPolicy/flow_policy_3d/common/sampler.py`

## FPJ022-VERIFY-01

- Task: FPJ-022 Jittor Zarr dataset with episode-level splits and edge padding
- Role completed: verification
- Result: passed
- Owner: orchestrator/verification
- Files changed:
  - none（只读；独立脚本在仓库外 `/mnt/f/workspace/2/verify_fpj022.py`）
- Commands run:
  - `wsl.exe -d Ubuntu -- /home/s/.local/share/flowpolicy/miniforge-py310/bin/python scripts/validate_agent_state.py`
  - `wsl.exe -d Ubuntu -- /home/s/.local/share/flowpolicy/miniforge-py310/bin/python -m pytest -q tests/test_dataset.py`
  - `wsl.exe -d Ubuntu -- /home/s/.local/share/flowpolicy/miniforge-py310/bin/python -m pytest -q`
  - `wsl.exe -d Ubuntu -- /home/s/.local/share/flowpolicy/miniforge-py310/bin/python /mnt/f/workspace/2/verify_fpj022.py`
- Exact outputs:
  - `pytest -q tests/test_dataset.py` → `18 passed in 3.03s`，exit 0
  - `pytest -q` → `149 passed, 1 skipped in 21.00s`，exit 0
  - 独立脚本：`[V1] windows_expected=23 windows_actual=23`、`[V2] padded_edges_checked=16 all_frames_covered=True`、`FPJ-022 independent verification: PASS`
- Independent evidence（实现者未提供）:
  - 按上游 `create_indices` 的语义**重新写了一份参考实现**（不调用被测函数），在 4 个长度不等 episode 的合成 store 上逐行比对 23 个窗口元组，全部一致。
  - 直接从 Zarr 数组读取帧值，独立重建每个窗口的期望内容（含首尾 padding），与 `__getitem__` 输出逐元素比对。
  - 覆盖率检查：所有真实帧都至少被一个窗口使用（没有静默丢帧）。
- Acceptance status: 与 implementation 记录一致，全部满足。
- Findings:
  - P1: none
  - P2: 参考实现只覆盖 `horizon=4 / pad 1,3`；其他 `horizon` 组合由单元测试覆盖，但没有独立参考实现。
- Remaining risks:
  - 未用真实 Zarr（fpj-026）验证；真实文件的 `episode_ends` dtype 可能是 `int32`，当前校验接受任意整型，但尚未实测。
- Prohibited claims:
  - 不得把窗口数量一致解释为“训练数据分布正确”；分布正确性要等真实示范与过拟合实验。
- Recommended next role: anti_pattern
- Next agent should read:
  - `src/flowpolicy_jittor/data/dataset.py`
  - 本记录与 FPJ022-IMPL-01

## FPJ022-AP-01

- Task: FPJ-022 Jittor Zarr dataset with episode-level splits and edge padding
- Role completed: anti_pattern
- Result: passed
- Owner: orchestrator/anti_pattern
- Files changed:
  - none（只读）
- Commands run:
  - `wsl.exe -d Ubuntu -- ... -m pytest -q`（见 gate 记录）
  - `wsl.exe -d Ubuntu -- ... /mnt/f/workspace/2/quality_fpj022.py`（只读哈希与 store 写保护检查）
  - `git status --short` + 上游目录时间戳检查
- Exact outputs:
  - `[Q3] store_unchanged=True`（读取全部窗口并拟合 normalizer 后，Zarr 目录内所有文件 size+sha256 未变）
  - `git status --short` 只显示既有 untracked 目录；`../flowpolicy-upstream/` 无写入
- Findings:
  - P1: none。没有发现“静默丢窗 / 补零 / 跨 episode 取窗 / 写回 store”的路径。
  - P2: `get_normalizer()` 一次性把训练 episode 的全部帧拼进内存；10 条示范可接受，扩大数据量时必须改为分块统计，否则会先撞内存而不是撞显存。
  - P2: `len(dataset)` 是 batch 数而不是样本数，容易被训练器误用为 steps_per_epoch；已在 docstring 提示 `num_windows`，FPJ-024 必须按此实现。
  - P3: 三字段 limits normalizer 会**逐轴**缩放点云 XYZ，这是上游行为；答辩时要说明“点云被各轴独立归一化”，不要说成保持米制尺度。
  - P3: 输入统一转 `float32`，`float64` store 会降精度，未断言。
- Remaining risks:
  - 仓库仍无首次 commit，无法用 diff 证明“只改了 4 个文件”。
  - 上游 `data/**`、`checkpoints/**`、`outputs/**` 目前不存在，禁止路径约束只能靠不创建来满足。
- Prohibited claims:
  - 不得声称已用真实示范验证数据分布。
- Recommended next role: quality
- Next agent should read:
  - `src/flowpolicy_jittor/data/dataset.py`（`_open_arrays`、`_validate_episode_ends`、`get_normalizer`）
  - 本记录

## FPJ022-QUAL-01

- Task: FPJ-022 Jittor Zarr dataset with episode-level splits and edge padding
- Role completed: quality
- Result: passed
- Owner: orchestrator/quality
- Files changed:
  - none（只读；对抗脚本在仓库外 `/mnt/f/workspace/2/quality_fpj022.py`）
- Commands run:
  - `wsl.exe -d Ubuntu -- /home/s/.local/share/flowpolicy/miniforge-py310/bin/python /mnt/f/workspace/2/quality_fpj022.py`
- Exact outputs:
  - `[Q1] windows_crossing_episode_boundary=0`
  - `[Q2] returned_array_is_copy=True`
  - `[Q3] store_unchanged=True`
  - `[Q4] horizon1_windows=4 pad_before=0 pad_after=0`
  - `[Q5] short_episode_windows=3 last_row_repeats=True`
  - `[Q6] lonely_train_mask=[True] windows=7`
  - `[Q7] shuffled_epoch_batches=15 windows=15`
  - `[Q8] kept_episodes=[0, 2] normalizer_max=5005.0 expected_max=5005.0`
  - `FPJ-022 quality counterexamples: PASS`
- Counterexamples constructed:
  1. 窗口不跨 episode 边界；
  2. 返回数组是副本（改一次读到的数组不影响下一次读取）；
  3. 读取全部窗口 + 拟合 normalizer 后 Zarr 文件字节级不变；
  4. `horizon=1` 时 `pad_before/pad_after` 被夹到 0，不崩；
  5. 比 horizon 短的 episode 仍产出窗口并复制末帧；
  6. 单 episode + `val_ratio>0` 时该 episode 不会既当训练又当验证（验证集为空并显式报错）；
  7. `shuffle=True` 的一轮遍历恰好访问 `num_windows` 个样本；
  8. `max_train_episodes` 下采样同时作用于 normalizer 统计（被丢弃的 episode 不会进入 min/max）。
- Findings:
  - P1: none
  - P2: 第 8 项最初用错了期望公式（脚本里写死 `10*index+5`，而 store 的 base 是 `[0,10,5000,9000]`），得到假阳性 `normalizer_max=5005 != 25`；改为按实际 base 计算后一致。**这是检查脚本的缺陷，不是产品的缺陷**，记录在此以免后来者误读。
  - P3: `get_validation_dataset()` 会重新打开 store 并重建索引，开销可接受但不为零；训练器应复用对象而不是每 epoch 重建。
- Remaining risks:
  - 未覆盖真实 Zarr（fpj-026）与多 worker（`num_workers>0`）路径；当前只用 `num_workers=0` 验证。
- Prohibited claims:
  - 不得把 8 个用例通过表述为“数据管道已充分验证”。
- Recommended next role: gate
- Next agent should read:
  - 本文件 FPJ022-IMPL-01 至本记录
  - `.agents/STATE.yaml`

## FPJ022-GATE-01

- Task: FPJ-022 Jittor Zarr dataset with episode-level splits and edge padding
- Role completed: gate
- Result: passed
- Owner: orchestrator/gate
- Files changed:
  - `.agents/STATE.yaml`
  - `.agents/HANDOFFS.md`
  - `.agents/QUEUE.md`
  - `docs/REPRODUCTION_PLAN.md`（勾选已完成的数据契约检查项）
- Commands run:
  - `wsl.exe -d Ubuntu -- /home/s/.local/share/flowpolicy/miniforge-py310/bin/python scripts/validate_agent_state.py`
  - `wsl.exe -d Ubuntu -- /home/s/.local/share/flowpolicy/miniforge-py310/bin/python -m pytest -q`
  - `wsl.exe -d Ubuntu -- /home/s/.local/share/flowpolicy/miniforge-py310/bin/python -m pytest -q tests/test_dataset.py`
- Exact outputs:
  - `pytest -q` → `149 passed, 1 skipped in 21.00s`，exit 0
  - `pytest -q tests/test_dataset.py` → `18 passed in 3.03s`，exit 0
  - 唯一 skip 仍是 `tests/test_conv1d_parity.py`（无 torch）
  - 基线更新：`131 passed, 1 skipped` → `149 passed, 1 skipped`
- 五角色证据链:
  - implementation: FPJ022-IMPL-01
  - verification: FPJ022-VERIFY-01
  - anti_pattern: FPJ022-AP-01
  - quality: FPJ022-QUAL-01
  - gate: 本记录
- Acceptance status: 全部满足（详见 FPJ022-IMPL-01 的 10 条勾选）。
- Findings:
  - P1: none
  - P2: `get_normalizer()` 全量拼内存、`len(dataset)` 的 batch 语义、`num_windows` 使用约定，已转交 FPJ-023/FPJ-024。
  - P3: 点云逐轴 limits 归一化属于上游行为，答辩需主动说明。
- Remaining risks:
  - 仍只用合成数据；真实示范、训练循环、仿真 rollout、CUDA 均未完成。
- Prohibited claims:
  - 不得声称已用真实数据训练或已得到任何成功率。
- Recommended next role: implementation（FPJ-023）
- Next agent should read:
  - `.agents/STATE.yaml`
  - `.agents/QUEUE.md`
  - `src/flowpolicy_jittor/data/dataset.py`
  - `tests/test_dataset.py`

## FPJ023-IMPL-01

- Task: FPJ-023 Overfit 32 deterministic sequence windows on a small CPU config
- Role completed: implementation
- Result: passed
- Owner: orchestrator/implementation
- Files changed:
  - `src/flowpolicy_jittor/data/dataset.py`（新增 `select_windows`）
  - `scripts/overfit_32_windows.py`（新增）
  - `tests/test_overfit.py`（新增）
  - `docs/DECISIONS.md`（D012、D013）
  - `docs/REPRODUCTION_PLAN.md`（勾选 10-batch 过拟合项）
- Commands run:
  - `wsl.exe -d Ubuntu -- ... -m pytest -q tests/test_overfit.py`
  - `wsl.exe -d Ubuntu -- ... scripts/overfit_32_windows.py --steps 200 --num-points 32 --log /tmp/fpj023.json`
  - `wsl.exe -d Ubuntu -- ... -m pytest -q`
- Exact outputs:
  - `pytest -q tests/test_overfit.py` → `5 passed, 8 warnings in 21.03s`
  - `scripts/overfit_32_windows.py --steps 200 --num-points 32` →
    `[action-error] initial=5.486259 final=1.418001 reduction=74.15%`，
    `[baselines] zero=2.120174 mean_action=2.109612`，
    `[loss] initial=0.000142 min=0.000013 final=0.000013`，
    `[time] 14.551s for 200 steps (OMP_NUM_THREADS=4)`
  - `pytest -q` → `154 passed, 1 skipped, 8 warnings in 29.90s`；skip 仍只有 `test_conv1d_parity.py`
  - 额外趋势数据点（非验收要求）：`--steps 800` → `final=0.941041`（82.85%），loss min `4e-6`，54.4s
- Implementation notes:
  - `select_windows()` 用 `np.random.default_rng(seed)` 无放回抽取并排序，固定 seed 下逐位可复现。
  - 脚本在 import jittor **之前** `os.environ.setdefault("OMP_NUM_THREADS", "4")`：本机 24 核时 Jittor 默认线程数会让同配置从 0.15 s/step 退化到 3 s/step（见 D013）。
  - 无 `--zarr` 时脚本自己写一个**可学习**的合成 store（动作 = 状态 @ 固定矩阵 + 0.01 噪声），因此过拟合考的是“能否拟合映射”，不是“能否记住纯噪声”。
  - 主指标为固定噪声（`seed+1`）下 NFE=1 的预测动作误差，并要求同时低于零预测与均值预测（D012）；CFM loss 作为辅助记录。
  - 两条守卫：`enforce_reduction=True`（CLI 默认，无法关闭）要求误差减半且打败常数预测器；非有限 loss 立即 `RuntimeError`。
- Acceptance status:
  - [x] 32 个窗口，固定 seed 可复现，且互不重复。
  - [x] 动作误差至少减半（5.49 → 1.42）。
  - [x] loss 的 initial/min/final 记录且有限。
  - [x] JSON 日志包含 loss、动作误差、窗口索引、标签。
  - [x] 日志与打印都标注 “32-window smoke overfit (NOT paper-scale training)”。
  - [x] AdamW 来自 `policy.trainable_parameters()`（独立校验 V5 也确认脚本内不出现 `policy.parameters()`）。
  - [x] 不能记住的配置（1 步）会抛错而不是通过。
  - [x] 全量基线不低于 149 passed，现为 154 passed, 1 skipped。
  - [x] 未声称仿真、CUDA、EMA 或论文结果。
- Findings:
  - P1: none
  - P2: 200 步只把误差压到 1.42（约为常数预测器的 67%），远未到“几乎记住”的程度；800 步到 0.941。说明小通道 + 32 点配置的容量/步数有限，正式训练需要更大预算（转交 FPJ-024/FPJ-027）。
  - P2: 动作误差只用**一个**固定噪声样本测量，没有对多个噪声样本取平均；噪声换一个，数值会有小幅变化。
  - P2: 合成数据的点云是纯噪声，只有 `agent_pos` 携带可学习信息，所以这次过拟合**没有**验证 PointNet 分支的有效性；真实示范（FPJ-026）之后必须重做一次。
  - P3: 默认 `--num-points 32` 是为了 CPU 冒烟；与 1024 点契约不同，日志里记录了实际值。
  - P3: `test_script_reduces_action_error_by_half` 用子进程跑真实脚本（约 20 s），把测试套件从 21 s 拉到 30 s。
- Remaining risks:
  - 训练 200 步的结论不能外推到论文配置；`num_segments`、NFE、EMA 都没有在这里被触及。
  - 只验证了 `val_ratio=0` 的合成 store，没有验证“训练集/验证集同时存在”时的冒烟行为。
- Prohibited claims:
  - 不得把该结果描述为论文训练或收敛证据。
  - 不得声称点云分支已被验证。
- Recommended next role: verification
- Next agent should read:
  - `scripts/overfit_32_windows.py`
  - `tests/test_overfit.py`
  - `docs/DECISIONS.md`（D012、D013）

## FPJ023-VERIFY-01

- Task: FPJ-023 Overfit 32 deterministic sequence windows on a small CPU config
- Role completed: verification
- Result: passed
- Owner: orchestrator/verification
- Files changed:
  - none（只读；独立脚本在仓库外 `/mnt/f/workspace/2/verify_fpj023.py`）
- Commands run:
  - `wsl.exe -d Ubuntu -- ... -m pytest -q tests/test_overfit.py`
  - `wsl.exe -d Ubuntu -- ... scripts/overfit_32_windows.py --steps 200 --num-points 32 --log /tmp/fpj023.json`
  - `wsl.exe -d Ubuntu -- ... /mnt/f/workspace/2/verify_fpj023.py`
- Exact outputs:
  - `pytest -q tests/test_overfit.py` → `5 passed, 8 warnings in 21.03s`
  - 独立脚本：
    - `[V1] windows_recorded=32 distinct=32 matches_reference=True`
    - `[V2] zero recomputed=2.120174 logged=2.120174 | mean recomputed=2.109612 logged=2.109612`
    - `[V3] trained_final=1.418001 best_baseline=2.109612`
    - `[V4] deterministic_repeat=True`
    - `[V5] no policy.parameters()=True no use_cuda=1=True no torch import=True uses trainable_parameters=True`
    - `FPJ-023 independent verification: PASS`
- Independent evidence（实现者未提供）:
  - 用 `np.random.default_rng(0)` **重新生成**合成 store 的内容（状态/点云/权重/动作），不经过被测模块，再按上游 `create_indices` 语义独立重建窗口表，重新抽取 32 个索引：与日志 `window_indices` 逐位一致。
  - 用独立重建的窗口内容（含首尾复制 padding）重算两个常数预测器误差，与日志中的数值一致到 1e-6，说明日志数字确实来自数据而不是硬编码。
  - 同参数跑两次，除 `seconds` 外日志完全相同（逐字段比较）。
  - 静态检查：脚本没有 `policy.parameters()`、没有 `use_cuda = 1`、没有 `import torch`，且确实使用了 `trainable_parameters()`。
- Acceptance status: 与 implementation 记录一致，全部满足。
- Findings:
  - P1: none
  - P2: 独立重算只覆盖默认 seed=0 / 32 点配置；换 seed 的正确性只由单元测试覆盖。
  - P2: 两次运行一致只能证明“同机同配置可复现”，不能证明跨机器可复现（线程数、编译器版本未固定）。
- Remaining risks:
  - 独立校验没有覆盖 `--zarr` 真实数据路径。
- Prohibited claims:
  - 不得把“可复现”解释为“结果与论文一致”。
- Recommended next role: anti_pattern
- Next agent should read:
  - 本记录与 FPJ023-IMPL-01

## FPJ023-AP-01

- Task: FPJ-023 Overfit 32 deterministic sequence windows on a small CPU config
- Role completed: anti_pattern
- Result: passed
- Owner: orchestrator/anti_pattern
- Files changed:
  - none（只读）
- Commands run:
  - `wsl.exe -d Ubuntu -- ... /mnt/f/workspace/2/quality_fpj023.py`
  - `grep`/静态检查（见下）
- Exact outputs:
  - `[Q2] divergence_raised='non-finite loss at step 2: nan'`
  - `[Q5] cli_can_disable_guard=False`
  - `[Q6] val_ratio_pinned_to_zero=True`
  - `[Q7] labelled=True prohibited_claims=4`
- Findings:
  - P1: none。没有发现“让 loss 好看但没学到东西”的路径：主指标是动作误差且必须打败常数预测器；守卫无法从 CLI 关闭。
  - P2: 训练只在**训练窗口**上评估（`val_ratio=0` + 同一批 32 个窗口），因此这是纯粹的记忆测试，不能说明泛化；答辩时必须明确区分。
  - P2: 默认 200 步的“74%”不足以称为“已过拟合”，真正接近记住需要 800 步以上（0.941）。
  - P3: 脚本会在系统临时目录写合成 store（`tempfile.mkdtemp`），不会自动清理；对本机磁盘影响很小，但注意别在 CI 上无限重跑。
  - P3: 日志里记录了 `prohibited_claims` 列表，但这是自证，不是自动检查；真正的防护是全量测试与 gate 记录。
- Remaining risks:
  - 没有和上游 PyTorch 的同类过拟合曲线对比，无法判断“下降速度是否正常”。
- Prohibited claims:
  - 不得把该过拟合结果当成泛化或论文性能证据。
- Recommended next role: quality
- Next agent should read:
  - `scripts/overfit_32_windows.py`
  - 本记录

## FPJ023-QUAL-01

- Task: FPJ-023 Overfit 32 deterministic sequence windows on a small CPU config
- Role completed: quality
- Result: passed
- Owner: orchestrator/quality
- Files changed:
  - none（只读；对抗脚本在仓库外 `/mnt/f/workspace/2/quality_fpj023.py`）
- Commands run:
  - `wsl.exe -d Ubuntu -- ... /mnt/f/workspace/2/quality_fpj023.py`
- Exact outputs:
  - `[Q1] too_few_windows_raised='requested 32 windows but the dataset only has 11'`
  - `[Q2] divergence_raised='non-finite loss at step 2: nan'`
  - `[Q3] non_positive_steps_rejected=True`
  - `[Q4] nested_log_created=True`
  - `[Q5] cli_can_disable_guard=False`
  - `[Q6] val_ratio_pinned_to_zero=True`
  - `[Q7] labelled=True prohibited_claims=4`
  - `[Q8] point_mismatch_reported=True returncode=1`
  - `FPJ-023 quality counterexamples: PASS`
- Counterexamples constructed:
  1. 窗口不足 32 的 store 必须显式报错，不得静默用现有窗口训练；
  2. `lr=1e3` 发散时必须以 `non-finite loss` 抛错，不得静默写出 NaN 日志；
  3. `steps<=0` 在开始训练前被拒绝；
  4. 多层不存在的日志目录会被创建且写出合法 JSON；
  5. CLI 不能关闭 50% / 常数预测器守卫；
  6. `val_ratio` 被脚本固定为 0；
  7. 日志带 smoke 标签与 4 条禁止声明；
  8. `--zarr` 与 `--num-points` 不一致时报 “32 points” 并以非零码退出。
- Findings:
  - P1: none
  - P2: 发散用例证明守卫有效，但它依赖 loss 在若干步内变成 NaN；若发散较慢（loss 缓慢增大但有限），只会被“常数预测器”守卫拦住，错误信息会指向另一条判据。
  - P3: 第 8 个用例依赖 dataset 的错误文本，若将来改文案需要同步改检查。
- Remaining risks:
  - 未覆盖 `--zarr` + 真实 1024 点数据的性能，CPU 上很可能不可行（转交 FPJ-027/CUDA）。
- Prohibited claims:
  - 不得把 8 个反例通过表述为“训练循环已充分验证”。
- Recommended next role: gate
- Next agent should read:
  - 本文件 FPJ023-IMPL-01 至本记录

## FPJ023-GATE-01

- Task: FPJ-023 Overfit 32 deterministic sequence windows on a small CPU config
- Role completed: gate
- Result: passed
- Owner: orchestrator/gate
- Files changed:
  - `.agents/STATE.yaml`
  - `.agents/HANDOFFS.md`
  - `.agents/QUEUE.md`
- Commands run:
  - `wsl.exe -d Ubuntu -- ... scripts/validate_agent_state.py`
  - `wsl.exe -d Ubuntu -- ... scripts/overfit_32_windows.py --steps 200 --num-points 32 --log /tmp/fpj023.json`
  - `wsl.exe -d Ubuntu -- ... -m pytest -q`
- Exact outputs:
  - `Agent state OK: task=FPJ-023 stage=implementation status=ready`（claim 时）
  - `[action-error] initial=5.486259 final=1.418001 reduction=74.15%`
  - `[loss] initial=0.000142 min=0.000013 final=0.000013`
  - `pytest -q` → `154 passed, 1 skipped, 8 warnings in 29.90s`；唯一 skip 仍是 `test_conv1d_parity.py`
  - 基线更新：`149 passed, 1 skipped` → `154 passed, 1 skipped`
- 五角色证据链:
  - implementation: FPJ023-IMPL-01
  - verification: FPJ023-VERIFY-01
  - anti_pattern: FPJ023-AP-01
  - quality: FPJ023-QUAL-01
  - gate: 本记录
- Acceptance status: 全部满足（详见 FPJ023-IMPL-01 的 9 条勾选）。
- Findings:
  - P1: none
  - P2: 动作误差单噪声测量、点云分支未被验证、200 步容量不足，三条都转交 FPJ-026/FPJ-027。
  - P3: 测试套件时长从 21 s 升到 30 s（真实脚本子进程）。
- Remaining risks:
  - 仍无真实示范数据；训练循环还是一次性脚本，没有 checkpoint / resume / 学习率调度。
- Prohibited claims:
  - 不得声称训练流程已端到端可用（缺 checkpoint、scheduler、resume）。
- Recommended next role: implementation（FPJ-024）
- Next agent should read:
  - `.agents/STATE.yaml`
  - `.agents/QUEUE.md`
  - `scripts/overfit_32_windows.py`
  - `src/flowpolicy_jittor/model/policy.py`（`save_checkpoint`/`load_checkpoint`）

## FPJ024-IMPL-01

- Task: FPJ-024 Minimal trainer with warmup+cosine schedule and resumable trainer checkpoint
- Role completed: implementation
- Result: passed
- Owner: orchestrator/implementation
- Files changed:
  - `src/flowpolicy_jittor/training/__init__.py`（新增）
  - `src/flowpolicy_jittor/training/trainer.py`（新增）
  - `tests/test_trainer.py`（新增，20 个测试）
  - `docs/DECISIONS.md`（D014）
  - `.agents/QUEUE.md`（新增 FPJ-031；更新 FPJ-024 条目）
- Commands run:
  - `wsl.exe -d Ubuntu -- ... -m pytest -q tests/test_trainer.py`
  - `wsl.exe -d Ubuntu -- ... -m pytest -q`
- Exact outputs:
  - `pytest -q tests/test_trainer.py` → `20 passed in 5.50s`
  - `pytest -q` → `174 passed, 1 skipped, 8 warnings in 27.68s`（OMP_NUM_THREADS=4）
  - 同一条基线命令不设 OMP 时约 71s（本机 24 核线程争用，见 D013）；通过数不变
- Implementation notes:
  - `warmup_cosine_lr(step, ...)` 是纯函数：`step=0` 返回 0，warmup 段线性上升到 `base_lr`，之后余弦衰减到 `base_lr*min_lr_ratio`；`warmup=0` 时从第 1 步的 base 开始衰减。
  - `Trainer` 用 `policy.trainable_parameters()` 构造 AdamW（测试断言 optimizer 的 param 列表与之一一对应），每步用 `(seed, step)` 派生 CFM 噪声与时间，因此续训不依赖 Jittor 全局 RNG。
  - trainer checkpoint = 内嵌 policy payload（经 policy 自己的严格写入器产生，恢复时也走 policy 的严格加载器）+ `m`/`values`/`n_step`/`global_step`/调度配置/`seed`/最近 50 步 history/`ema: null`；`kind` 字段区分两类 checkpoint。
  - 保存走同目录临时文件 + `os.replace`，默认拒绝覆盖；恢复前校验调度、seed、参数布局，任一项不符即失败且不改动状态。
- Acceptance status:
  - [x] AdamW 只来自 `policy.trainable_parameters()`，26 个 stop-grad 参数在 12 步后逐位不变。
  - [x] warmup+cosine 是 step 的纯函数，端点（0 / warmup / total / 无 warmup / total=1）都有断言。
  - [x] checkpoint 含 policy 状态、optimizer 状态、global step、调度配置与 `ema: null`。
  - [x] 状态恢复逐位精确（权重、AdamW 矩、n_step、global_step 全部 `array_equal`），学习率调度精确延续；轨迹在 2e-3 绝对容差内一致（见 P1）。
  - [x] policy 与 trainer checkpoint 互斥报错。
  - [x] 日志记录 step/lr/loss，非有限 loss 立即抛错且不推进 global_step。
  - [x] 保存不隐式覆盖，失败不留临时文件。
  - [x] 全量基线由 154 升到 174 passed，skip 不变。
  - [x] 未声称仿真、CUDA、EMA 或论文结果。
- Findings:
  - P1: CPU 后端无法保证 bit-exact 续训轨迹：实测“保存→恢复→再训三步”与“不中断六步”最大权重差 `8.3e-4`、第 5 步 loss 相对差 `6.6e-5`（权重/矩/步数恢复本身逐位一致）。原因：本任务 loss 与梯度极小，AdamW 除以 `sqrt(v)` 放大 float32 舍入。已写入 D014，并新开 FPJ-031 排查；**项目不声称 bit-exact 续训**。
  - P2: checkpoint 内嵌完整 policy payload，文件体积约为参数量的 2 倍（policy + AdamW 矩）。
  - P2: checkpoint 只保留最近 50 步 history，长训练需要外部日志（JSONL 已是 append 模式）。
  - P2: 续训要求 `total_steps`/`warmup_steps`/`min_lr_ratio`/`lr`/`seed` 完全一致；改变总步数必须新建 trainer，且会被拒绝加载，属于刻意设计但会限制“延长训练”的场景。
  - P3: 每次保存会先写一个临时 policy 文件再内嵌，多了两次磁盘往返。
  - P3: JSONL 日志是 append 的，重复运行写同一路径会累积，调用方需自行区分 run。
- Remaining risks:
  - 训练器还没接真实数据（FPJ-026/027）、没有 EMA、没有梯度裁剪与混合精度。
  - 只在 num_points=8/16 的合成 batch 上验证，未在 1024 点真实数据上跑过。
- Prohibited claims:
  - 不得声称 bit-exact 续训。
  - 不得声称训练器等同于论文训练配方（缺 EMA、数据规模、CUDA）。
- Recommended next role: verification
- Next agent should read:
  - `src/flowpolicy_jittor/training/trainer.py`
  - `tests/test_trainer.py`
  - `docs/DECISIONS.md`（D014）

## FPJ024-VERIFY-01

- Task: FPJ-024 Minimal trainer with warmup+cosine schedule and resumable trainer checkpoint
- Role completed: verification
- Result: passed
- Owner: orchestrator/verification
- Files changed:
  - none（只读；独立脚本在仓库外 `/mnt/f/workspace/2/verify_fpj024.py`）
- Commands run:
  - `wsl.exe -d Ubuntu -- ... -m pytest -q tests/test_trainer.py`
  - `wsl.exe -d Ubuntu -- ... /mnt/f/workspace/2/verify_fpj024.py`
- Exact outputs:
  - `pytest -q tests/test_trainer.py` → `20 passed in 5.50s`
  - `[V1] schedule compared against an independent implementation`
  - `[V2] lr_trace_ok=True`
  - `[V3] moments_match_live_optimizer m=True v=True n_step=4==4`
  - `[V4] kind='flowpolicy-jittor.trainer' policy_keys_equal=True types_equal=True`
  - `[V5] ema_marker=None`
  - `[V6] trainer_rejects_policy=True policy_rejects_trainer=True`
  - `[V7] leftover_temp_files=[]`
  - `[V8] log_records=4 history=4`
  - `FPJ-024 independent verification: PASS`
- Independent evidence（实现者未提供）:
  - 用**独立写出的**余弦调度公式在 4 组配置（含 `total=1`、`warmup=total`、`min_lr_ratio=0`）上逐步比对 0..total，误差 < 1e-12。
  - 逐步日志里的 lr 与独立公式逐值一致（不是“最后一步 lr 被复用”）。
  - `optimizer_state()` 返回的 `m`/`values` 与 Jittor optimizer 内部的 `param_groups[0]["m"]/["values"]` 逐位相同，`n_step` 与 optimizer 同步 → 训练器没有伪造状态。
  - trainer checkpoint 内嵌的 `policy` 与直接用 `policy.save_checkpoint` 写出的 payload 键集合与类型逐一相同。
  - 双向互斥：trainer 拒绝 policy 文件，policy 拒绝 trainer 文件。
  - 保存后目录内无临时文件残留；JSONL 与内存 history 完全一致。
- Acceptance status: 与 implementation 记录一致（含 P1 的容差说明）。
- Findings:
  - P1: 与实现一致，CPU 轨迹漂移已量化并记录，未发现“日志与真实 lr 不一致”或“伪造 optimizer 状态”的问题。
  - P2: 独立公式只覆盖 `warmup<=total` 的合法区间，非法区间由 quality 记录覆盖。
- Remaining risks:
  - 未验证多进程/多 epoch 的真实数据加载下的调度连续性。
- Prohibited claims:
  - 不得把“独立公式一致”表述为“与上游 PyTorch 训练器一致”；上游训练器尚未对照。
- Recommended next role: anti_pattern
- Next agent should read:
  - 本记录与 FPJ024-IMPL-01

## FPJ024-AP-01

- Task: FPJ-024 Minimal trainer with warmup+cosine schedule and resumable trainer checkpoint
- Role completed: anti_pattern
- Result: passed
- Owner: orchestrator/anti_pattern
- Files changed:
  - none（只读）
- Commands run:
  - `wsl.exe -d Ubuntu -- ... /mnt/f/workspace/2/quality_fpj024.py`
  - 静态检查：`grep -n "policy.parameters()" src/flowpolicy_jittor/training/trainer.py`（无命中）、`grep -n "use_cuda" ...`（无命中）
- Exact outputs:
  - `[Q3] frozen_parameters_changed=False (26 frozen)`
  - `[Q6] leftovers_after_failure=[]`
  - `[Q7 mutated policy] raised RuntimeError: 'policy behavior changed after construction; rebuild the poli'`
  - `[Q8] rejected_resume_left_policy_untouched=True global_step=0`
- Findings:
  - P1: none。训练器没有绕过 policy 的配置指纹、没有用 `policy.parameters()` 建优化器、没有在失败路径留下半应用状态。
  - P2: 训练器把 `seed` 写进 checkpoint 并强制一致，这是正确做法，但也意味着“换 seed 续训”必须显式新建 run，不能悄悄换。
  - P2: 训练器没有实现梯度裁剪；上游训练配置若有 `grad_clip`，正式训练前必须补齐，否则与原论文训练细节不一致。
  - P3: `histroy` 截断到 50 条会让“从 checkpoint 恢复完整 loss 曲线”不可能，正式实验必须依赖 JSONL。
  - P3: `_policy_payload()` 的临时文件中转在 Windows 路径/权限异常时会先失败再清理，错误信息是 policy 写入器的原文，可接受但不够显眼。
- Remaining risks:
  - 未与上游 `train_flow_policy_3d` 的超参数逐项对照（学习率、warmup 比例、weight_decay、batch size）。
- Prohibited claims:
  - 不得声称训练器已对齐论文训练配方。
- Recommended next role: quality
- Next agent should read:
  - `src/flowpolicy_jittor/training/trainer.py`
  - 本记录

## FPJ024-QUAL-01

- Task: FPJ-024 Minimal trainer with warmup+cosine schedule and resumable trainer checkpoint
- Role completed: quality
- Result: passed
- Owner: orchestrator/quality
- Files changed:
  - none（只读；对抗脚本在仓库外 `/mnt/f/workspace/2/quality_fpj024.py`）
- Commands run:
  - `wsl.exe -d Ubuntu -- ... /mnt/f/workspace/2/quality_fpj024.py`
- Exact outputs:
  - `[Q1 ...]` 6 组非法构造参数全部 `ValueError`
  - `[Q2 malformed action]` `ValueError: batch action must be [B, T, A], got shape (3, 4)`；`[Q2 missing action]` `KeyError`
  - `[Q3] frozen_parameters_changed=False (26 frozen)`
  - `[Q4 wrong kind]` / `[Q4 ema present]` / `[Q4 missing trainer key]` / `[Q4 float global_step]` / `[Q4 corrupted moment shape]` 全部按预期抛错
  - `[Q5 seed mismatch] raised ValueError: 'trainer checkpoint seed 5 does not match trainer seed 6'`
  - `[Q6] leftovers_after_failure=[]`
  - `[Q7 mutated policy] raised RuntimeError`
  - `[Q8] rejected_resume_left_policy_untouched=True global_step=0`
  - `FPJ-024 quality counterexamples: PASS`
- Counterexamples constructed:
  1. 非法构造参数（`total_steps=0`、`warmup>total`、`min_lr_ratio>1`、负 lr、NaN lr、`log_every=0`）；
  2. 形状错误与缺少 action 的 batch；
  3. 26 个 stop-grad 参数在 12 步后逐位不变；
  4. 5 类被篡改的 checkpoint（kind、EMA、缺键、非整型 global_step、矩形状错误）；
  5. 不同 seed 不能续训；
  6. 不可用父路径保存失败且不留临时文件；
  7. 构造后被修改的 policy 拒绝保存；
  8. 被拒绝的恢复不改变权重与 `global_step`。
- Findings:
  - P1: none
  - P2: Q8 第一次写法有 bug（拿 `fresh_policy` 的快照去比 `other2_policy`），得到假阳性；修正为对同一对象取快照后通过。**这是检查脚本的缺陷，不是产品缺陷**。
  - P3: 篡改用例依赖 `jt.load` 能读回被手工修改的 dict；如果 checkpoint 格式将来改成二进制校验和，这些用例需要重写。
- Remaining risks:
  - 未覆盖磁盘写入中断（真正断电/进程被杀）时的原子性；只覆盖了“拒绝覆盖”和“父路径不可用”。
- Prohibited claims:
  - 不得把 8 类反例通过表述为“训练器已可安全用于长跑实验”。
- Recommended next role: gate
- Next agent should read:
  - 本文件 FPJ024-IMPL-01 至本记录

## FPJ024-GATE-01

- Task: FPJ-024 Minimal trainer with warmup+cosine schedule and resumable trainer checkpoint
- Role completed: gate
- Result: passed
- Owner: orchestrator/gate
- Files changed:
  - `.agents/STATE.yaml`
  - `.agents/HANDOFFS.md`
  - `.agents/QUEUE.md`
- Commands run:
  - `wsl.exe -d Ubuntu -- ... scripts/validate_agent_state.py`
  - `wsl.exe -d Ubuntu -- ... -m pytest -q tests/test_trainer.py`
  - `wsl.exe -d Ubuntu -- ... -m pytest -q`
- Exact outputs:
  - `Agent state OK: task=FPJ-024 stage=implementation status=ready`（claim 时）
  - `pytest -q tests/test_trainer.py` → `20 passed in 5.50s`
  - `pytest -q` → `174 passed, 1 skipped, 8 warnings in 27.68s`
  - 唯一 skip 仍是 `test_conv1d_parity.py`（无 torch）
  - 基线更新：`154 passed, 1 skipped` → `174 passed, 1 skipped`
- 五角色证据链:
  - implementation: FPJ024-IMPL-01
  - verification: FPJ024-VERIFY-01
  - anti_pattern: FPJ024-AP-01
  - quality: FPJ024-QUAL-01
  - gate: 本记录
- Acceptance status: 全部满足；其中“续训一致”按 D014 的实测口径执行（状态与调度精确、轨迹 2e-3 容差），并在 QUEUE 中登记 FPJ-031。
- Findings:
  - P1: CPU 轨迹漂移（8.3e-4）作为已知限制记录，不作为通过条件被隐藏。
  - P2: 缺梯度裁剪、history 截断、续训不能改调度、checkpoint 体积翻倍，转交 FPJ-027/FPJ-028。
  - P3: 临时文件中转与 JSONL append 语义。
- Remaining risks:
  - 训练器尚未接触真实数据；没有 EMA；没有 CUDA 路径；没有与上游超参数逐项对照。
- Prohibited claims:
  - 不得声称训练器已可用于论文级实验或已实现完整训练配方。
- Recommended next role: implementation（FPJ-025）
- Next agent should read:
  - `.agents/STATE.yaml`
  - `.agents/QUEUE.md`
  - `src/flowpolicy_jittor/training/trainer.py`

## FPJ025-IMPL-01

- Task: FPJ-025 MetaWorld / MuJoCo / EGL demonstration environment (or a documented substitute)
- Role completed: implementation
- Result: passed
- Owner: orchestrator/implementation
- Files changed:
  - `env/sim_setup_log.md`（新增，603 行，由子代理产出后本人复核）
  - `env/metaworld_contract.md`（新增，官方观测契约转写 + 实际实现差异）
  - `scripts/record_demos.py`（新增：质量检查、录制、写入器、契约校验）
  - `src/flowpolicy_jittor/data/dataset.py`（支持官方 `data/` + `meta/` 布局）
  - `src/flowpolicy_jittor/data/__init__.py`（导出数据集 API）
  - `tests/test_dataset.py`（新增 3 个布局测试）
  - `docs/DECISIONS.md`（D015、D016）
  - `docs/REPRODUCTION_PLAN.md`（Phase 3 清单更新）
- Commands run:
  - `wsl.exe -d Ubuntu -- ... -m pytest -q`
  - `MUJOCO_GL=egl /home/s/.local/share/flowpolicy/sim-env/bin/python scripts/record_demos.py --record --task reach-v3 --episodes 1 --max-steps 60 --seed 7 --output /tmp/mw_prov.zarr`
  - `... miniforge-py310/bin/python scripts/record_demos.py --check-store /tmp/mw_prov.zarr`
  - `/home/s/.local/share/flowpolicy/miniforge-py310/bin/python /mnt/f/workspace/2/verify_fpj025.py`
- Exact outputs:
  - `pytest -q` → `177 passed, 1 skipped`（基线 174 → 177）
  - 录制：`[collect] episode 1/1: kept, 60 steps, 20 successful, attempt 1`
  - `--check-store` → `status ok`，`windows 61`，`point_cloud [2, 1024, 3]`，`agent_pos [2, 9]`，`action [4, 4]`
  - 校验脚本 → `FPJ-025 verification and adversarial checks: PASS`
- Implementation notes:
  - 仿真栈：`/home/s/.local/share/flowpolicy/sim-env`（Python 3.10.14，mujoco 3.3.7、metaworld 3.0.0、
    gymnasium 1.4.0、zarr 2.12.0），`MUJOCO_GL=egl` 走 Mesa llvmpipe 软件渲染；Jittor 环境零污染。
  - 保留官方契约：`corner2` 相机、`cam_pos=[0.6,0.295,0.8]`、znear/zfar 0.1/1.5、128×128 深度、
    `Rx(61.4°)@Ry(-7°)`、FPS 到 1024 点、9 维 `agent_pos`、脚本专家、成功步数 ≥5 的过滤、官方 Zarr 布局。
  - 关键改动（D016）：现代 `mujoco.Renderer` 深度已是米制且已翻正，不能再套旧的
    `near/(1-d(1-near/far))` 与 `verticalFlip`；官方裁剪框在现代栈上只剩 0~100/16384 点，
    改为世界坐标框 `(-0.35,0.25,-0.05)~(0.35,0.95,0.55)`（几何验证：1173 点落在 z=-0.913 地面、
    5233 点落在 z=0 桌面，框内保留 28.5%）。
  - 数据集（D015）：同时支持扁平布局与官方 `data/` + `meta/episode_ends`；用成员测试而非
    `array_keys()` 枚举，因为 zarr 2.x 对名为 `meta` 的子组有断言。
  - 写入器默认拒绝覆盖、放在 `meta`/attrs 里的 `provenance` 明确标记 `synthetic: true/false`，
    以及任务、seed、相机、世界框和 sim 版本，防止合成数据冒充真实示范。
- Acceptance status:
  - [x] 可用的仿真环境；真实示范已产出（reach-v3，60 步，20 次成功判定）。
  - [x] 采集环境与 Jittor 环境隔离（`import mujoco|metaworld|gymnasium` 在 Jittor 环境均为 False，
      `include-system-site-packages = false`）。
  - [x] 新路径写入，绝不覆盖（重复录制被 `refusing to overwrite existing demonstration data` 拒绝）。
  - [x] 数据集支持官方布局，并能在 1024 点、horizon 4 下加载真实数据。
  - [x] 未声称成功率、CUDA 或论文结果。
  - [x] 全量基线不退化（177 passed, 1 skipped）。
- Findings:
  - P1: 仿真栈是**替代实现**（metaworld 3.0/mujoco 3.3 vs 论文的 2.0/mujoco-py），点云不可能逐像素一致，
    裁剪框为重新推导；已写入 D016 与 `env/metaworld_contract.md`。
  - P2: 只落盘训练实际使用的字段（action/state/point_cloud/episode_ends），官方还写 img/depth/full_state，
    属披露的省略项。
  - P2: FPS 在 numpy 里是 O(N·K)，128×128 场景每帧约 4700 点尚可，若提高渲染分辨率需换实现。
  - P3: 成功判定按 `float(info["success"]) > 0.5`，与官方 bool 语义等价但类型不同。
  - P3: `--record` 只写盘、不读回；读回验证必须切回 Jittor 环境跑 `--check-store`。
- Remaining risks:
  - 只验证了 `reach-v3` 一个任务；其余 49 个任务的专家是否都能稳定成功未知。
  - 未验证多 episode 连续录制的内存与耗时；未做帧级重复性检查（同 seed 两次录制是否一致）。
  - 软件渲染（llvmpipe）比 GPU 慢，正式 10 episodes 的耗时未测。
- Prohibited claims:
  - 不得声称与论文使用同一仿真数据或同一分布。
  - 不得声称 1024 点与论文逐点对应。
  - 不得把合成 self-test 数据当作真实示范。
- Recommended next role: verification
- Next agent should read:
  - `scripts/record_demos.py`
  - `env/metaworld_contract.md`
  - `docs/DECISIONS.md`（D015/D016）

## FPJ025-VERIFY-01

- Task: FPJ-025 MetaWorld / MuJoCo / EGL demonstration environment (or a documented substitute)
- Role completed: verification
- Result: passed
- Owner: orchestrator/verification
- Files changed:
  - none（只读；独立脚本在仓库外 `/mnt/f/workspace/2/verify_fpj025.py`）
- Commands run:
  - `wsl.exe -d Ubuntu -- ... -m pytest -q tests/test_dataset.py`
  - `wsl.exe -d Ubuntu -- ... /mnt/f/workspace/2/verify_fpj025.py`
  - 独立探针：`check_expert.py`、`check_world_box.py`（仓库外）
- Exact outputs:
  - `pytest -q tests/test_dataset.py` → `21 passed`
  - `[V1] mujoco/metaworld/gymnasium importable in Jittor env: False, False, False`；`sim venv isolated: True`
  - `[V2] {'geometry_1024': True, 'geometry_deterministic': True, 'dataset_windows': True, 'dataset_points': True, 'rejection_recorded': True}`
  - `[V3] frames=60 points=[60, 1024, 3] state=[60, 9] kept=1`；`check-store status=ok problems=[]`
  - `[Q1..Q5]` 全部按预期失败（见 quality 记录）
  - `FPJ-025 upstream tree unchanged (950 pre-existing entries)`
  - `FPJ-025 verification and adversarial checks: PASS`
- Independent evidence（实现者未提供）:
  - 用**独立脚本**在 sim-env 里跑 `SawyerReachV3Policy`：200 步内成功 160 次，证明脚本专家在本机真的能用。
  - 用**独立脚本**验证相机→世界映射：16384 个像素点中 1173 个落在 z=-0.913 地面、5233 个落在 z=0 桌面，
    与模型几何一致，因此裁剪框不是凭空取的。
  - 直接读取录制出的 npy/PNG：1024×3 float32 且全有限；PNG 480×480、std 56.9（真实渲染，非纯色）。
  - 逐目录检查 Jittor 环境无 `mujoco/metaworld/gymnasium`，上游工作树在验证前后 `git status` 完全一致。
- Acceptance status: 与 implementation 记录一致；另独立确认“专家可用”与“几何正确”两项。
- Findings:
  - P1: 独立复现了实现者关于“官方裁剪框不可移植”的结论（自己跑出来 0/16384，而非采信其说法）。
  - P2: `muJoCo` 现代渲染器深度上限受 `zfar*extent` 影响；换任务时 extent 变化会改变远处点，需在正式实验前固定。
- Remaining risks:
  - 未验证 49 个其它任务的专家稳定性；未验证同 seed 两次录制的逐位一致性。
- Prohibited claims:
  - 不得把“单任务 60 步录制成功”表述为“数据管线已覆盖论文全部任务”。
- Recommended next role: anti_pattern
- Next agent should read:
  - 本记录与 FPJ025-IMPL-01

## FPJ025-AP-01

- Task: FPJ-025 MetaWorld / MuJoCo / EGL demonstration environment (or a documented substitute)
- Role completed: anti_pattern
- Result: passed
- Owner: orchestrator/anti_pattern
- Files changed:
  - none（只读）
- Commands run:
  - `grep -n "fill\|pad\|repeat\|tile" scripts/record_demos.py`（检查是否用填充凑满 1024 点）
  - `grep -n "success" scripts/record_demos.py`
  - 人工审阅 `collect_episodes`、`MetaWorldAdapter.point_cloud`、`write_released_store`
- Exact outputs:
  - 无 `tile/repeat` 补齐逻辑；点数不足时 `farthest_point_sample` 抛
    `cannot sample 1024 points from a cloud of N; widen the crop bounds`
  - 成功判定只有一处：`float(info.get("success", 0.0)) > 0.5`
- Findings:
  - P1: 没有发现“凑点”“伪造成功”“覆盖旧数据”“把合成数据混入真实数据”的作弊路径；
    `provenance.synthetic` 会随数据落盘。
  - P2: 省略了 `img/depth/full_state`，与官方字段集不同（已写入 D016），答辩时不能含糊说“完全一致”。
  - P2: 裁剪框是重新推导的，属于实现者的自由度；如果被质疑，应给出几何验证数字，而不是“看起来对”。
  - P3: `--self-test` 用的是假适配器，若有人误把其输出目录当真数据，仍可能混淆；目前只靠 provenance 标记防护。
  - P3: 单任务验证（reach-v3）；其它任务的专家失败会导致 `collect_episodes` 直接报错，这是刻意的快速失败。
- Remaining risks:
  - 多任务批量录制脚本还不存在；FPJ-026 若直接循环 10 个 episode，需要先确认专家长期稳定性。
- Prohibited claims:
  - 不得声称已覆盖论文的任务集合。
- Recommended next role: quality
- Next agent should read:
  - `scripts/record_demos.py`（`collect_episodes` 与 `write_released_store`）

## FPJ025-QUAL-01

- Task: FPJ-025 MetaWorld / MuJoCo / EGL demonstration environment (or a documented substitute)
- Role completed: quality
- Result: passed
- Owner: orchestrator/quality
- Files changed:
  - none（只读；对抗检查与 verification 同一个外部脚本）
- Commands run:
  - `/home/s/.local/share/flowpolicy/miniforge-py310/bin/python /mnt/f/workspace/2/verify_fpj025.py`
- Exact outputs:
  - `[Q1] missing --output exit=2`（argparse 用法错误，不是崩溃）
  - `[Q2] unknown task exit=1 message='ValueError: not-a-task-v3 is not a V3 environment'`
  - `[Q3] max-steps 3 exit=1`（成功次数不足，未写盘）
  - `[Q4] overwrite attempt exit=1`（`refusing to overwrite existing demonstration data`）
  - `[Q5] 512-point store status=failed exit=1`
  - `FPJ-025 verification and adversarial checks: PASS`
- Counterexamples constructed:
  1. 缺 `--output` 的录制调用；
  2. 不存在的任务名；
  3. 成功步数必然不足的 `--max-steps 3`；
  4. 向已存在目录重复录制；
  5. 手写 512 点的“像官方布局”的假数据。
- Findings:
  - P1: 修正了一个真实缺陷：`--check-store` 原先在点数不符时直接抛异常退出，现在改为返回
    `status: failed` + 具体原因（Q5 就是靠这条用例发现的）。
  - P2: Q3 曾经因为 `--max-steps 20` 先撞上“专家成功次数不足”而没有走到覆盖保护，
    说明用例参数必须保证能走到被测分支；已改为 60 步。
  - P3: 未覆盖磁盘写入中断、zarr 部分写入等真实故障。
- Remaining risks:
  - 512 点假数据的用例证明契约检查有效，但没有测试“点数正确但字段错位”的更隐蔽造假。
- Prohibited claims:
  - 不得把 5 条反例通过表述为“数据管线已可无人值守长跑”。
- Recommended next role: gate
- Next agent should read:
  - 本文件 FPJ025-IMPL-01 至本记录

## FPJ025-GATE-01

- Task: FPJ-025 MetaWorld / MuJoCo / EGL demonstration environment (or a documented substitute)
- Role completed: gate
- Result: passed
- Owner: orchestrator/gate
- Files changed:
  - `.agents/STATE.yaml`
  - `.agents/HANDOFFS.md`
  - `.agents/QUEUE.md`
- Commands run:
  - `wsl.exe -d Ubuntu -- ... scripts/validate_agent_state.py`
  - `wsl.exe -d Ubuntu -- ... -m pytest -q`
  - `wsl.exe -d Ubuntu -- ... /mnt/f/workspace/2/verify_fpj025.py`
- Exact outputs:
  - `Agent state OK: task=FPJ-025 stage=implementation status=in_progress`（claim 时）
  - `pytest -q` → `177 passed, 1 skipped, 8 warnings in 29.80s`
  - `FPJ-025 verification and adversarial checks: PASS`
  - 基线更新：`174 passed, 1 skipped` → `177 passed, 1 skipped`
- 五角色证据链:
  - implementation: FPJ025-IMPL-01
  - verification: FPJ025-VERIFY-01
  - anti_pattern: FPJ025-AP-01
  - quality: FPJ025-QUAL-01
  - gate: 本记录
- Acceptance status: 全部满足（含“要么可用环境、要么诚实阻塞”分支：本次走的是“可用环境 + 披露替代差异”）。
- Findings:
  - P1: 仿真栈是替代实现（D016），所有对外表述必须带“替代仿真栈”限定。
  - P2: 只验证了 reach-v3；其它任务与多 episode 连续录制转交 FPJ-026。
  - P3: 成功判定类型差异（float vs bool）、画质依赖软件渲染。
- Remaining risks:
  - 正式 10 条示范的耗时与稳定性未知；未做同 seed 可复现性检查。
- Prohibited claims:
  - 不得声称数据与论文同分布、不得声称覆盖全部任务、不得声称 GPU 渲染已验证。
- Recommended next role: implementation（FPJ-026）
- Next agent should read:
  - `.agents/STATE.yaml`
  - `.agents/QUEUE.md`
  - `scripts/record_demos.py`
  - `env/metaworld_contract.md`

## FPJ026-IMPL-01

- Task: FPJ-026 Expert demonstrations: one temporary episode, then ten official episodes
- Role completed: implementation
- Result: passed
- Owner: orchestrator/implementation
- Files changed:
  - `scripts/record_demos.py`（新增 `meta/success_steps`、provenance、全局 numpy 播种修复、
    `--check-store` 的契约失败改为“判定”而非崩溃）
  - `docs/DECISIONS.md`（D016 增补：动作是原始专家输出、可超出环境裁剪范围；省略字段披露）
  - `data/reach-v3.zarr`（新增，10 条正式示范）
  - `data/archive/reach-v3.unseeded-archive.zarr`（新增，早期未播种版本，保留作审计）
  - `.agents/STATE.yaml`、`.agents/HANDOFFS.md`
- Commands run:
  - 临时：`MUJOCO_GL=egl .../sim-env/bin/python scripts/record_demos.py --record --task reach-v3 --episodes 1 --max-steps 80 --seed 11 --output /tmp/fpj026_temp.zarr`
  - 正式：`... --episodes 10 --max-steps 120 --seed 0 --output data/reach-v3.zarr`
  - 复现：`... --episodes 2 --max-steps 120 --seed 0 --output /tmp/fpj026_repro.zarr`
  - 校验：`.../miniforge-py310/bin/python scripts/record_demos.py --check-store data/reach-v3.zarr`
  - 独立校验：`/mnt/f/workspace/2/verify_fpj026.py`
- Exact outputs:
  - 临时 episode：`[collect] episode 1/1: kept, 80 steps, 35 successful, attempt 1`
  - 正式：10 条 × 120 步，成功步数 `[70, 77, 87, 81, 67, 78, 85, 73, 63, 76]`，`data/reach-v3.zarr` 8.4 MB
  - `--check-store`：`status ok`，`episodes 10`，`windows 1210`，`point_cloud [2,1024,3]`
  - 复现：seed 0 重录 2 条 → `successes [70, 77]`，与正式前两条**逐位相同**（含动作数组）
  - 数据集视角：`validation mask [0,0,0,0,0,0,0,0,1,0]`，train windows `1089`，
    batch `action (16,4,4)`、`point_cloud (16,2,1024,3)`，normalizer 覆盖 action/agent_pos/point_cloud
  - 全量：`177 passed, 1 skipped`
- Implementation notes:
  - 复现性曾失败：同 seed 两次录制的成功计数不同（85 vs 71）。根因是 metaworld 在**构造 benchmark 时**
    用全局 numpy RNG 采样任务目标，而 `--seed` 只作用于 `reset`。修复：构造 `MT1` 前 `np.random.seed(seed)`；
    实测修复后两次独立进程的 goal 与成功计数完全一致。
  - 早期未播种的 10 条数据已移到 `data/archive/`，不删除，作为“修复前”的审计记录。
  - `meta/success_steps` 把每条 episode 的成功步数写进数据，`--check-store` 会回报，
    避免只在日志里可见。
  - provenance 记录任务、seed、相机、图像尺寸、世界裁剪框与 sim 包版本，`synthetic: false`。
- Acceptance status:
  - [x] 先录 1 条临时 episode 并逐字段检查（80 步、35 次成功、形状与 dtype 全对）。
  - [x] 10 条正式示范写入新路径，未删除/覆盖既有数据（旧数据归档保留）。
  - [x] 每条 episode 的成功步数写入 `meta/success_steps` 并随报告输出。
  - [x] 正式存储通过 `MetaworldZarrDataset`（1024 点、horizon 4）加载，报出 1210 个窗口。
  - [x] 同 seed 重录得到一致的长度、成功计数与逐位相同的动作数组。
  - [x] 存储带 `synthetic: false` provenance。
  - [x] 未声称成功率/CUDA/论文结果。
  - [x] 全量基线不退化（177 passed, 1 skipped）。
- Findings:
  - P1: 发现并修复了“同 seed 不可复现”的真实缺陷（全局 RNG 未播种）。修复前的数据归档保留，可对照。
  - P2: 动作是脚本专家原始输出，1.96% 的条目在第 1 维超出 [-1,1]（最大 1.48），环境内部会裁剪；
    与官方采集脚本行为一致，已写入 D016。
  - P2: 10 条 episode 下验证集只有 1 条（索引 8），因此验证 loss 的方差会很大；
    正式实验若要稳定的验证曲线需要更多 episode 或更大的 val_ratio。
  - P3: 每条 episode 固定 120 步（不是自然终止），因为脚本专家在 reach 上会一直成功；
    这不影响成功判定，但与论文“跑到 done”不完全相同，需在实验文档里说明。
  - P3: 录制耗时约 3.5 分钟/10 条（软件渲染），比 GPU 慢。
- Remaining risks:
  - 只覆盖 `reach-v3` 一个任务；其它任务的专家成功率未验证。
  - 未做 episode 内观测连续性检查（例如点云在两帧之间是否跳变过大）。
- Prohibited claims:
  - 不得声称数据与论文同分布；不得声称已验证其它任务。
- Recommended next role: verification
- Next agent should read:
  - `scripts/record_demos.py`
  - `docs/DECISIONS.md`（D016）

## FPJ026-VERIFY-01

- Task: FPJ-026 Expert demonstrations: one temporary episode, then ten official episodes
- Role completed: verification
- Result: passed
- Owner: orchestrator/verification
- Files changed:
  - none（只读；脚本在仓库外 `/mnt/f/workspace/2/verify_fpj026.py`）
- Commands run:
  - `/home/s/.local/share/flowpolicy/miniforge-py310/bin/python /mnt/f/workspace/2/verify_fpj026.py`
  - `/mnt/f/workspace/2/check_action_range.py data/reach-v3.zarr`
  - `/mnt/f/workspace/2/check_official_dataset.py data/reach-v3.zarr`
- Exact outputs:
  - `[V1] status=ok episodes=10 windows=1210`；`success_steps=[70, 77, 87, 81, 67, 78, 85, 73, 63, 76]`
  - `[V2] action (1200,4) float32 range [-0.565, 1.481]`；`state (1200,9)`；`cloud (1200,1024,3)`
  - `[V2] entries beyond the env clip range: 94/4800 (1.96%), dims=[1]`
  - `[V2] {'action_4d_finite': True, 'state_9d_finite': True, 'cloud_1024_finite': True,
    'ends_monotonic': True, 'ends_consistent': True, 'successes_recorded': True,
    'actions_within_policy_range': True, 'clip_fraction_small': True}`
  - `[V3] repro lengths=[120,120] successes=[70,77] same_lengths=True same_successes=True same_actions=True`
  - `[V4] re-record onto the official store exit=1`
  - `[V5] temp store exists=True repo data=['archive', 'reach-v3.zarr']`
  - `FPJ-026 verification: PASS`
- Independent evidence（实现者未提供）:
  - 直接从 zarr 读数组核对 dtype、范围、`episode_ends` 单调性与 `success_steps` 一致性，
    不依赖录制脚本的自报。
  - 用**另一个进程**重录 2 条并与正式存储前两条做 `np.array_equal`，确认动作逐位相同。
  - 归档存储（修复前数据）仍能被 `--check-store` 正常读取，说明归档不是坏数据。
- Acceptance status: 与 implementation 记录一致。
- Findings:
  - P1: 复现性问题在实现阶段已修复，验证阶段独立确认（同 seed 动作逐位一致）。
  - P2: 1.96% 的动作超出环境裁剪范围，属预期行为，但应在论文对比时说明“训练目标是专家原始输出”。
  - P2: 10 条 episode 只有 1 条进验证集，验证指标噪声大。
- Remaining risks:
  - 未验证其它任务；未做帧间连续性检查。
- Prohibited claims:
  - 不得把“单任务 10 条示范可复现”表述为“整条数据管线对全部任务可复现”。
- Recommended next role: anti_pattern
- Next agent should read:
  - 本记录与 FPJ026-IMPL-01

## FPJ026-AP-01

- Task: FPJ-026 Expert demonstrations: one temporary episode, then ten official episodes
- Role completed: anti_pattern
- Result: passed
- Owner: orchestrator/anti_pattern
- Files changed:
  - none（只读）
- Commands run:
  - `grep -n "shuffle\|permute\|angle\|jitter" scripts/record_demos.py`
  - `grep -n "success_steps" scripts/record_demos.py`
  - 人工审阅 `collect_episodes` 的成功计数与 `Episode.validate`
- Exact outputs:
  - 未发现对动作/状态/点云做随机扰动或增强的代码
  - `success_steps` 只在 `step` 返回 `success` 为真时自增，并原样写入 `meta/success_steps`
- Findings:
  - P1: 没有“为了凑够 10 条而放松成功阈值”“失败 episode 混入成功数据”“数据增强冒充真实采样”的路径。
  - P2: 每条 episode 固定 `max_steps` 截断（不是自然终止），若某任务专家会一直成功，
    截断本身合理；但若专家在某任务上长期不终止，截断会让“episode 长度”失去意义，需在 FPJ-029 前确认。
  - P2: 归档目录 `data/archive/` 与正式目录并列在 `data/` 下，使用者必须显式指定路径；
    这是刻意的（不做隐式“取最新”）。
  - P3: provenance 只在写入时记录一次；若日后手工改数据，标记不会自动失效。
  - P3: 全局 `np.random.seed()` 有副作用（污染调用方 RNG），仅存在于采集脚本，未进入库代码。
- Remaining risks:
  - 未对“点云与状态是否同一时刻”做逐帧交叉检查（同一 `mj_forward` 后取值，理论上一致）。
- Prohibited claims:
  - 不得声称已做数据增强或多样性优化。
- Recommended next role: quality
- Next agent should read:
  - `scripts/record_demos.py`

## FPJ026-QUAL-01

- Task: FPJ-026 Expert demonstrations: one temporary episode, then ten official episodes
- Role completed: quality
- Result: passed
- Owner: orchestrator/quality
- Files changed:
  - none（只读）
- Commands run:
  - `/home/s/.local/share/flowpolicy/miniforge-py310/bin/python /mnt/f/workspace/2/verify_fpj026.py`
- Exact outputs:
  - `[V4] re-record onto the official store exit=1`（拒绝覆盖）
  - `[V5] temp store exists=True repo data=['archive', 'reach-v3.zarr']`（临时数据未被删、目录可信）
  - `FPJ-026 verification: PASS`
- Counterexamples constructed:
  1. 向已存在正式目录重复录制 → 被拒绝；
  2. 临时 episode 在正式录制后仍然存在 → 未被“顺手清理”；
  3. 归档目录中的旧存储仍可被契约校验读取 → 归档未损坏；
  4. 动作越界检查 → 明确量化 1.96% 并确认只在第 1 维。
- Findings:
  - P1: 覆盖保护与“不删除既有数据”两条都有实测证据，不是口头承诺。
  - P2: 动作越界最初被我的校验脚本判为失败；核对 D016 后确认这是与官方一致的原始输出语义，
    已把校验条件改为“在策略范围内且越界比例 < 5%”，避免把正确行为误判为缺陷。
  - P3: 未构造“10 条里混入失败 episode”的反例（失败会被 `collect_episodes` 直接丢弃，路径已被 FPJ-025 覆盖）。
- Remaining risks:
  - 采集脚本对磁盘写入中断没有保护（与 FPJ-025 同一风险）。
- Prohibited claims:
  - 不得把“拒绝覆盖”表述为“数据写入具备事务性”。
- Recommended next role: gate
- Next agent should read:
  - 本文件 FPJ026-IMPL-01 至本记录

## FPJ026-GATE-01

- Task: FPJ-026 Expert demonstrations: one temporary episode, then ten official episodes
- Role completed: gate
- Result: passed
- Owner: orchestrator/gate
- Files changed:
  - `.agents/STATE.yaml`
  - `.agents/HANDOFFS.md`
  - `.agents/QUEUE.md`
- Commands run:
  - `wsl.exe -d Ubuntu -- ... scripts/validate_agent_state.py`
  - `wsl.exe -d Ubuntu -- ... -m pytest -q`
  - `/home/s/.local/share/flowpolicy/miniforge-py310/bin/python /mnt/f/workspace/2/verify_fpj026.py`
- Exact outputs:
  - `Agent state OK: task=FPJ-026 stage=implementation status=in_progress`（claim 时）
  - `pytest -q` → `177 passed, 1 skipped`
  - `FPJ-026 verification: PASS`
  - 基线保持 `177 passed, 1 skipped`（本任务只加数据，不加测试）
- 五角色证据链:
  - implementation: FPJ026-IMPL-01
  - verification: FPJ026-VERIFY-01
  - anti_pattern: FPJ026-AP-01
  - quality: FPJ026-QUAL-01
  - gate: 本记录
- Acceptance status: 全部满足。
- Findings:
  - P1: 同 seed 复现性是本任务发现的真实缺陷并已修复；修复前数据归档保留。
  - P2: 验证集只有 1 条 episode、动作 1.96% 越界、episode 固定 120 步截断，转交 FPJ-027/FPJ-029 说明。
  - P3: 数据目录含归档子目录，使用时必须显式路径。
- Remaining risks:
  - 只有 `reach-v3`；其它任务未采集。
- Prohibited claims:
  - 不得声称数据覆盖论文任务集合或与论文同分布。
- Recommended next role: implementation（FPJ-027）
- Next agent should read:
  - `.agents/STATE.yaml`
  - `.agents/QUEUE.md`
  - `scripts/record_demos.py`
  - `data/reach-v3.zarr`（10 条真实示范）

## FPJ027-IMPL-01

- Task: FPJ-027 Training smoke on real demonstrations plus a 5-episode closed-loop rollout
- Role completed: implementation
- Result: passed
- Owner: orchestrator/implementation
- Files changed:
  - `scripts/train_smoke.py`（新增：真实数据训练、JSONL 日志、checkpoint、覆盖保护）
  - `scripts/sim_server.py`（新增：sim-env 里的 JSON 行仿真服务）
  - `scripts/rollout.py`（新增：Jittor 侧策略 + 跨环境进程桥的闭环 rollout）
  - `src/flowpolicy_jittor/training/trainer.py`（修复：采样时间低于 flow matcher `eps` 导致训练崩溃）
  - `tests/test_trainer.py`（新增采样时间域回归测试）
  - `tests/test_rollout.py`（新增 7 个 rollout/桥/覆盖保护测试）
  - `docs/DECISIONS.md`（D017：进程桥、动作切片显式化、报告口径）
  - `docs/REPRODUCTION_PLAN.md`、`README.md`、`env/metaworld_contract.md`（状态、命令与任务可用性）
  - `outputs/smoke/**`、`outputs/probe/**`（训练证据）、`outputs/rollout/rollout.json`（rollout 证据）
- Commands run:
  - `OMP_NUM_THREADS=4 python scripts/train_smoke.py --data data/reach-v3.zarr --epochs 100 --out outputs/smoke`
  - `MUJOCO_GL=egl python scripts/rollout.py --data data/reach-v3.zarr --checkpoint outputs/smoke/policy.ckpt --episodes 5 --max-steps 200 --out outputs/rollout`
  - `/home/s/.local/share/flowpolicy/miniforge-py310/bin/python /mnt/f/workspace/2/verify_fpj027.py`
  - `python -m pytest -q`
- Exact outputs:
  - 训练：3400 步（100 epochs × 34），loss `1.365e-4 → 2.229e-7`（min），验证 loss `2.20e-5`，
    9052 秒（2.66 秒/步），checkpoint `outputs/smoke/policy.ckpt`
  - rollout：`ep1 111/200 成功步、ep2 0、ep3 172、ep4 163、ep5 0` → 5 条中 3 条有成功步，
    1000 步共 227 秒，单次推理 2.886 ms，NFE=1，`checkpoint_reload_max_abs_delta = 0.0`
  - 全量：`185 passed, 1 skipped`
  - 校验脚本：`FPJ-027 verification: PASS`
- Implementation notes:
  - 训练脚本按 `(seed, step)` 生成噪声与时间，日志逐步行 `{step, lr, loss}`，学习率用
    `warmup_cosine_lr`，`--overwrite` 外默认拒绝复用输出目录。
  - 修了一个真实缺陷：`Trainer._step_inputs` 用 `rng.random()` 采样时间，偶尔落到
    `eps=0.01` 以下，`flow_matching._time_pair` 直接抛 `t must lie in [0.01, 1]`；
    真实数据训练第一次运行就崩在这里。现改为带 margin 的 clip，并加回归测试。
  - 跨环境桥（D017）：`sim_server.py` 在 sim-env 里跑，`rollout.py` 在 Jittor 环境里跑，
    数组用 base64 `.npy` 传输；`step` 响应区分 `done`（环境终止）与 `step_cap`（步数上限）。
  - 动作切片显式写进报告：`action_slice_mode=code` → 动作边界 `[1, 5)`，即每 chunk 3 步
    （沿用 code-faithful 行为，README 有说明）。
  - 训练 loss 与验证 loss 差两个数量级（2e-7 vs 2e-5）是 10 条示范下的预期过拟合，
    脚本 `summary.json` 里明确写了“训练 loss 不蕴含成功率”。
- Acceptance status:
  - [x] 真实数据训练打通，JSONL 记录 step/lr/loss。
  - [x] loss 从首步到最小值单调改善，无非有限值。
  - [x] 两个环境互不泄漏依赖（双向探测均为空）。
  - [x] 5 条 episode 全部跑到 200 步上限，逐条记录成功/失败与成功步数。
  - [x] 动作切片模式显式记录（`code`，边界 `[1,5)`）。
  - [x] 延迟报告含 NFE 与 CPU 后端限定。
  - [x] checkpoint 重载后首个动作差为 0。
  - [x] 未声明论文成功率/CUDA/完整规模训练。
  - [x] 全量基线 `177 → 185 passed, 1 skipped`。
- Findings:
  - P1: 训练时间采样越界是真实缺陷，已在 `Trainer` 修掉并加测试。
  - P2: 10 条示范严重过拟合（train 2e-7 / val 2e-5），闭环成功并不稳定（3/5）。
    FPJ-029 需要更多示范或更强正则，不能拿这 5 条当成功率。
  - P2: `--max-steps 200` 下 5 条 episode 全部“step_cap”结束，说明脚本里 `reach` 任务不会自然终止；
    报告已区分 `step_cap` 与 `env_done`，避免把截断误写成环境结束。
  - P3: rollout 里每 chunk 只执行 3 步（code 模式），与 `n_action_steps=4` 的字段值不同，
    报告同时给出 `n_action_steps` 与 `policy_action_bounds` 以免误解。
  - P3: 完整 100 epochs CPU 训练耗时 2.5 小时，正式实验必须上 GPU 或增加示范。
- Remaining risks:
  - 只有 `reach-v3` 一个任务；`button-press-wall-v3` 尚未采集与训练（专家实测 141/200 可用）。
  - 单 seed、5 条 episode，不构成统计结论。
- Prohibited claims:
  - 不得把 3/5 说成成功率，也不得说“已复现论文指标”。
- Recommended next role: verification
- Next agent should read:
  - `scripts/rollout.py`、`scripts/sim_server.py`、`scripts/train_smoke.py`
  - `docs/DECISIONS.md`（D017）

## FPJ027-VERIFY-01

- Task: FPJ-027 Training smoke on real demonstrations plus a 5-episode closed-loop rollout
- Role completed: verification
- Result: passed
- Owner: orchestrator/verification
- Files changed:
  - none（只读；脚本在仓库外 `/mnt/f/workspace/2/verify_fpj027.py`）
- Commands run:
  - `/home/s/.local/share/flowpolicy/miniforge-py310/bin/python /mnt/f/workspace/2/verify_fpj027.py`
  - `python -m pytest -q`
- Exact outputs:
  - `[V1] epochs=100 steps=3400/3400 first=1.365e-04 min=2.229e-07 val=2.2033087589079514e-05`
  - `[V2] schedule mismatches=0 peak_lr=0.001000 final_lr=0.000050`
  - `[V3] episodes=5 successes=3 steps=1000 nfe=1 slice=code bounds=[1, 5]`，
    逐条 `111/0/172/163/0`，`policy latency mean 2.886 ms`
  - `[V4] jittor env sim packages=[] sim env jittor/torch=[]`
  - `[V5] train overwrite exit=1 rollout overwrite exit=1`
  - `[Q1] mismatched-config rollout exit=1`、`[Q2] missing sim python exit=1`、`[Q3] zero episodes exit=1`
  - `FPJ-027 verification: PASS`
  - `pytest -q` → `185 passed, 1 skipped`
- Independent evidence（实现者未提供）:
  - 用 `warmup_cosine_lr` 独立重算日志里的学习率（首尾各 40 步），0 处不匹配。
  - 双向探测环境隔离：Jittor 环境无法 import `mujoco/metaworld/gymnasium`，
    sim-env 无法 import `jittor/torch`。
  - 逐条核对 `success` 布尔与 `success_steps` 计数一致，`steps ≥ 1`。
  - 校准 `checkpoint_reload_max_abs_delta`：同一噪声下重载前后动作完全相同（0.0）。
- Acceptance status: 与 implementation 记录一致。
- Findings:
  - P1: 学习率曲线与调度公式逐点吻合，说明日志不是事后编造。
  - P2: 5 条 episode 全为 `step_cap` 结束；若后续用自然终止任务（如 `button-press-wall-v3`），
    需要确认 `env_done` 分支真的被覆盖到。
  - P2: 训练 loss 与验证 loss 相差两个数量级，验证阶段独立确认这是小数据过拟合而非 bug。
- Remaining risks:
  - 未验证多任务、多 seed；未验证 GPU。
- Prohibited claims:
  - 不得把 `[V3] successes=3` 表述为 60% 成功率。
- Recommended next role: anti_pattern
- Next agent should read:
  - 本记录与 FPJ027-IMPL-01

## FPJ027-AP-01

- Task: FPJ-027 Training smoke on real demonstrations plus a 5-episode closed-loop rollout
- Role completed: anti_pattern
- Result: passed
- Owner: orchestrator/anti_pattern
- Files changed:
  - none（只读）
- Commands run:
  - `grep -n "success" scripts/rollout.py`
  - `grep -n "episode\\\\|reward" scripts/sim_server.py`
  - 人工审阅 `rollout_episode` 的成功累加与 `done_reason` 逻辑
- Exact outputs:
  - 成功只来自 `bridge.step()` 返回的 `success`（仿真侧 `float(info["success"]) > 0.5`），
    没有任何“按奖励猜测成功”或“按剩余距离判定成功”的替代路径
- Findings:
  - P1: 没有“把失败 episode 丢掉只报成功”的路径：`episodes` 列表包含全部 5 条，
    `success_episodes` 由列表求和得出。
  - P2: 失败 episode（0 成功步）没有被特判或重跑；这正是要求的行为，但也意味着
    smoke 结果依赖 seed，答辩时不能挑好看的 seed。
  - P2: 训练脚本写出 `prohibited_claims` 字段，属于自我约束而非机制约束；
    真正的防护是 `summary.json` 里 loss 与 `val_windows` 同时可见。
  - P3: `sim_server` 用 `adapter.reset(index)`，index 同时决定任务变体与 reset 种子；
    这让“重放同一 index”可复现，但也意味着 episode 之间不独立同分布（同一任务不同变体）。
  - P3: 桥接层没有超时：若 sim-env 进程卡死，`readline()` 会一直阻塞。
- Remaining risks:
  - 桥接层缺少读超时与心跳；长时间批量 rollout 有挂死风险（已列为后续改进）。
- Prohibited claims:
  - 不得声称评测具备统计显著性。
  - 不得声称进程桥等价于同进程评测（多了一次序列化，延迟口径不同）。
- Recommended next role: quality
- Next agent should read:
  - `scripts/rollout.py`（`rollout_episode`）、`scripts/sim_server.py`

## FPJ027-QUAL-01

- Task: FPJ-027 Training smoke on real demonstrations plus a 5-episode closed-loop rollout
- Role completed: quality
- Result: passed
- Owner: orchestrator/quality
- Files changed:
  - none（只读）
- Commands run:
  - `/home/s/.local/share/flowpolicy/miniforge-py310/bin/python /mnt/f/workspace/2/verify_fpj027.py`
  - `python -m pytest -q tests/test_rollout.py`
- Exact outputs:
  - `[V5] train overwrite exit=1 rollout overwrite exit=1`
  - `[Q1] mismatched-config rollout exit=1`
  - `[Q2] missing sim python exit=1`
  - `[Q3] zero episodes exit=1`
  - `tests/test_rollout.py` 全绿（含成功计数、done 分支、噪声可复现、base64 往返、覆盖保护、collate）
- Counterexamples constructed:
  1. 配置不匹配的 checkpoint（`--down-dims 16 32`）→ 被配置指纹拒绝，不是静默加载；
  2. 不存在的 sim 解释器 → 明确报错；
  3. `--episodes 0` → 直接拒绝，不产出空报告；
  4. 向已有结果目录重复训练/rollout → 拒绝覆盖；
  5. 环境自然终止分支 → 用假桥单测覆盖（`done_at=3` 时停在 3 步且记为 `env_done`）。
- Findings:
  - P1: 反例 1 最初确实能跑通并静默使用错误通道数？——不是：配置指纹先拦住了，
    说明 FPJ-021 的 checkpoint 指纹在闭环路径上同样生效。
  - P2: `--episodes 0` 与 `--max-steps 0` 原先会产出空/无意义报告，现已加显式校验。
  - P3: 未构造“sim 进程中途崩溃”的反例（需要注入故障，成本高，且桥接层没有超时机制，已知风险）。
- Remaining risks:
  - 桥接层无读超时；sim 崩溃时会阻塞而不是快速失败。
- Prohibited claims:
  - 不得把“反例均被拒绝”表述为“评测脚本达到生产级健壮性”。
- Recommended next role: gate
- Next agent should read:
  - 本文件 FPJ027-IMPL-01 至本记录

## FPJ027-GATE-01

- Task: FPJ-027 Training smoke on real demonstrations plus a 5-episode closed-loop rollout
- Role completed: gate
- Result: passed
- Owner: orchestrator/gate
- Files changed:
  - `.agents/STATE.yaml`
  - `.agents/HANDOFFS.md`
  - `.agents/QUEUE.md`
- Commands run:
  - `python scripts/validate_agent_state.py`
  - `python -m pytest -q`
  - `/home/s/.local/share/flowpolicy/miniforge-py310/bin/python /mnt/f/workspace/2/verify_fpj027.py`
- Exact outputs:
  - `Agent state OK: task=FPJ-027 stage=implementation status=in_progress`（claim 时）
  - `pytest -q` → `185 passed, 1 skipped`
  - `FPJ-027 verification: PASS`
  - 基线更新：`177 passed, 1 skipped` → `185 passed, 1 skipped`
- 五角色证据链:
  - implementation: FPJ027-IMPL-01
  - verification: FPJ027-VERIFY-01
  - anti_pattern: FPJ027-AP-01
  - quality: FPJ027-QUAL-01
  - gate: 本记录
- Acceptance status: 全部满足。
- Findings:
  - P1: 真实数据第一次训练就暴露 `Trainer` 的时间采样越界缺陷，已修复并加回归测试。
  - P2: 3/5 的闭环结果只证明管线可用；train/val loss 差两个数量级说明严重过拟合，
    正式实验需要更多示范或更强正则（转 FPJ-029）。
  - P2: 桥接层缺少读超时，转 FPJ-030 之后的可靠性工作。
  - P3: CPU 上 100 epochs 训练 2.5 小时，正式实验必须上 GPU。
- Remaining risks:
  - 单任务、单 seed、5 条 episode；`button-press-wall-v3` 尚未采集。
- Prohibited claims:
  - 不得把 3/5 说成成功率；不得声称复现论文指标；不得声称 GPU 已验证。
- Recommended next role: implementation（FPJ-028）
- Next agent should read:
  - `.agents/STATE.yaml`
  - `.agents/QUEUE.md`
  - `outputs/smoke/summary.json`、`outputs/rollout/rollout.json`

## FPJ028-IMPL-01

- Task: FPJ-028 CUDA kernel smoke, full channel configuration, PyTorch parity and the EMA model
- Role completed: implementation
- Result: passed
- Owner: orchestrator/implementation
- Files changed:
  - `src/flowpolicy_jittor/training/ema.py`（新增）
  - `src/flowpolicy_jittor/training/trainer.py`
  - `src/flowpolicy_jittor/training/__init__.py`
  - `scripts/cuda_smoke.py`（新增）
  - `scripts/parity_check.py`（新增）
  - `scripts/parity_torch_model.py`（新增）
  - `scripts/train_smoke.py`
  - `tests/test_ema.py`、`tests/test_full_channels.py`、`tests/test_parity_bundle.py`（新增）
  - `docs/DECISIONS.md`（D018）、`README.md`
- Commands run:
  - `python scripts/cuda_smoke.py --out outputs/cuda`
  - `python scripts/parity_check.py --out outputs/parity`
  - `python scripts/train_smoke.py --data data/reach-v3.zarr --epochs 2 --batch-size 32 --out outputs/ema_smoke --ema-decay 0.999 --ema-warmup-steps 5`
  - `python -m pytest -q`
- Exact outputs:
  - CUDA：`available=False stage=set_use_cuda error=... Check failed: value==0  No CUDA found.`
  - 全通道：`params=255138308 batch=2 points=1024 s/step=3.71 peak=4460MB action=[2, 3, 4] finite=True`
  - parity：`action_pred_max_abs=9.537e-07 loss_max_abs=7.276e-10 grad_max_abs=1.644e-09 checked=170 failed=0`
  - EMA 训练：68 steps、loss `1.365e-4` → min `4.863e-5`、val `5.452e-5`、182 s、`trainer.ckpt` 写出
  - `pytest -q` → `203 passed, 1 skipped`
- Implementation notes:
  - `EmaModel` 只平均模型权重，排除 `normalizer` 与 `var_est`；衰减先按 `min(decay,(1+step)/(10+step))` 爬升，
    再在 `warmup_steps` 内过渡到 `decay`；`store/restore` 保证评测时在线权重可原位还原。
  - `Trainer(ema_decay=...)` 每步 `ema.update(policy)`，`train_step` 记录里带 `ema_decay`；
    trainer checkpoint 的 `ema` 段不再是 `null`，恢复时两侧不一致直接报错。
  - `train_smoke.py` 新增 `--ema-decay/--ema-warmup-steps`，启用 EMA 时额外写出 `trainer.ckpt`
    （否则 EMA 权重会随 policy checkpoint 一起丢失）。
  - `scripts/parity_torch_model.py` 是从模块语义独立重写的 PyTorch 镜像（不是翻译），
    在独立的 `parity-env`（torch 2.14.1+cpu）里运行，按同名权重 `strict` 加载。
- Remaining risks:
  - EMA 权重未作为 `v_r` 的目标（`target_mode="code"`，与 released 配置一致），差异写进 D018；
  - CUDA 未验证；全通道只跑一步前反向，不是训练。
- Prohibited claims:
  - 不得声称 CUDA 已验证、不得把一步前反向称为“全通道训练”、不得声称 paper 指标。
- Recommended next role: verification
- Next agent should read:
  - 本记录、`outputs/cuda/report.json`、`outputs/parity/parity.json`、`outputs/ema_smoke/summary.json`

## FPJ028-VERIFY-01

- Task: FPJ-028 CUDA kernel smoke, full channel configuration, PyTorch parity and the EMA model
- Role completed: verification
- Result: passed
- Owner: orchestrator/verification
- Files changed: 无（只读复核 + 外部脚本 `/mnt/f/workspace/2/verify_fpj028.py`）
- Commands run:
  - `/home/s/.local/share/flowpolicy/miniforge-py310/bin/python /mnt/f/workspace/2/verify_fpj028.py`
  - `python scripts/cuda_smoke.py --out outputs/cuda --overwrite`（复核可重跑）
  - `python -m pytest -q`
- Exact outputs:
  - `FAILURES: none`（30 项检查）
  - 关键断言：CUDA 未声明可用且原始报错含 `No CUDA found`；全通道 `255,138,308` 参数且 loss 有限；
    parity 170 项检查全过、最差梯度偏差 `1.644e-09`；EMA 训练记录 `ema_decay` 从 `0.1` 单调升到 `0.999`；
    重新加载 `trainer.ckpt` 后 EMA 步数 68、120 个张量与在线权重不同；
    **没有** EMA 的 trainer 加载带 EMA 的 checkpoint 会抛 `trainer checkpoint carries an EMA state ...`
  - `pytest -q` → `203 passed, 1 skipped`
- Findings:
  - P2: 首次复核发现 EMA 名字过滤用前缀匹配，导致 `model.var_est.*` 被平均（见 AP 记录的 P1），
    修复后重新训练 `outputs/ema_smoke`（旧 checkpoint 的 EMA 名字集合已失效）。
  - P3: verification 脚本本身也有 bug（`_verdict` 键名、`min_lr_ratio`、`dataset.normalizer`），
    均已修正后复跑；这些是脚本问题，不是产品缺陷。
- Remaining risks:
  - 复核依赖 `outputs/` 里的既有证据；若删除产物，需要重跑三条命令才能复现结论。
- Prohibited claims:
  - 不得把“复核脚本全绿”当成“论文指标复现”。
- Recommended next role: anti_pattern

## FPJ028-AP-01

- Task: FPJ-028 CUDA kernel smoke, full channel configuration, PyTorch parity and the EMA model
- Role completed: anti_pattern
- Result: passed（并发现 2 个真实缺陷）
- Owner: orchestrator/anti_pattern
- Files changed:
  - `src/flowpolicy_jittor/training/ema.py`（过滤改为按路径段匹配）
  - `scripts/parity_check.py`（容差与判定重写）
  - `tests/test_ema.py`、`tests/test_parity_bundle.py`（对应回归测试）
- Commands run:
  - `/home/s/.local/share/flowpolicy/miniforge-py310/bin/python /mnt/f/workspace/2/antipattern_fpj028.py`
- 反例与结果:
  1. 篡改 bundle 里 `model.final_conv.1.weight`（+1e-3）→ 必须被判失败：现在 `failed` 含
     `action_pred` 与该张量相关梯度项，**旧容差下会漏判**。
  2. 篡改 `action` 输入（+0.05）→ loss 变化 `2.0e-6`（相对 1.2%），旧容差判过、新容差判失败。
  3. `--ema-decay` ∈ {0, 1, -0.5, 1.5}、`--ema-warmup-steps -1` → 全部显式拒绝。
  4. 把 bundle 的 `down_dims` 改成 `[16,32,128]` → PyTorch 镜像加载时报 shape mismatch 并退出非零。
  5. `cuda_smoke.py --out` 指向非空目录 → 拒绝覆盖。
- Findings:
  - P1: `EmaModel` 用 `name.startswith(("normalizer","var_est"))` 过滤，而 policy 前缀是 `model.`，
    所以冻结的 `var_est` 一直被平均进影子权重（数值无害但语义错误）。改为按点分段匹配，并加测试掩码。
  - P1: parity 判定用 `atol + rtol*max(scale, 1.0)`，对小张量等于只比 `atol=2e-4`；
    一个 24% 相对变化的梯度被判为通过。改为 `atol + rtol*scale`，默认 `atol=1e-7, rtol=1e-4`，
    实测最差项只用掉 1.56% 容差预算。
  - P2: `--overwrite` 重训会向旧 `train.jsonl` 追加（同一输出里出现两段 step 1..68），已改为覆盖前清空。
- Remaining risks:
  - 反例集中在 EMA/parity 两个新面；仿真与数据契约未被本轮反例覆盖。
- Prohibited claims:
  - 不得把“反例均被拒绝”表述为“parity 对所有模型规模都成立”（默认容差是按当前小模型标定的）。
- Recommended next role: quality

## FPJ028-QUAL-01

- Task: FPJ-028 CUDA kernel smoke, full channel configuration, PyTorch parity and the EMA model
- Role completed: quality
- Result: passed
- Owner: orchestrator/quality
- Files reviewed:
  - `src/flowpolicy_jittor/training/ema.py`、`trainer.py`、`training/__init__.py`
  - `scripts/cuda_smoke.py`、`parity_check.py`、`parity_torch_model.py`、`train_smoke.py`
  - `tests/test_ema.py`、`tests/test_full_channels.py`、`tests/test_parity_bundle.py`
  - `docs/DECISIONS.md`（D018）、`README.md`
- Checks:
  - `grep -rn 'TODO|FIXME|XXX|breakpoint()'` → 无命中；
  - `python -m compileall -q src scripts tests` → `compile-ok`；
  - 新增测试 18 个（9 EMA + 4 全通道/CUDA + 5 parity 夹具），全部为行为断言，无“只断言不抛异常”的空测试；
  - `outputs/` 产物齐全：`cuda/report.json`、`parity/{bundle.npz,parity.json,torch_report.json}`、
    `ema_smoke/{policy.ckpt,trainer.ckpt,summary.json,epochs.json,train.jsonl}`；
  - 文档与代码同步：D018 记录 CUDA 结论、parity 口径与 EMA 未作目标的原因；README 状态清单已更新。
- Findings:
  - P2: `parity_torch_model.py` 里 `load_mirror` 用 `strict=False` + 显式 missing/unexpected 校验，
    等价于严格加载但错误信息更可读；注释已说明。
  - P3: `cuda_smoke.py` 的 `--skip-full-channels` 仅用于快速回归，默认路径仍跑全通道。
- Remaining risks:
  - `parity-env` 不在仓库里，重建需要联网安装 torch（CPU 版）；已在 D018/README 说明。
- Prohibited claims:
  - 不得把 `outputs/ema_smoke` 的两轮训练描述为“论文训练配方”。
- Recommended next role: gate

## FPJ028-GATE-01

- Task: FPJ-028 CUDA kernel smoke, full channel configuration, PyTorch parity and the EMA model
- Role completed: gate
- Result: passed
- Owner: orchestrator/gate
- Files changed:
  - `.agents/STATE.yaml`、`.agents/HANDOFFS.md`、`.agents/QUEUE.md`
- Commands run:
  - `python scripts/validate_agent_state.py`
  - `python -m pytest -q`
  - `/home/s/.local/share/flowpolicy/miniforge-py310/bin/python /mnt/f/workspace/2/verify_fpj028.py`
- Exact outputs:
  - `pytest -q` → `203 passed, 1 skipped`
  - `FAILURES: none`（30 项独立复核）
  - 基线更新：`185 passed, 1 skipped` → `203 passed, 1 skipped`
- Acceptance status:
  - CUDA 可测不假设：满足（探测 + 原始报错留证，`cuda_verified=false`）；
  - 全通道配置跑通一步真实前反向并记录参数数：满足（255,138,308）；
  - parity 脚本比较输出与梯度并记录最大绝对/相对偏差：满足（170 项、168 梯度）；
  - EMA helper 实现且有 save/restore/target 覆盖：满足（并记录“未作 loss 目标”的决定）；
  - checkpoint EMA 显式化：满足（有则存、无则 `null`，两侧不一致报错）；
  - 不从 CPU 运行得出 CUDA 结论、不用单标量宣称 parity：满足；
  - 测试基线不退化：满足（185 → 203）。
- 五角色证据链:
  - implementation: FPJ028-IMPL-01
  - verification: FPJ028-VERIFY-01
  - anti_pattern: FPJ028-AP-01
  - quality: FPJ028-QUAL-01
  - gate: 本记录
- Findings:
  - P1: `var_est` 前缀过滤缺陷与 parity 容差过宽都是反例角色抓到的，说明“自证”不够，必须留反例；
  - P2: EMA 仍未接入 `target_mode="detached"`，是本任务明确保留的边界（D018）；
  - P3: 全通道只验证“能跑”，显存/时间数据来自 CPU，不能外推 GPU。
- Remaining risks:
  - 无 GPU；无多任务多样性；`test_conv1d_parity.py` 在主环境仍因缺 torch 跳过。
- Prohibited claims:
  - 不得声称 CUDA 已验证、parity 覆盖所有规模、或 EMA 已作为训练目标使用。
- Recommended next role: implementation（FPJ-029）
- Next agent should read:
  - `.agents/STATE.yaml`、`.agents/QUEUE.md`、`docs/DECISIONS.md`（D018）

## FPJ029-IMPL-01

- Task: FPJ-029 two tasks, three seeds, twenty episodes per seed
- Role completed: implementation
- Result: passed
- Owner: orchestrator/implementation
- Files changed:
  - `scripts/experiments.py`（新增，含并发训练 / 串行 rollout）
  - `tests/test_experiments.py`（新增 5 个测试）
  - `data/button-press-wall-v3.zarr`（新增，20 条 × 120 步）
  - `docs/REPRODUCTION_PLAN.md`、`docs/DEFENSE_GUIDE.md`、`README.md`
- Commands run:
  - `MUJOCO_GL=egl python scripts/record_demos.py --record --task button-press-wall-v3 --episodes 20 --max-steps 120 --seed 0 --output data/button-press-wall-v3.zarr`
  - `python scripts/experiments.py --tasks reach-v3 button-press-wall-v3 --seeds 0 1 2 --episodes 20 --max-steps 200 --epochs 10 --batch-size 32 --parallel 3 --threads 4 --out outputs/experiments`
  - `python -m pytest -q`
- Exact outputs:
  - 示范：20 条成功 episode，成功步数 51–62，`windows: 2420`
  - `button-press-wall-v3`：12/20、18/20、20/20 → **83.33% ± 20.82%**，latency 2.97 ms
  - `reach-v3`：2/20、3/20、2/20 → **11.67% ± 2.89%**，latency 2.93 ms
  - 6 个 cell 全部 `step_cap` 结束、NFE=1、reload delta 0
  - `pytest -q` → `208 passed, 1 skipped`
- Notes:
  - 训练单元格并发（`--parallel 3`，每进程 4 线程），rollout 串行，避免 CPU 争用污染延迟；
  - 每个 cell 的 `train.log` / `rollout.log` 记录了完整命令行，可单独重跑；
  - 10 epochs、小通道（32/64）是明确标注的低预算设置，不是论文配方。
- Remaining risks:
  - `reach-v3` 只有 10 条示范、`button-press-wall-v3` 20 条；两者难度不同，跨任务比较没有意义；
  - 成功率对训练预算非常敏感，本网格只能说明管线与评测是可信的。
- Prohibited claims:
  - 不得把这组数字写成论文成功率；不得只引用 100% 的那个 seed。
- Recommended next role: verification

## FPJ029-VERIFY-01

- Task: FPJ-029 two tasks, three seeds, twenty episodes per seed
- Role completed: verification
- Result: passed
- Owner: orchestrator/verification
- Commands run:
  - `/home/s/.local/share/flowpolicy/miniforge-py310/bin/python /mnt/f/workspace/2/verify_fpj029.py`
- Exact outputs:
  - `FAILURES: none`（40+ 项检查）
  - 从 6 份原始 `rollout.json` 重算：每 cell 20 条 episode、成功数与报告一致、均值/标准差与
    `experiments.json` 逐位一致（`0.116667 vs 0.116667`、`0.028868 vs 0.028868`）；
  - `SUMMARY.md` 里的百分比必须等于重算值（`83.33% ± 20.82%`、`11.67% ± 2.89%`）；
  - 每个 cell `reused=false`（没有偷偷复用旧 checkpoint）、`wall_seconds > 0`、训练 loss 有下降；
  - `done_reasons` 覆盖全部 60 条 episode；延迟 2.93–2.97 ms。
- Findings:
  - P2: 两个任务的难度差异巨大（83% vs 12%），若只报告前者会严重误导；
    `SUMMARY.md` 同时给出逐 seed 明细与“这不代表什么”段落。
- Remaining risks:
  - 复核只读产物，删除 `outputs/experiments/` 后需重跑 45–60 分钟网格才能复现。
- Prohibited claims:
  - 不得把“复核全绿”说成“结果达到论文水平”。
- Recommended next role: anti_pattern

## FPJ029-AP-01

- Task: FPJ-029 two tasks, three seeds, twenty episodes per seed
- Role completed: anti_pattern
- Result: passed
- Owner: orchestrator/anti_pattern
- 反例与结果:
  1. 缺示范的任务名 → `missing demonstration stores` 直接拒绝，不会产出空报告；
  2. 重复 seed（`--seeds 0 0`）→ 拒绝；
  3. 已有 `experiments.json` 且未加 `--overwrite`/`--reuse` → 拒绝覆盖；
  4. 单元格失败（模拟 `train_failed`）→ 不进入成功率统计但保留在报告里（`cells_failed`>0）；
  5. 单 seed 时标准差必须是 `None` 而不是 `0`（否则会假装“零方差”）。
- Findings:
  - P2: 最初的 `_markdown` 测试写成 `1/2` 期望，实际是 `2/2`（两者都是 ok），
    反映“分母容易写错”的风险，已在测试里固定为真实语义。
  - P3: 未构造“rollout 中途崩溃”的反例；桥接层仍无读超时（FPJ-027 已记录）。
- Remaining risks:
  - 网格只覆盖 2 个任务；第三个任务（`reach-wall-v3`）尚未录制示范。
- Prohibited claims:
  - 不得把“反例均被拒绝”表述为“评测对所有任务都稳健”。
- Recommended next role: quality

## FPJ029-QUAL-01

- Task: FPJ-029 two tasks, three seeds, twenty episodes per seed
- Role completed: quality
- Result: passed
- Owner: orchestrator/quality
- Checks:
  - `scripts/experiments.py` 单文件 300 行内，函数职责清晰（train_cell / rollout_cell / _aggregate / _markdown）；
  - 5 个单测覆盖聚合、失败可见性、markdown 口径、三条守卫、报告必带 `prohibited_claims`；
  - `outputs/experiments/` 每个 cell 有 `train/`、`rollout/`、`train.log`、`rollout.log`，可单独重跑；
  - README 与 DEFENSE_GUIDE 的数字与 `SUMMARY.md` 一致，并显式标注“低预算 CPU、非论文指标”。
- Findings:
  - P3: `--parallel` 只作用于训练，rollout 串行；文档已说明原因（延迟测量）。
- Remaining risks:
  - 网格训练 10 epochs，未做“训练预算 vs 成功率”的曲线，无法说明是否收敛。
- Recommended next role: gate

## FPJ029-GATE-01

- Task: FPJ-029 two tasks, three seeds, twenty episodes per seed
- Role completed: gate
- Result: passed
- Owner: orchestrator/gate
- Commands run:
  - `python scripts/validate_agent_state.py`
  - `python -m pytest -q`
  - `/home/s/.local/share/flowpolicy/miniforge-py310/bin/python /mnt/f/workspace/2/verify_fpj029.py`
- Exact outputs:
  - `python -m pytest -q` → `208 passed, 1 skipped`
  - `FAILURES: none`
  - 基线更新：`203 → 208 passed, 1 skipped`
- Acceptance status:
  - 两个任务且第二个任务有示范：满足（`button-press-wall-v3`，20 条）；
  - 3 seeds × 20 episodes 并报均值±标准差：满足；
  - 记录延迟/NFE/step cap/失败原因/checkpoint：满足；
  - 不宣称论文成功率、CUDA 仍未验证：满足；
  - 测试基线不退化：满足（203 → 208）。
- 五角色证据链: FPJ029-IMPL-01 / FPJ029-VERIFY-01 / FPJ029-AP-01 / FPJ029-QUAL-01 / 本记录
- Findings:
  - P1: 两任务成功率相差 7 倍，最有价值的信息是“管线与评测可信”，而不是任何一个数字；
  - P2: 训练预算只有 10 epochs，成功率不能外推到长训练；
  - P3: 第三个任务的示范尚未采集。
- Remaining risks:
  - 无 GPU；无训练预算消融；第三个任务缺数据。
- Prohibited claims:
  - 不得写成论文指标或“复现成功”；不得只报最好的 seed。
- Recommended next role: implementation（FPJ-031）
- Next agent should read: `.agents/STATE.yaml`、`outputs/experiments/SUMMARY.md`、`docs/DECISIONS.md`（D014）

## FPJ030-GATE-01

- Task: FPJ-030 the unreproduced `test_policy` intermittent failure
- Role completed: gate
- Result: passed（原始失败仍未复现，未声称定位根因）
- Owner: orchestrator/gate
- Files changed:
  - `tests/test_policy.py`
  - `.agents/QUEUE.md`、`.agents/STATE.yaml`、`.agents/HANDOFFS.md`
- Commands run:
  - 单测重复 20 次：`pytest -q tests/test_policy.py::test_top_level_adamw_updates_active_but_not_frozen_parameters`
  - 整文件重复 3 次：`pytest -q tests/test_policy.py`
- Exact outputs:
  - `repeat summary: pass=20 fail=0`
  - `file run 1 ok`、`file run 2 ok`、`file run 3 ok`
- Changes:
  - 显式 `jt.set_global_seed(0)`（消除初始化随机性这一可能原因）；
  - 断言从“任一参数变化”改为“最大变化量 > 1e-6”，失败时打印 loss、参数数与最大变化量。
- Findings:
  - P2: 仍然无法复现原始失败，因此只声称“加固 + 20 次通过”，不声称“根因已定位”。
- Prohibited claims:
  - 「该测试历史上失败的根因已找到」——未找到。
- Recommended next role: implementation（FPJ-031）

## FPJ031-IMPL-01

- Task: FPJ-031 CPU float nondeterminism
- Role completed: implementation
- Result: passed
- Owner: orchestrator/implementation
- Files changed:
  - `scripts/determinism_probe.py`（新增）
  - `tests/test_determinism.py`（新增 5 个测试）
  - `tests/test_trainer.py`（续训容差 2e-3 → 1e-2，docstring 写实测依据）
  - `docs/DECISIONS.md`（D014 追加 FPJ-031 复测表）
- Commands run:
  - `OMP_NUM_THREADS={1,4} [MKL_CBWR=COMPATIBLE] python scripts/determinism_probe.py --mode single|resume`
  - `python scripts/determinism_probe.py --compare a.npz b.npz`
  - `python -m pytest -q tests/test_determinism.py tests/test_trainer.py`
- Exact outputs:
  - 4 组设置下 `bit_identical=True max_abs_delta=0.000e+00`（单次 6 步 vs 3+恢复+3，独立进程）；
  - 同进程两次 6 步：`max_delta=0.000e+00`；
  - 同进程“已训练后再恢复续训”：`max_delta=5.492e-03`、`loss_delta=9.532e-09`，`clean_graph`/复用对象均无效；
  - `OMP_NUM_THREADS=1` 与 `4` 的 final hash 不同（`ff08f597d6d6bcd2` vs `d66e83e8b9623dfe`）。
- Notes:
  - 探针第一版有对齐缺陷（resume 模式的快照索引从 0 重开，导致“step 1 对 step 4”并伪造一致），
    已改为保留保存前快照 + 校验两边覆盖的 step 集合；因此**不声称**长进程内 bit-exact 续训。
- Prohibited claims:
  - 「续训完全一致」；「CPU 可复现 bit-exact 训练」。
- Recommended next role: verification

## FPJ031-VERIFY-01

- Task: FPJ-031 CPU float nondeterminism
- Role completed: verification
- Result: passed
- Owner: orchestrator/verification
- Commands run:
  - `python -m pytest -q tests/test_determinism.py`（5 passed）
  - `python /mnt/f/workspace/2/inproc_determinism.py`、`python /mnt/f/workspace/2/variant_determinism.py`
- Exact outputs:
  - `run1 vs run2 (both 6 steps, same process): (0.0, None)`
  - `run1 vs 3+resume (same process): (0.0054915957152843475, 'obs_encoder.extractor.mlp.6.weight')`
  - 三种恢复变体（新对象 / `clean_graph` / 同对象重载）漂移同为 `5.492e-03`。
- Findings:
  - P2: 结论依赖“进程是否已训练过”，文档必须写清场景；
  - P3: `MKL_CBWR` 在本机对这组算子没有可测影响。
- Prohibited claims:
  - 不得把“独立进程一致”推广成“任意场景一致”。
- Recommended next role: anti_pattern

## FPJ031-AP-01

- Task: FPJ-031 CPU float nondeterminism
- Role completed: anti_pattern
- Result: passed
- Owner: orchestrator/anti_pattern
- 反例与结果:
  1. 篡改恢复后第 4 步的某个权重 `+1e-6` → `compare` 在 step 4 报 `all_bit_identical=False`；
  2. 用 5 步与 3 步两次运行对比 → `SystemExit: did not cover the same steps`；
  3. 快照索引统计写成“键里含 `:` 计数”导致“7 个快照”错误 → 改为按 step 索引集合比较；
  4. `max_abs_delta == approx(1e-6)` 在 float32 下实际是 `1.013e-06` → 改成区间断言。
- Findings:
  - P1: 探针自身的对齐 bug 会产出“全绿”假证据，这是最危险的失败模式，已用两个测试锁住。
- Prohibited claims:
  - 「第一版探针的 bit-exact 结果可信」——第一版结果作废。
- Recommended next role: quality

## FPJ031-QUAL-01

- Task: FPJ-031 CPU float nondeterminism
- Role completed: quality
- Result: passed
- Owner: orchestrator/quality
- Checks:
  - `scripts/determinism_probe.py` 提供 `--mode` 与 `--compare` 两条路径、JSON 报告可直接引用；
  - 5 个测试覆盖容差内续训、同进程可重复、篡改可检出、覆盖不一致拒绝、meta 记录运行设置；
  - D014 给出 4 行实测表；测试基线 `208 → 213 passed, 1 skipped`。
- Remaining risks:
  - 只测到 6 步；更长轨迹漂移曲线未做。
- Recommended next role: gate

## FPJ031-GATE-01

- Task: FPJ-031 CPU float nondeterminism
- Role completed: gate
- Result: passed
- Owner: orchestrator/gate
- Commands run:
  - `python scripts/validate_agent_state.py`
  - `python -m pytest -q`
- Exact outputs:
  - `213 passed, 1 skipped`；基线更新：`208 → 213 passed, 1 skipped`
- Acceptance status:
  - 至少三种设置下测量漂移：满足（线程 1/4 × `MKL_CBWR` 有无 = 4 组 + 同进程对照组）；
  - 给出容差并由测试锁住：满足（`1e-2`，两种场景结论写进 D014）；
  - 不无证据声称 bit-exact：满足（长进程场景明确报 5.49e-3）；
  - 测试基线不退化：满足。
- 五角色证据链: FPJ031-IMPL-01 / FPJ031-VERIFY-01 / FPJ031-AP-01 / FPJ031-QUAL-01 / 本记录
- Prohibited claims:
  - 「续训完全一致」；「CPU 训练 bit-exact」。
- Recommended next role: waiting on user（FPJ-032：硬件或答辩形式）

## FPJ032-IMPL-01

- Task: FPJ-032 documentation correction and repository organization before GPU / paper-scale work
- Role completed: implementation (documentation sub-scope; task remains blocked on external GPU/user inputs)
- Owner: /root
- Files changed:
  - `README.md`
  - `docs/ENVIRONMENT.md`
  - `docs/REPRODUCTION_PLAN.md`
  - `docs/DECISIONS.md`
  - `docs/DEFENSE_GUIDE.md`
  - `.agents/QUEUE.md`
  - `.agents/STATE.yaml`
- Corrections:
  - unified the measured task pair as `button-press-wall-v3` + `reach-v3`; retained `reach-wall-v3` only as a future candidate;
  - replaced the stale planned Python 3.8/mujoco-py environment description with the actually used isolated modern sim stack;
  - updated training, EMA, checkpoint and rollout sections from “pending” to the evidence-backed completed state;
  - changed the long-process resume tolerance from stale `2e-3` to measured `1e-2` and recorded the observed `5.49e-3` drift;
  - removed the duplicate unchecked FPJ-031 queue entry and labeled FPJ-029/README results as a low-budget grid;
  - added the measured low-budget results and their non-paper-comparable boundary to the defense guide.
- Commands run:
  - `wsl.exe -d Ubuntu -- /home/s/.local/share/flowpolicy/miniforge-py310/bin/python scripts/validate_agent_state.py`
  - `wsl.exe -d Ubuntu -- /home/s/.local/share/flowpolicy/miniforge-py310/bin/python -m pytest -q`
  - `wsl.exe -d Ubuntu -- /home/s/.local/share/flowpolicy/miniforge-py310/bin/python scripts/check_environment.py --strict-train --allow-cpu`
  - PowerShell local Markdown link check over `README.md` and `docs/*.md`
- Exact results:
  - state validator: passed;
  - pytest: `213 passed, 1 skipped, 8 warnings in 158.13s`; skipped PyTorch module remains separate from passed count;
  - strict CPU environment: `passed: true`, Jittor CPU smoke value `5.0`, `has_cuda: false`;
  - Markdown local links: `Markdown local links OK`.
- Remaining blockers:
  - no compatible CUDA environment has been supplied; paper-scale training cannot start;
  - success/failure videos and final presentation deliverables still need their required format/deadline;
  - the repository still has no initial commit and all project files are untracked; the user did not authorize commit/push in this task.
- Prohibited claims:
  - paper success rate reproduced;
  - CUDA verified;
  - paper-scale training completed;
  - low-budget CPU rates are comparable to the paper;
  - bit-exact long-process save/resume.
- Recommended next role: resume implementation only after the user chooses GPU work or specifies the defense deliverable; repository commit/push requires explicit authorization.

## FPJ032-IMPL-02

- Task: FPJ-032 cloud GPU feasibility, full-channel limited-budget experiment and initial publication
- Role completed: implementation
- Result: passed
- Owner: /root
- Files changed in this continuation:
  - `.gitignore` (root-anchor artifact directories so `src/flowpolicy_jittor/data` is publishable)
  - `.agents/STATE.yaml` (record the publication state and remove the resolved ignore-rule gap)
  - `.agents/HANDOFFS.md` (this append-only record)
- Commands run:
  - `git check-ignore -v data/reach-v3.zarr outputs/cuda/report.json checkpoints/example.pkl`
  - `git check-ignore -q src/flowpolicy_jittor/data/dataset.py`
  - `wsl.exe -d Ubuntu -- /home/s/.local/share/flowpolicy/miniforge-py310/bin/python -m pytest -q`
  - `wsl.exe -d Ubuntu -- /home/s/.local/share/flowpolicy/miniforge-py310/bin/python scripts/validate_agent_state.py`
  - `git commit -m "Initial Jittor reproduction of FlowPolicy"`
  - `git push -u origin main`
- Exact results:
  - root artifact paths remain ignored; `src/flowpolicy_jittor/data/dataset.py` is no longer ignored;
  - publish candidate set contained 62 files / 691128 bytes and no staged dataset, output, checkpoint or file above 10 MiB;
  - secret-pattern filename scan found no common private-key or access-token signature;
  - full regression: `213 passed, 1 skipped, 8 warnings in 180.55s`; the skipped PyTorch module is not counted as passed;
  - root commit `560df9c` pushed to `https://github.com/shview/FlowPolicy-Jittor` on `main`;
  - GitHub page independently showed the commit, source tree, README and MIT license.
- Preserved evidence from the earlier implementation work:
  - A10/CUDA 12.1 real Jittor kernels, real-data updates, full-channel feasibility and headless simulation passed;
  - full `512/1024/2048` configuration completed 3 seeds x 30 epochs x 20 episodes on `button-press-wall-v3`;
  - all local evidence remains outside the public repository under the documented artifact/data/output paths.
- Remaining risks:
  - the 100% result has a ceiling effect and covers only one easy task;
  - detached EMA target is not implemented;
  - no paper-scale 3000-epoch or 37-task claim is allowed.
- Prohibited claims:
  - paper success rate reproduced;
  - full paper recipe completed;
  - detached EMA target used;
  - skipped test counted as passed.
- Recommended next role: verification

## FPJ032-VERIFY-01

- Task: FPJ-032 cloud GPU feasibility, full-channel limited-budget experiment and initial publication
- Role completed: verification
- Result: needs_fix
- Owner: /root/fpj032_verification
- Product files changed: none (read-only verification)
- Coordination files changed:
  - `.agents/STATE.yaml`
  - `.agents/HANDOFFS.md` (this append-only record)
- Commands run and exact results:
  - `wsl.exe -d Ubuntu -- /home/s/.local/share/flowpolicy/miniforge-py310/bin/python -m pytest -q`
    - exit 0; `213 passed, 1 skipped, 8 warnings in 203.73s`;
    - skipped: `tests/test_conv1d_parity.py` because the core environment has no PyTorch; it is not counted as passed.
  - `wsl.exe -d Ubuntu -- /home/s/.local/share/flowpolicy/miniforge-py310/bin/python scripts/check_environment.py --strict-train --allow-cpu`
    - exit 0; `strict_check.passed=true`, CPU smoke value `5.0`, local `has_cuda=false`.
  - independent Python recomputation from the 3 train summaries and 60 raw rollout episodes under `artifacts/aliyun_dsw/formal_pilot_20261115/`
    - exit 0 / `status=PASS`;
    - successes `[20,20,20]`, 60 episodes, 12000 environment steps, success-step range `[107,152]`;
    - latency seed means `[3.8290180806,3.8001388604,3.8026988134]` ms, aggregate `3.8106185848 ± 0.0159857569` ms;
    - last-epoch loss mean `6.8190333531e-6`, validation-loss mean `7.3595988397e-6`, mean train wall time `688.1382816810` s;
    - checkpoint hashes agree between each train summary and its rollout, while three config hashes are distinct by seed and the data hash is common.
  - cloud evidence inspection:
    - `environment_check.log` records real A10/CUDA 12.1 Jittor CUDA kernel success;
    - `gpu_training_summary.json` records 34 real-data updates and a checkpoint path;
    - `full_channels_a10_report.json` plus `03_full_channels_a10_pass.png` record one finite 512/1024/2048 forward/backward/update, 255138308 parameters and the explicit boundary “one feasibility step; not paper-scale training”;
    - `simulation_probe.log` and `rollout_bridge_report.json` record EGL RGB, 1024-point cloud, 9-D state and the cross-environment closed-loop bridge.
  - Git/publication checks:
    - GitHub page independently showed public `main`, latest commit `031d485`, two commits, source tree, README and MIT license;
    - local `HEAD` and `origin/main` both resolve to `031d48533800`;
    - `git ls-tree -r HEAD` contains no root `data/`, `outputs/`, `checkpoints/`, `wandb/`, `artifacts/` or `.zarr/` path;
    - common private-key/token signature scan found none; largest tracked blob is 123275 bytes;
    - `src/flowpolicy_jittor/data/{__init__.py,dataset.py}` is tracked while root data/output/checkpoint paths remain ignored.
- Acceptance status:
  - CUDA, real-data update, full-channel feasibility, headless simulation, hashes and non-paper-result boundaries: verified.
  - P1/blocker — documentation synchronization acceptance is not met after publication:
    - `README.md:72` still lists “GitHub 发布” among unfinished work;
    - `docs/REPRODUCTION_PLAN.md:277` still leaves the verified GitHub hygiene item unchecked;
    - `.agents/QUEUE.md:97` still says the current task is at implementation, conflicting with `.agents/STATE.yaml`;
    - `AGENTS.md:132` still says the worktree has no first commit (protocol debt; outside the current product scope unless handled as protocol maintenance).
- Required fix:
  - update the three in-scope publication/stage statements without changing experimental claims;
  - handle the stale `AGENTS.md:132` statement through an allowed protocol-maintenance edit or explicitly record why it remains historical;
  - rerun the state validator and focused publication checks, then return to verification.
- Prohibited claims remain unchanged:
  - paper success rate reproduced;
  - full 3000-epoch / 37-task recipe completed;
  - detached EMA target used;
  - bit-exact CUDA inference or long-process resume;
  - skipped test counted as passed.
- Recommended next role: implementation

## MAINT-20261115-01

- Type: protocol maintenance
- Owner: /root
- Trigger: the user authorized and completed the first GitHub publication, making one repository-state sentence stale.
- File changed: `AGENTS.md`
- Change: replaced “the worktree has no first commit” with the current safety rule that committed, uncommitted and untracked project assets must not be reset or discarded.
- Protocol impact: none; task id, stage transitions, role responsibilities and acceptance rules were not changed.

## FPJ032-IMPL-03

- Task: FPJ-032 publication-state documentation retry
- Role completed: implementation
- Result: passed
- Owner: /root
- Files changed:
  - `README.md`
  - `docs/REPRODUCTION_PLAN.md`
  - `.agents/QUEUE.md`
  - `AGENTS.md` (factual protocol-maintenance edit recorded separately above)
  - `.agents/STATE.yaml`
  - `.agents/HANDOFFS.md`
- Fixes for `FPJ032-VERIFY-01`:
  - README now links the published repository and no longer lists GitHub publication as unfinished;
  - the reproduction checklist records the verified publication hygiene while explicitly allowing documented environment/mount paths;
  - the queue no longer hard-codes a stale current stage and instead says serial review remains before gate;
  - AGENTS keeps its asset-preservation rule but no longer claims there is no first commit.
- Commands run and exact results:
  - stale-statement `rg` check: `none`;
  - focused boundary check found the GitHub URL, non-paper-result warning, tracked-artifact statement and remaining serial review statement;
  - `git -c core.whitespace=-blank-at-eof diff --check`: no whitespace error;
  - state validator: passed while implementation was in progress;
  - full regression: `213 passed, 1 skipped, 8 warnings in 175.52s`; skipped PyTorch parity is not counted as passed.
- Remaining risks and prohibited claims: unchanged from `FPJ032-VERIFY-01`.
- Recommended next role: verification

## FPJ032-VERIFY-02

- Task: FPJ-032 publication-state documentation retry
- Role completed: verification
- Result: passed
- Owner: /root/fpj032_verification
- Product files changed: none (read-only verification)
- Coordination files changed:
  - `.agents/STATE.yaml`
  - `.agents/HANDOFFS.md` (this append-only record)
- Commands run and exact results:
  - `wsl.exe -d Ubuntu -- /home/s/.local/share/flowpolicy/miniforge-py310/bin/python -m pytest -q`
    - exit 0; `213 passed, 1 skipped, 8 warnings in 183.19s`;
    - skipped: `tests/test_conv1d_parity.py` because PyTorch is absent from the core Jittor environment; it is not counted as passed.
  - `git diff --exit-code 031d485..ae289a6 -- src tests scripts pyproject.toml`
    - exit 0; no source, test, script or package metadata changed in the retry.
  - focused stale-statement and replacement checks
    - all four previous blockers are resolved: README publication status, reproduction checklist, queue review stage and AGENTS first-commit statement;
    - README still states that the result is not the paper success rate.
  - independent publication-hygiene checks
    - `git ls-tree -r ae289a6` contains no root data/output/checkpoint/wandb/artifact or `.zarr/` path;
    - common private-key/token signature scan found none;
    - no measured-result literal (`255138308`, channel sizes, success counts, latency, memory, losses or CPU rates) changed in the retry diff.
  - ref check
    - `HEAD=ae289a66979b8989bf5c1722a962f01c5157d4a4`;
    - `origin/main=ae289a66979b8989bf5c1722a962f01c5157d4a4`.
- Acceptance status:
  - the documentation synchronization blocker from `FPJ032-VERIFY-01` is fixed;
  - the prior independent A10 artifact recomputation and cloud-evidence checks remain applicable because the retry changed no executable or experiment artifact;
  - all FPJ-032 acceptance items are verified at this stage, subject to anti-pattern, quality and gate review.
- Remaining risks and prohibited claims: unchanged; especially no paper success-rate, 3000-epoch/37-task, detached-EMA or bit-exact claim.
- Recommended next role: anti_pattern

## FPJ032-AP-01

- Task: FPJ-032 cloud/publication anti-pattern review
- Role completed: anti_pattern
- Result: passed (no P1/blocker)
- Owner: /root/fpj032_anti_pattern
- Product files changed: none (read-only review)
- Coordination files changed:
  - `.agents/STATE.yaml`
  - `.agents/HANDOFFS.md` (this append-only record)
- Adversarial checks and exact results:
  - `git ls-tree -r --name-only HEAD` filtered for root `data/`, `outputs/`, `checkpoints/`, `wandb/`, `artifacts/`, `.zarr` and model/array extensions: `none`;
  - `git log --all --name-only` with the same artifact filter: `none`, so deleted historical commits do not hide such paths;
  - all reachable blobs were sized; largest was `.agents/HANDOFFS.md` at 129242 bytes, with no large binary/model object;
  - tracked-text scan for common AWS/GitHub/OpenAI/private-key/Aliyun-secret signatures: `none`;
  - root artifact paths remain ignored while `src/flowpolicy_jittor/data/dataset.py` remains visible/tracked;
  - local `HEAD` and `origin/main` both resolved to `ae289a6` before this coordination-only handoff;
  - no `214 passed` claim was found; current sources consistently say `213 passed, 1 skipped`, and `docs/REPRODUCTION_PLAN.md:271` explicitly says the skip is not counted as passed;
  - task names are separated correctly: README/results use `reach-v3` and `button-press-wall-v3`; `reach-wall-v3` is only a future candidate (`README.md:20`, `docs/REPRODUCTION_PLAN.md:205-208`);
  - local CPU and cloud A10 evidence are separated (`README.md:59-68`, `docs/DEFENSE_GUIDE.md:152-160`, `.agents/STATE.yaml:83-88`);
  - one-step feasibility and 30-epoch/one-task training are not presented as the paper recipe (`README.md:48-57`, `docs/REPRODUCTION_PLAN.md:194-201`, `docs/CLOUD_ALIYUN.md:63,84-89`);
  - detached EMA is explicitly absent (`README.md:55-56`, `docs/DEFENSE_GUIDE.md:155-157`, `.agents/STATE.yaml:86`).
- Findings:
  - P2 (non-blocking): the public README cites ignored/local evidence paths (`README.md:40,43,57,68`), so a GitHub reader cannot inspect the raw parity/experiment JSON without rerunning commands or receiving a separate evidence bundle. The README is transparent that the A10 evidence is in the corresponding workspace, so this is a publication-completeness issue, not a false claim.
  - P3 (non-blocking): the historical task title still says “paper-scale/论文规模 training” (`.agents/STATE.yaml:38`, `.agents/QUEUE.md:87`) even though all result sections correctly say limited budget. Future task naming should prefer “full-channel limited-budget experiment” to reduce headline-level ambiguity.
  - P3 (non-blocking): append-only historical handoffs contain obsolete facts by design; current-state claims must continue to cite STATE/README rather than old handoff paragraphs.
- Remaining risks:
  - evidence JSON/screenshots are local and must be preserved separately until a sanitized public evidence bundle is intentionally added;
  - the 100% full-channel result still has a one-task ceiling effect;
  - detached EMA target, second full-channel task, 3000 epochs and 37-task coverage remain incomplete.
- Prohibited claims:
  - paper success rate reproduced;
  - paper-scale/37-task/3000-epoch recipe completed;
  - detached EMA target used;
  - local CUDA available;
  - skipped test counted as passed.
- Recommended next role: quality

## FPJ032-QUAL-01

- Task: FPJ-032 cloud/publication quality review
- Role completed: quality
- Result: passed (no P1/blocker)
- Owner: /root/fpj032_quality
- Product files changed: none (read-only review)
- Coordination files changed:
  - `.agents/STATE.yaml`
  - `.agents/HANDOFFS.md` (this append-only record)
- Interface and usability checks:
  - all five public entry points returned usable `--help` output with exit 0: `check_environment.py`, `train_smoke.py`, `rollout.py`, `experiments.py` and `record_demos.py`;
  - `import flowpolicy_jittor` succeeded in the documented local environment and reported version `0.1.0`;
  - every relative Markdown link in README/docs resolved locally;
  - package metadata pins Jittor and bounded NumPy/Zarr/PyYAML dependencies, and third-party provenance identifies the audited FlowPolicy commit and Jittor source;
  - output/data writers retain refusal-to-overwrite defaults, while CLI help exposes explicit `--overwrite` / `--reuse` controls where applicable.
- Counterexamples constructed:
  1. Fresh GitHub clone: the README contains no install/quick-start commands, while every cited `data/`, `outputs/` and `artifacts/` result path is intentionally untracked. A reader can understand the results but cannot reproduce them from README alone.
  2. Different Linux username: documented rollout/grid commands omit `--sim-python`, but `scripts/rollout.py:37` and `scripts/experiments.py:36` default to `/home/s/.local/share/flowpolicy/sim-env/bin/python`; the CLI override exists, yet the default is not portable.
  3. Interrupted long run: `scripts/train_smoke.py:175-205` exposes no resume argument and only writes final policy/trainer checkpoints at lines 308-315. A future 3000-epoch run would restart after interruption rather than resume, despite trainer-level restore support.
  4. Independent evaluator: README cites `outputs/parity/parity.json`, `outputs/cuda/report.json`, `outputs/experiments/SUMMARY.md` and `artifacts/aliyun_dsw/formal_pilot_20261115/`, but none is in the Git tree; raw A10 numbers cannot be audited from the public repository alone.
- Findings:
  - P1: none.
  - P2 (non-blocking): add a compact README quick start covering two environments, editable install, demonstration generation, training, rollout and tests before calling the public repository self-contained. `docs/REPRODUCTION_PLAN.md` honestly leaves “new machine follows README” unchecked, so this is not a false completion claim.
  - P2 (non-blocking): replace the user-specific simulator interpreter default with a portable configuration mechanism or document `--sim-python` in every public command.
  - P2 (non-blocking): publish a small sanitized evidence bundle (summaries/manifests, not datasets or 255M-parameter checkpoints) so external readers can audit the reported tables and hashes.
  - P2 (future-cost risk): add periodic trainer checkpoints and a tested CLI resume path before any 3000-epoch or other costly cloud run.
  - P3 (non-blocking): rename future task headings from “paper-scale/论文规模” to “full-channel limited-budget”; the body text already uses the correct boundary.
- Positive boundaries:
  - current docs clearly separate CPU from A10, full-channel limited-budget from the 3000-epoch/37-task paper recipe, and stored EMA state from a detached EMA loss target;
  - missing public evidence, videos, second full-channel task and fresh-machine README validation are explicitly disclosed rather than marked complete.
- Remaining risks and prohibited claims:
  - the public repository is source-complete but not yet evidence-complete or fresh-machine-quick-start complete;
  - no paper success-rate, 3000-epoch/37-task, detached-EMA, local-CUDA or bit-exact-resume claim is allowed.
- Recommended next role: gate

## FPJ032-GATE-01

- Task: FPJ-032 cloud GPU feasibility, full-channel limited-budget experiment and initial publication
- Role completed: gate
- Result: passed
- Owner: /root/fpj032_gate
- Five-role evidence chain:
  - implementation: `FPJ032-IMPL-03` (with prior experimental/publication evidence in `FPJ032-IMPL-01/02`);
  - verification: `FPJ032-VERIFY-02`;
  - anti-pattern: `FPJ032-AP-01`;
  - quality: `FPJ032-QUAL-01`;
  - gate: this record.
- Gate command and exact result:
  - `wsl.exe -d Ubuntu -- /home/s/.local/share/flowpolicy/miniforge-py310/bin/python -m pytest -q`
  - exit 0; `213 passed, 1 skipped, 8 warnings in 150.82s`;
  - skipped `tests/test_conv1d_parity.py` because PyTorch is absent from the core Jittor environment; it is not counted as passed.
- Acceptance decisions:
  1. documentation synchronization: passed after attempt-2 fixes and independent stale-statement/link checks;
  2. hardware-compatible CUDA before GPU claims: passed on A10/CUDA 12.1, while local CUDA remains explicitly unavailable;
  3. real Zarr CUDA update and reloadable checkpoint: passed (34 updates plus saved checkpoint evidence);
  4. full 512/1024/2048 feasibility without paper-scale wording: passed (255138308 parameters, finite update, explicit one-step boundary);
  5. isolated RGB/point-cloud/closed-loop simulator: passed (EGL RGB, 1024 points, 9-D state and process bridge);
  6. run fingerprints: passed (hardware/config/data/checkpoint hashes and per-episode results independently recomputed);
  7. no paper-success claim from CPU grid: passed; README and defense guide retain the low-budget/non-comparable boundary.
- Gate decision:
  - no unhandled P1/blocker;
  - FPJ-032 is complete;
  - baseline remains `213 passed, 1 skipped`;
  - activate FPJ-033, then FPJ-034 and FPJ-035 in the user's agreed order.
- Non-blocking quality debt carried into FPJ-033 acceptance:
  - fresh-machine README quick start and portable simulator path;
  - sanitized public evidence bundle;
  - periodic checkpoint plus tested CLI resume before another costly run;
  - future headings use “full-channel limited-budget,” not “paper-scale.”
- Prohibited claims remain:
  - paper success rate or 3000-epoch/37-task recipe reproduced;
  - detached EMA target used;
  - local CUDA available;
  - skipped test counted as passed.
- Recommended next role: FPJ-033 implementation



