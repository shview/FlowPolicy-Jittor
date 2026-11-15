# FlowPolicy-Jittor 串行 Agent 协议

本文件是本仓库唯一权威的 agent 行为规范，作用于整个仓库。`agent.md` 只是入口，不得维护第二套规则。

## 1. 开始工作前必须阅读

按以下顺序完整读取：

1. `AGENTS.md`
2. `.agents/STATE.yaml`
3. `.agents/QUEUE.md`
4. `.agents/HANDOFFS.md` 的最后一条记录
5. 当前任务引用的技术文件

技术内容以现有文件为准，不在 `.agents/` 重复维护：

- 总体路线：`docs/REPRODUCTION_PLAN.md`
- 技术取舍：`docs/DECISIONS.md`
- 环境与命令：`docs/ENVIRONMENT.md`
- 答辩口径：`docs/DEFENSE_GUIDE.md`
- 当前项目说明：`README.md`

## 2. 串行状态机

同一时刻只能有一个 current task 和一个 active role：

```text
implementation
  -> verification
  -> anti_pattern
  -> quality
  -> gate
  -> 下一任务 implementation
```

每个 agent 只执行 `.agents/STATE.yaml` 中当前 `stage` 对应的职责，不得顺手进入下一角色。

父 orchestrator 同一时刻只允许启动一个当前角色 agent。必须等该 agent 返回 final，并完成 state 与 handoff 更新后，才能启动下一角色。若状态为 `in_progress` 且已有 `owner`，其他 agent 只能只读检查并退出，不得抢占。

### implementation

- 可以修改 `current.scope.allowed_paths` 中的文件。
- 必须实现当前 acceptance，不得扩展到下一任务。
- 使用 `apply_patch` 修改文本文件。
- 完成后运行专项测试和要求的全量测试。

### verification

- 默认只读，不修代码。
- 重跑实现者提供的命令，并增加至少一项独立验证。
- 必须报告精确命令、退出码和测试数量。
- skipped 必须单独说明，不能计为 passed。

### anti_pattern

- 只读检查当前任务规定的已知风险和仓库级禁区。
- 使用精确路径和行号报告发现。
- 发现 P1/阻断项时，不得把任务推进到 quality。

### quality

- 只读检查接口、边界条件、组合性、恢复语义和未覆盖路径。
- 主动构造能绕过当前测试的反例。
- 按 P1/P2/P3 报告，不直接修改实现。

### gate

- 不实现功能，只汇总前四个角色的证据。
- 只有没有未处理 P1/阻断项，且 acceptance 全部满足时，才能把任务标为 `done`。
- Gate 通过后更新测试基线，从 `.agents/QUEUE.md` 选取下一任务。
- 未通过则把 `stage` 退回 `implementation`，`status` 设为 `needs_fix`，`attempt` 加一，并清空当前任务的 `role_handoffs` 映射；旧证据仍保留在 append-only handoff 历史中。

## 3. 状态更新与交接

开始任务时：

1. 确认 `status` 为 `ready` 或 `needs_fix`，且 `owner` 为 `null`。
2. 将 `owner` 写为当前 agent 名称。
3. 将 `status` 改为 `in_progress`。
4. 运行 `scripts/validate_agent_state.py`；校验失败则停止产品修改。
5. 不得改动 task id、scope 或 acceptance 来降低验收标准。

结束当前角色时：

1. 在 `.agents/HANDOFFS.md` 末尾追加一条记录，禁止改写历史记录。
2. 写明修改文件、命令、精确结果、剩余风险和禁止声称的结果。
3. 在 `role_handoffs` 中写入本角色 handoff id。
4. 一次性把状态改为 `stage: <next_role>`、`status: ready`、`owner: null`，并更新 `last_handoff_id`。不得停在中间状态等待别人补写。
5. 交给下一角色后立即停止，不继续实现下一阶段。

若发现阻塞：

- 记录已尝试方法和原始错误。
- 若当前角色能在 scope 内修复，退回 implementation。
- 需要用户授权、外部资源或扩大范围时，设置 `status: waiting_user`，不得自行购买云算力、安装大型依赖或修改外部系统。

若 agent 在 `in_progress` 时异常终止，只有 root orchestrator 或用户可以恢复：先追加一条 `recovery` handoff，记录旧 owner、已有改动和恢复原因，再清空 owner 并把状态设为 `needs_fix`。普通 agent 不得自行接管 stale owner。

用户明确要求修改串行协议时，允许以 `protocol_maintenance` 身份修改 `AGENTS.md`、`agent.md`、`.agents/**` 和 validator，而不推进当前产品任务；必须追加 maintenance handoff，并保持 current task、stage 和 role handoffs 不变。

## 4. 当前仓库的验证基线

标准 CPU 测试命令：

```powershell
wsl.exe -d Ubuntu -- /home/s/.local/share/flowpolicy/miniforge-py310/bin/python -m pytest -q
```

严格 CPU 环境检查：

```powershell
wsl.exe -d Ubuntu -- /home/s/.local/share/flowpolicy/miniforge-py310/bin/python scripts/check_environment.py --strict-train --allow-cpu
```

当前已知基线写在 `.agents/STATE.yaml`。若实际结果与基线不同，必须说明原因，不能直接覆盖旧结果。

每次交接前还必须验证协调状态：

```powershell
wsl.exe -d Ubuntu -- /home/s/.local/share/flowpolicy/miniforge-py310/bin/python scripts/validate_agent_state.py
```

## 5. 仓库级安全与边界

- `../flowpolicy-upstream/` 是只读对照，不得修改。
- 不得直接执行上游带硬编码路径或自动删除逻辑的 shell 脚本。
- 不得使用递归删除处理数据、checkpoint、仓库或环境。
- 不得自动覆盖已有 checkpoint；覆盖必须是任务明确要求并经过测试的行为。
- 不得把 PyTorch 放入 Jittor 训练计算图；PyTorch 只能用于独立 parity 测试或仿真预处理，并需说明边界。
- 训练器必须使用 `policy.trainable_parameters()`，不能直接使用 `policy.parameters()`。
- 未经用户明确要求，不 commit、push、创建 GitHub 仓库、上传文件、租用云 GPU 或运行高成本正式实验。
- 当前工作树尚未首次提交，所有既有文件均视为用户/项目资产，不得重置或丢弃。

## 6. 禁止的结果表述

在相应证据完成前，任何 agent 都不得声称：

- 已复现论文成功率；
- CUDA 或完整 `512/1024/2048` 通道已经通过；
- 已完成 PyTorch/Jittor 精确对齐；
- 已完成 EMA 或 paper-faithful target；
- 已完成真实 Zarr 数据训练；
- 已完成 MetaWorld/MuJoCo rollout；
- policy checkpoint 等同于完整 trainer checkpoint；
- skipped 测试等于 passed。

## 7. 文件所有权原则

- implementation 只修改当前 scope。
- verification、anti_pattern、quality 默认不修改产品代码。
- `.agents/STATE.yaml` 是唯一当前状态来源。
- `.agents/HANDOFFS.md` 只追加。
- `.agents/QUEUE.md` 只在 gate 通过、任务重排或用户改变目标时修改。
- 新的长期技术决策写入 `docs/DECISIONS.md`，不要塞进 handoff。

