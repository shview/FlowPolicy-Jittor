# 交接模板

复制以下内容并追加到 `.agents/HANDOFFS.md` 末尾。ID 建议使用 `任务ID-角色-序号`，例如 `FPJ-021-VERIFY-01`。

```markdown
## <HANDOFF_ID>

- Task: <task id and title>
- Role completed: implementation | verification | anti_pattern | quality | gate
- Result: passed | needs_fix | blocked | waiting_user
- Owner: <agent name>
- Files changed:
  - <path or "none">
- Commands run:
  - `<exact command>`
- Exact outputs:
  - `<exit code, passed/skipped/failed counts, important values>`
- Acceptance status:
  - [ ] <criterion 1>
  - [ ] <criterion 2>
- Findings:
  - P1: <finding or none>
  - P2: <finding or none>
  - P3: <finding or none>
- Remaining risks:
  - <risk>
- Prohibited claims:
  - <what is still not demonstrated>
- Recommended next role: <role>
- Next agent should read:
  - `<path>`
```

交接要求：

- 不写“测试通过”而省略测试数量；
- 不把 skipped 算进 passed；
- 不使用“应该没问题”代替证据；
- verification/anti_pattern/quality 若修改了产品代码，必须说明原因并把流程退回 implementation；
- 发现 P1 时推荐下一角色必须是 implementation。
- 完成本角色时，在 `STATE.yaml.current.role_handoffs` 写入本记录 ID，并一次性切换到下一 `stage`、`status: ready`、`owner: null`。
- 不得留下 `in_progress` 状态让下一 agent 代为收尾。

