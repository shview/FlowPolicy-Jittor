# 串行协作目录

`.agents/` 只记录执行状态，不重复论文、算法或环境文档。

## 文件职责

- `STATE.yaml`：唯一机器可读的当前任务和角色。
- `QUEUE.md`：按 gate 顺序排列的后续任务。
- `HANDOFFS.md`：只追加的角色交接证据。
- `HANDOFF_TEMPLATE.md`：每次交接必须使用的字段。

## 状态流转

```text
ready
  -> in_progress
  -> 下一角色 ready（一次更新 stage/status/owner/handoff）
```

发现阻断项：

```text
in_progress
  -> needs_fix
  -> implementation
```

需要用户或外部资源：

```text
in_progress
  -> waiting_user
```

五个角色全部通过后：

```text
gate passed
  -> current task done
  -> queue 中下一任务 ready
```

交接前运行：

```powershell
wsl.exe -d Ubuntu -- /home/s/.local/share/flowpolicy/miniforge-py310/bin/python scripts/validate_agent_state.py
```

该脚本只读检查 YAML schema、状态枚举、当前任务是否在队列中，以及 handoff id 是否存在。

当状态为 `in_progress` 时必须存在 owner；当状态为 `ready/needs_fix` 时 owner 必须为空。若 agent 异常退出留下 stale owner，只能由 root/user 通过 recovery handoff 恢复。

## 交接的最低证据

- 改了哪些文件；
- 运行了哪些命令；
- 精确测试数量和退出状态；
- 哪些测试被跳过以及原因；
- 是否出现 P1/P2/P3；
- 下一角色应该验证什么；
- 当前仍不能声称什么结果。


