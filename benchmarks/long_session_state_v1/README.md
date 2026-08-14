# Long-session State Benchmark v1

这是一份按字节冻结的长会话状态恢复测试集。`events.jsonl` 恰好包含
100 个事件；`ground_truth.json` 包含当前有效状态、失效状态、任务、证据和
20 个可自动评分问题。`manifest.json` 保存两份输入的 SHA-256，运行器会在
任何模型调用前校验哈希，防止不同方案误用不同数据。

覆盖情况：

- 20 个初始或新增需求，18 次明确需求更新；
- 5 个显式撤销的决策；
- 5 次工具失败，每次都有 retry 和成功结果；
- 3 次 rewind 和 3 次 resume；
- 3 组 rewind 分支证据 `EV004`、`EV008`、`EV011` 明确失效。

运行冻结校验：

```powershell
.\.venv\Scripts\python.exe scripts\run_long_session_state_benchmark.py --validate-only
```

使用 `.lite.toml` 当前 provider/model 跑真实评测（默认 5 次、temperature 0）：

```powershell
.\.venv\Scripts\python.exe scripts\run_long_session_state_benchmark.py
```

若进程在已完成若干轮后中断，可复用通过 identity 校验的完整轮次，只补跑缺失或
blocked 的轮次：

```powershell
.\.venv\Scripts\python.exe scripts\run_long_session_state_benchmark.py --resume-existing
```

每轮结果和总报告均原子写入；重新执行 blocked 轮次前，旧结果会保存在同一轮目录的
`result.failed-NNN.json` 中。

当前仓库 TOML 选择的模型会被原样使用；运行器没有 `--model` 覆盖参数。
结果写入 `artifacts/long-session-state-benchmark/results.json` 和 `report.md`。

每次重复使用同一 journal，分别评测：

1. `full_history`：完整 100 个事件；
2. `compacted_history`：在 25/50/75/96/100 游标增量压缩后的状态；
3. `resumed_context`：E096 的恢复状态，加 E097-E100 的恢复后事件。

运行器为三个 variant 建立互相隔离的 prompt-cache lane。压缩构建链和最终探测
也使用不同 lane，最终探测只能看到压缩状态，不会泄漏此前的完整压缩对话。Resume
会先把 E096 状态原子写为 checkpoint，再从磁盘加载到全新的 provider 对话，最后
追加 E097-E100。每轮共 12 次模型调用，默认 5 轮共 60 次。

报告把可直接横向比较的 `probe input` 与 `compaction input` 分开，同时记录
total/cached/billable input、cache-key stability、probe P95、compaction-call P95、
顺序 pipeline P95 和包含 checkpoint 读取的 Resume recovery。variant 成本互不重叠，
其合计必须等于 Provider 实际总输入；不再把共享调用重复计入多条路径。

结构化答案由代码精确评分。事实错误、陈旧事实、证据错误和 JSON 契约错误分别
报告；非规范键、额外约束或带说明的 evidence ID 属于 `schema_violations`，不会再
被错误标成 hallucination。严格通过要求事实门槛和 schema contract 同时通过。
只有将来加入非结构化语义题时才应启用模型裁判；当前 v1 不用模型裁判，避免把
裁判噪声混入状态恢复分数。

本基准评测的是冻结 journal 上的 provider 压缩/恢复提示链，不会经过 Lite Agent
Runtime 的真实 session lifecycle；其结果不能单独作为 Runtime 回归结论。
