# Journal-backed Session Tree

Lite 将 Session Tree 作为 Session Journal 的一个确定性投影，而不是另建一份
可变的树文件。权威文件是：

```text
.lite/sessions/<session-id>.journal.jsonl
```

`.json` 是旧 Session 的迁移源；`.events.jsonl` 是面向用户的粗粒度事件流；
`.lite/runs/<run-id>/trace.jsonl` 是单次运行诊断 trace。三者都不是 Session Tree
的权威历史。

## 一份 Journal，多个投影

每条 Journal record 先通过 schema 和 reducer invariant 校验，再追加并 fsync，最后
更新内存投影：

- Session 投影：兼容现有 runtime 的 `session["history"]`。
- Tree 投影：所有节点、父子索引、label 和 active head。
- Effect 投影：未完成以及已完成的 provider/tool/permission 等外部操作。
- Branch-state 投影：沿 active path 推导 message、plan、todo、working state 和
  checkpoint。

`history` 是 active head 的兼容视图，不再是完整历史。切换 head 后 reducer 会重新
生成它；未选中的分支节点仍保留在 Journal 中。

## Record 与节点

兼容的 Journal envelope 版本保持 `lite.session_journal.v1`。新增 record kind：

- `tree_entry_appended`：追加一个父节点固定的不可变节点，并将其设为 head。
- `head_moved`：只移动游标，不生成会进入模型上下文的伪消息。
- `tree_label_updated`：为已有节点添加稳定名称。

旧的 `history_appended` 会确定性映射为 `message` 节点；旧的
`history_replaced` 会映射为 `context_replacement` 节点。因此老 Journal、迁移后的
JSON Session 和新树记录可以使用同一个 reducer 恢复。

节点公共字段为：

```json
{
  "entry_id": "entry_...",
  "parent_id": "entry_... or null",
  "entry_type": "message",
  "turn_id": "turn_...",
  "run_id": "run_...",
  "created_at": "2026-08-10T00:00:00+00:00",
  "data": {"message": {"role": "user", "content": "..."}}
}
```

当前支持 `message`、`tool_exchange`、`compaction`、`branch_summary`、
`task_checkpoint`、`plan_delta`、`todo_delta`、`working_state` 和
`context_replacement`。

## Effect 与上下文的原子提交

`effect_result` 可选携带：

```json
{
  "tree_delta": {
    "expected_head": "entry_before_effect",
    "entries": ["one or more serialized tree entries"]
  }
}
```

reducer 要求 `expected_head` 与 intent 锚定的当前 head 一致，且 entries 构成连续
append。整个 `effect_result` 写盘成功后，Effect 完成状态与这些节点才会同时变为
可见；写盘前不会更新任何投影。`tool_exchange` 将 assistant tool calls 与按相同
call-id 顺序排列的全部 results 放在一个节点中，防止恢复出半个工具交换。

## Compaction 与 rewind

Compaction 追加 `compaction` 节点，其中保存新的有界 context view；被摘要的原始
节点不删除。`/rewind` 和 `/branch` 只移动会话 head，绝不会自动回滚或覆盖工作区
文件。恢复旧 checkpoint 时会比较 workspace fingerprint 并提示漂移。

## 关键 invariant

- Journal sequence 必须连续，record id 内容冲突视为损坏。
- 节点 id 唯一，parent 必须存在，新节点 parent 必须是 active head。
- Effect 未结束时禁止追加节点、移动 head 或修改 Session。
- Effect result 必须匹配唯一的 open intent。
- Effect tree delta 必须从 `expected_head` 连续追加。
- `tool_exchange` 的 result call ids 必须与 assistant tool calls 完全同序。
- 快照同时校验 Journal 前缀哈希、Session 投影和 Tree 投影。
