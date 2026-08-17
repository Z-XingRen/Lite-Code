# Lite-Code

**Lite-Code** 是一个运行在本地代码仓库中的终端 coding agent。它通过你配置的
OpenAI-compatible 或 Anthropic-compatible 模型读取代码、搜索文件、执行命令、
修改工作区，并将会话、事件和运行证据保存在本地。

CLI 命令为 `lite`。默认界面是基于 Textual 的 TUI，同时支持普通 REPL 和一次性任务。

<p align="center">
  <img src="assets/screenshots/lite-tui-intro.png" alt="Lite-Code terminal UI" width="960">
</p>

## 核心能力

| 能力 | 说明 |
| --- | --- |
| 本地仓库工作流 | 围绕当前 workspace 读取、搜索、修改和验证代码，不依赖托管控制面。 |
| 原生工具调用 | 支持 OpenAI-compatible 与 Anthropic-compatible 协议，工具参数统一经过本地 Schema 校验。 |
| 可恢复运行时 | 使用 append-only Session Journal 记录会话和外部 effect，支持恢复、压缩、分支与回退。 |
| 上下文治理 | 按预算组织 workspace、history、memory 和当前请求，并记录压缩与 prompt cache 决策。 |
| 受控执行 | 提供风险操作审批、路径边界、敏感变量脱敏、验证回执和可选 shell sandbox。 |
| 可观测性 | 为每次 session 和 run 写入事件流、trace、报告与工具结果 artifact。 |
| 可扩展工作流 | 内置 plan、todo、memory 和 Markdown skills；多 agent 能力可通过实验开关启用。 |

## 快速开始

### 1. 安装

要求：Python 3.10 或更高版本、Git，以及至少一个可用的模型 API key。

```bash
git clone https://github.com/Z-XingRen/Lite-Code.git
cd Lite-Code
python -m venv .venv
```

激活虚拟环境：

```bash
# macOS / Linux
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

安装 Lite-Code：

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

### 2. 配置 Provider

复制项目配置模板：

```bash
# macOS / Linux
cp .lite.toml.example .lite.toml
```

```powershell
# Windows PowerShell
Copy-Item .lite.toml.example .lite.toml
```

然后在 `.lite.toml` 中选择 provider。最小的 OpenAI-compatible 配置如下：

```toml
provider = "openai"

[providers.openai]
protocol = "openai"
base_url = "https://api.openai.com/v1"
model = "your-model"
strict_tools = false
```

API key 推荐放在环境变量或仓库根目录的 `.env` 中，不要写入 TOML：

```dotenv
OPENAI_API_KEY=sk-...
```

`.lite.toml`、`.env` 和运行状态目录 `.lite/` 已被项目的 `.gitignore` 忽略。

### 3. 启动

```bash
lite
```

启动后可以直接输入任务，例如：

```text
分析当前测试失败的根因，修复后运行相关测试。
```

## Provider 配置

Provider profile 的名称用于选择配置，真正决定请求格式的是 `protocol`。当前支持
`openai` 和 `anthropic` 两种协议，因此兼容网关和第三方模型也可以接入。

| Provider 示例 | `protocol` | 推荐 API key 变量 |
| --- | --- | --- |
| OpenAI-compatible | `openai` | `OPENAI_API_KEY` |
| Anthropic-compatible | `anthropic` | `ANTHROPIC_API_KEY` |
| DeepSeek Anthropic endpoint | `anthropic` | `DEEPSEEK_API_KEY` |

Anthropic-compatible profile 示例：

```toml
provider = "anthropic"

[providers.anthropic]
protocol = "anthropic"
base_url = "https://api.anthropic.com"
model = "your-model"
models = ["your-model"]
reasoning_effort = "high"
reasoning_efforts = ["low", "medium", "high"]
strict_tools = false
```

`models` 和 `reasoning_efforts` 用于 TUI/REPL 中的 `/model` 选择器；`model` 和
`reasoning_effort` 是启动时的默认值。

普通配置的优先级为：

```text
CLI 参数 > 项目 .lite.toml > 全局 ~/.config/lite/config.toml > 环境变量/.env > 默认值
```

API key 单独按 `CLI > 环境变量/.env > TOML 兼容值` 解析。建议始终通过环境变量或
`.env` 提供密钥。

常见临时覆盖：

```bash
lite --provider openai --model your-model
lite --provider anthropic --reasoning-effort high
lite --config /path/to/config.toml --cwd /path/to/repo
```

完整字段、环境变量和默认值见 [配置文档](docs/configuration.md)。

### Prompt Cache

Lite-Code 会为 OpenAI-compatible 对话维护 append-only provider projection，使稳定的
对话前缀可以跨 turn 复用。模型、endpoint、工具定义、规范会话历史或上下文代际变化时，
runtime 会重新建立 projection；workspace 更新则以新快照追加，保留此前可缓存的前缀。

只有确认网关支持并转发 OpenAI 显式 prompt-cache 字段时，才启用：

```toml
[providers.openai]
supports_explicit_prompt_cache = true
```

仅支持自动缓存的 endpoint 应保持 `false`。每轮 projection 决策和 provider usage
都会进入 run evidence，便于区分 cached、billable 与总输入 token。

## 运行方式

```bash
lite                                  # 默认启动 TUI
lite --tui                            # 显式启动 TUI
lite --repl                           # 使用普通行式 REPL
lite "修复这个仓库中的类型错误"        # 执行一次 one-shot 任务
lite --prompt-file task.txt           # 从 UTF-8 文件读取一次性任务
lite --resume latest                  # 恢复最近的 session
lite --cwd /path/to/repo              # 指定工作目录
```

常用运行控制：

```bash
lite --approval ask                   # 风险操作前询问，默认值
lite --approval auto                  # 自动批准风险操作
lite --approval never                 # 拒绝风险操作
lite --sandbox best_effort            # 尝试隔离，不可用时记录并回退
lite --sandbox required               # 无可用隔离后端时拒绝 shell
lite --final-readiness enforce        # 未满足完成条件时阻止直接结束
lite --max-steps 80                   # 限制单次请求的模型/工具迭代
```

运行 `lite --help` 查看完整 CLI 参数。

## 交互命令

TUI 和 REPL 都支持 slash command：

| 命令 | 作用 |
| --- | --- |
| `/help` | 查看全部命令。 |
| `/model [name]` | 选择模型与 reasoning effort，或按名称直接切换模型。 |
| `/session`、`/history`、`/resume` | 查看、列出和恢复 session。 |
| `/context`、`/usage` | 查看上下文预算和 provider usage。 |
| `/plan <topic>`、`/plan-exit` | 进入或退出只允许写 plan artifact 的规划模式。 |
| `/compact` | 压缩较早的会话上下文，原始 Journal 记录不会删除。 |
| `/tree`、`/branch`、`/rewind`、`/label` | 浏览和移动 append-only Session Tree 的当前 head。 |
| `/memory`、`/working-memory` | 查看 durable memory 索引和当前工作记忆。 |
| `/remember <text>`、`/dream` | 写入 daily log，或手动整理 durable topics。 |
| `/skills`、`/skill <name> [args]` | 查看并调用 Markdown skill。 |
| `/clear`、`/reset` | 创建空 session，或重置当前 session 的历史与工作记忆。 |
| `/exit` | 退出 Lite-Code。 |

`/rewind` 和 `/branch` 只移动会话 head，不会回滚工作区文件。

## 安全模型

### 操作审批

| 策略 | 行为 |
| --- | --- |
| `ask` | 只读工具直接执行，风险工具在执行前请求确认；默认值。 |
| `auto` | 风险工具自动放行，并记录审批原因。 |
| `never` | 只读工具仍可使用，风险工具直接拒绝。 |

无论审批策略如何，工具 Schema、workspace 路径、write scope、patch 唯一匹配和敏感变量
脱敏等检查都会执行。

### Shell Sandbox

Sandbox 默认是 `off`，需要显式启用：

| 模式 | 行为 |
| --- | --- |
| `off` | 通过主机 shell 执行；sandbox 的网络和挂载策略不生效。 |
| `best_effort` | 优先使用隔离后端；不可用时记录 `sandbox_unavailable` 并回退主机 shell。 |
| `required` | 后端不可用或策略无法安全表达时拒绝执行。 |

支持的后端包括 Linux `bubblewrap`、macOS `sandbox-exec`，以及跨平台的 Docker/Podman。
Windows 当前没有原生 restricted-token 后端，`required` 模式需要可用的容器 daemon。
启用 sandbox 后网络默认关闭；单条命令可以申请一次性的网络或额外目录权限，该申请仍受
审批和 hard deny 约束。

平台边界、配置项和已知限制见 [Sandbox 文档](docs/sandbox.md)。

## Session、状态与证据

Lite-Code 默认把本地状态写入仓库根目录的 `.lite/`：

| 内容 | 默认路径 |
| --- | --- |
| 项目配置 | `.lite.toml` |
| 全局配置 | `~/.config/lite/config.toml` |
| 权威 Session Journal | `.lite/sessions/<id>.journal.jsonl` |
| 旧 Session / 迁移源 | `.lite/sessions/<id>.json` |
| Session 事件流 | `.lite/sessions/<id>.events.jsonl` |
| 单次运行证据 | `.lite/runs/<run_id>/` |
| Working / durable memory | `.lite/memory/` |
| Plan artifacts | `.lite/plans/` |

Journal 采用“校验、持久化 append、推进内存投影”的顺序记录 provider、tool、permission
等 effect。恢复时可以重放出 session、tree 和未完成 effect 状态；Session Tree 则在
不复制另一份可变树文件的前提下提供分支、标签和 rewind。设计与 invariant 见
[SESSION_TREE.md](SESSION_TREE.md)。

每次 run 的目录包含 trace、report 和按需外置的大型工具结果。密钥值会在这些 artifact
写盘前脱敏，但 `.lite/` 仍应只保留在本地，不应提交到版本库。

## Memory 与 Skills

Working memory 默认启用，用于保存当前任务、最近文件和短摘要。`/remember` 会追加本地
daily log，`/dream` 可以将日志整理为 durable topics。自动 dream 和 durable topic 检索
属于 opt-in 实验能力，模板中的默认值均为 `false`：

```toml
[experimental]
multi_agent = false
auto_dream = false
durable_memory_retrieval = false
```

详细的目录结构、写入和检索规则见 [Memory 文档](docs/memory.md)。

Skill 是 Markdown 定义的可复用工作流。Lite-Code 内置 `/review`、`/test`、`/commit`
和 `/simplify`，并按以下顺序加载同名 skill，后者覆盖前者：

1. Lite-Code 内置 skills
2. `~/.lite/skills/<name>/SKILL.md`
3. `<repo>/skills/<name>/SKILL.md` 或 `<repo>/.lite/skills/<name>/SKILL.md`

最小示例：

```markdown
---
name: deploy
description: 部署前检查
argument-hint: target
allowed-tools: read_file, search
---

检查 $ARGUMENTS 环境的测试、配置和发布清单。
```

调用方式：

```text
/deploy staging
```

Frontmatter、参数替换、工具限制和 fork context 见 [Skills 文档](docs/skills.md)。

## 开发与验证

推荐使用 [uv](https://docs.astral.sh/uv/) 同步锁定的开发环境：

```bash
uv sync --group dev --locked
uv run ruff check .
uv run mypy
uv run python -m pytest -q
```

默认 pytest 配置不运行 stress 标记；长会话和扩展性门禁单独执行：

```bash
uv run python -m pytest -m stress -q
```

不使用 uv 时：

```bash
python -m pip install -e .
python -m pip install pytest pytest-asyncio ruff mypy coverage
python -m pytest -q
```

普通单元测试不依赖网络。真实 provider smoke 需要显式启用并提供有效 endpoint 与 key：

```bash
LITE_LIVE_SMOKE=1 python -m pytest tests/test_release_smoke.py -q
```

仓库还提供可复现的 runtime 与长会话验证入口：

```bash
python scripts/run_runtime_evidence.py --output-dir artifacts/runtime-evidence
python scripts/run_long_session_state_benchmark.py --validate-only
python scripts/run_prompt_cache_turn_harness.py --smoke
```

这些脚本分别覆盖 workspace/effect/journal 证据、冻结的长会话状态恢复数据，以及跨 turn
prompt cache projection。涉及真实 provider 的运行会产生费用；具体参数和结论边界以脚本
输出及 [Runtime 加固报告](docs/runtime-hardening-20260810.md) 为准。

## 项目结构

```text
lite/
├── cli.py                 # CLI 参数、启动模式和 REPL
├── commands/              # slash command 注册与解析
├── config/                # Provider profile、TOML 和环境变量解析
├── core/                  # Runtime、engine、session、context 和 workers
├── features/              # Memory、skills 和 sandbox
├── providers/             # OpenAI / Anthropic compatible client
├── tools/                 # 工具 Schema、注册表和具体实现
├── tui/                   # Textual 终端界面
└── evaluation/            # Evidence、metrics、benchmark 和 verifier
```

## 延伸阅读

- [配置](docs/configuration.md)
- [Shell Sandbox](docs/sandbox.md)
- [Memory](docs/memory.md)
- [Skills](docs/skills.md)
- [Session Tree](SESSION_TREE.md)
- [Runtime 优化与证据](docs/runtime-hardening-20260810.md)
- [长会话状态基准](benchmarks/long_session_state_v1/README.md)

## License

MIT
