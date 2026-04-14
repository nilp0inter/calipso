# Components

## Model

**Source:** `src/calipso/model.py`

Configures the Pydantic AI `Model` instance for provider-agnostic LLM communication. Uses `OpenAIChatModel` with an OpenRouter provider (currently `x-ai/grok-4-fast`). Exposes `create_model()` and `create_http_client()` for HTTP-level instrumentation.

## Runner

**Source:** `src/calipso/runner.py`

The agentic loop. Takes a `Model` and a `Context` widget, then:
1. Materializes the context's views (`view_messages()`, `view_tools()`)
2. Calls `Model.request()` directly
3. Dispatches tool calls back to the context
4. Loops until the model produces a text response

## CLI

**Source:** `src/calipso/cli.py`

A REPL entry point registered as the `calipso` console script in `pyproject.toml`. Creates the model, assembles the widget tree into a `Context`, and runs the turn loop.

Each turn's raw HTTP request/response payloads are saved to `prompts/NNNN.json`, captured via httpx event hooks on the underlying HTTP client.

## Widgets

Everything in the agent's context is a widget — an Elm-inspired component with state, view functions (generators yielding messages or tool definitions), and update handlers. Widgets compose via `yield from`.

**Base class:** `src/calipso/widget.py`

### Implemented

| Widget | Source | State | DSL (tools) | Renders |
|---|---|---|---|---|
| **SystemPrompt** | `src/calipso/widgets/system_prompt.py` | None | None | Static instruction text |
| **Goal** | `src/calipso/widgets/goal.py` | Current objective text | `set_goal`, `clear_goal` | Goal text in instructions |
| **TaskList** | `src/calipso/widgets/task_list.py` | Tasks with statuses (`pending`, `in_progress`, `done`) | `create_task`, `update_task_status`, `remove_task` | Compact checklist with status icons |
| **ActionLog** | `src/calipso/widgets/action_log.py` | Log entries + active action | `action_log_start`, `action_log_end` | Rules in instructions, collapsed entry summaries in history |
| **Conversation** | `src/calipso/widgets/conversation.py` | Turn pairs + compaction summaries | None (fed by runner) | Recent turns as messages, old turns as compacted summaries |
| **Context** | `src/calipso/widgets/context.py` | Child widgets | None | Composes all children via `yield from`, dispatches tool calls |

### Planned

| Widget | State | DSL | Renders |
|---|---|---|---|
| **Pytest** | Test results | None (reactive) | `TESTS PASSING` or failure details |
| **Code Explorer** | LSP-sourced structural view | Subset of LSP commands | Module/class/function overview |
| **Code Summary** | Compact description from a cheap model | Select target and detail level | Dense summary of what the code does |
