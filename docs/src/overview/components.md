# Components

## Model

**Source:** `src/calipso/model.py`

Configures the Pydantic AI `Model` instance for provider-agnostic LLM communication. Uses `OpenAIChatModel` with an OpenRouter provider (currently `x-ai/grok-4-fast`). Exposes `create_model()` and `create_http_client()` for HTTP-level instrumentation.

## Summarizer

**Source:** `src/calipso/summarizer.py`

A Pydantic AI agent that takes Python code (with comments already stripped) and produces a summary preserving signatures verbatim while replacing code bodies with `[...REDACTED...]` followed by a description. Uses a cheap, fast model (`google/gemini-3.1-flash-lite-preview` via OpenRouter) to keep costs low. Called by the `CodeExplorer` widget after every tree-sitter query.

## Runner

**Source:** `src/calipso/runner.py`

The agentic loop. Takes a `Model` and a `Context`, then:
1. Materializes the context's views (`view_messages()`, `view_tools()`)
2. Calls `Model.request()` directly
3. Dispatches tool calls back to the context via `handle_response()` (which calls `dispatch_llm()` on owning widgets)
4. Records tool results in the conversation log via `send(ToolResultsReceived(...))`
5. Calls an optional `on_update` callback after every state mutation (used by the dashboard)
6. Loops until the model produces a text response

## Dashboard Server

**Source:** `src/calipso/server.py`

An aiohttp HTTP + WebSocket server that provides a live browser dashboard. Serves the SPA at `/` and accepts WebSocket connections at `/ws`. On connection, sends all widget HTML. After every state mutation, pushes only changed widget fragments via htmx out-of-band swaps. Also handles turn lifecycle signals (thinking indicator, input disabling).

The server handles two types of inbound WebSocket messages: `user_input` (enqueued for the runner's main loop) and `widget_event` (dispatched synchronously to the target widget's `dispatch_ui()` method via `Context.handle_widget_event()`, bypassing the LLM and action log protocol). After a widget event, changed HTML is pushed immediately.

## CLI

**Source:** `src/calipso/cli.py`

An async entry point registered as the `calipso` console script in `pyproject.toml`. Creates the model, assembles the widget tree by calling factory functions (`create_system_prompt()`, `create_goal()`, etc.) and passing the resulting `WidgetHandle` instances into a `Context`, starts the `DashboardServer`, and loops reading user input from the server's WebSocket queue.

Each turn's raw HTTP request/response payloads are saved to `prompts/NNNN.json`, captured via httpx event hooks on the underlying HTTP client.

## SPA

**Source:** `src/calipso/static/index.html`

A single-page htmx application served by the dashboard server. Uses the htmx WebSocket extension to connect to `/ws`. The layout follows a VS Code-style pattern: a narrow activity bar on the left with icon tabs (one per non-conversation widget), a collapsible side panel (320px) that shows the selected widget's content, and a chat area (ConversationLog + input bar) filling the remaining space. Each widget div lives inside a stable wrapper container; the server's `hx-swap-oob` updates replace the inner widget div without disturbing the wrapper's visibility state. Activity bar tabs show a blue notification badge when their widget receives an update while not currently active (the first WebSocket message — initial state — is excluded). A thinking indicator with animated dots appears during turns while the input is disabled. A global `sendWidgetEvent(toolName, args)` JS function sends `widget_event` messages over the WebSocket, allowing widgets to render interactive HTML (buttons, checkboxes, inputs) that trigger their own updates without an LLM round-trip.

## Widgets

Everything in the agent's context is a widget — an Elm-inspired component following the Model/Update/View pattern. Each widget module defines:

- **Model** — a `@dataclass(frozen=True)` holding pure state
- **Msg** — a union of `@dataclass(frozen=True)` variants declaring valid messages (e.g., `GoalMsg = SetGoal | ClearGoal`)
- **update(model, msg) → (model, str)** — a pure function that pattern-matches on Msg, returns new model + tool result text
- **View functions** — free functions `view_messages(model)`, `view_tools(model)`, `view_html(model)` that render the model into the LLM context and browser dashboard
- **from_llm(model, tool_name, args) → Msg** — async anticorruption layer converting LLM tool calls to typed Msgs (may do I/O; `ValueError` is caught and returned as tool result)
- **from_ui(model, event_name, args) → Msg | None** — sync anticorruption layer converting browser events to Msgs
- **create_xxx() → WidgetHandle** — factory function assembling all of the above

Widgets are created via factory functions that return a `WidgetHandle` (`src/calipso/widget.py`) — the uniform interface holding a model reference + function table. The handle exposes: `view_messages()`, `view_tools()`, `view_html()`, `widget_id()`, `frontend_tools()`, `dispatch_llm()` (from_llm → update), `dispatch_ui()` (from_ui → update, sync), `send(msg)` (direct Msg dispatch bypassing anticorruption layers), and `.model` (read access to current state). Views compose via `yield from`. HTML is rendered via `render_md()` (markdown to safe HTML).

I/O resources (e.g., CodeExplorer's tree-sitter parser and summarizer agent) are captured in `from_llm` closures by the factory, not stored in the model. The ConversationLog is a regular `WidgetHandle` — Context interacts with it via `send()` for direct Msgs and pure query functions (`check_protocol()`, `current_segment()`) imported from the module.

### Implemented

| Widget | Source | Model | Msg types | Tools | Renders |
|---|---|---|---|---|---|
| **SystemPrompt** | `system_prompt.py` | `SystemPromptModel(text)` | None (no update) | None | Identity and workspace framing text |
| **AgentsMd** | `agents_md.py` | `AgentsMdModel(loaded_path, content, error)` | `AgentsReloaded` | `reload_agents_md`\* | Behavioral instructions from `AGENTS.md`/`CLAUDE.md` |
| **Goal** | `goal.py` | `GoalModel(text)` | `SetGoal \| ClearGoal` | `set_goal`\*, `clear_goal`\* | Goal panel with inline edit input and clear button |
| **TaskList** | `task_list.py` | `TaskListModel(tasks, next_id)` | `CreateTask \| UpdateTaskStatus \| RemoveTask` | `create_task`, `update_task_status`\*, `remove_task`\* | Checklist with interactive checkboxes and remove buttons |
| **ConversationLog** | `conversation_log.py` | `ConversationLogModel(turns, active_action, ...)` | `UserMessageReceived \| ResponseReceived \| ToolResultsReceived \| ActionLogStart \| ActionLogEnd \| ToolTracked` | `action_log_start`, `action_log_end` | Action protocol rules + conversation history; summarized segments render summary + tool parts |
| **CodeExplorer** | `code_explorer.py` | `CodeExplorerModel(open_files, query_results)` | `FileOpened \| FileOpenError \| FileClosed \| QueryCompleted \| QueryError` | `open_file`, `close_file`\*, `query`, `query_all` | Open files + tree-sitter query results (signatures + `[...REDACTED...]` body summaries) |
| **FileExplorer** | `file_explorer.py` | `FileExplorerModel(listing_*, open_file_*)` | `DirectoryListed \| DirectoryListError \| FileRead \| FileReadError \| CloseReadFile` | `list_directory`, `read_file`, `close_read_file`\* | Directory listing + open file content; rejects `.py` files |
| **Context** | `context.py` | N/A (compositor) | N/A | None | Composes: system prompt → conversation log → state panels (wrapped in `CURRENT STATE` markers), dispatches via `dispatch_llm()`/`dispatch_ui()`, detects changes via `changed_html()` |

\* = frontend-callable (invocable from the browser without LLM involvement via `dispatch_ui()`)

### Planned

| Widget | State | DSL | Renders |
|---|---|---|---|
| **Pytest** | Test results | None (reactive) | `TESTS PASSING` or failure details |
