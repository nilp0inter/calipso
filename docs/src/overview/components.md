# Components

## Model

**Source:** `src/calipso/model.py`

Configures the Pydantic AI `Model` instance for provider-agnostic LLM communication. Uses `OpenAIChatModel` with an OpenRouter provider (currently `x-ai/grok-4-fast`). Exposes `create_model()` and `create_http_client()` for HTTP-level instrumentation.

## Summarizer

**Source:** `src/calipso/summarizer.py`

A Pydantic AI agent that takes Python code (with comments already stripped) and produces a summary preserving signatures verbatim while replacing code bodies with `[...REDACTED...]` followed by a description. Uses a cheap, fast model (`google/gemini-3.1-flash-lite-preview` via OpenRouter) to keep costs low. Called by the `CodeExplorer` widget after every tree-sitter query.

## Runner

**Source:** `src/calipso/runner.py`

The agentic loop. Takes a `Model` and a `Context` widget, then:
1. Materializes the context's views (`view_messages()`, `view_tools()`)
2. Calls `Model.request()` directly
3. Dispatches tool calls back to the context
4. Calls an optional `on_update` callback after every state mutation (used by the dashboard)
5. Loops until the model produces a text response

## Dashboard Server

**Source:** `src/calipso/server.py`

An aiohttp HTTP + WebSocket server that provides a live browser dashboard. Serves the SPA at `/` and accepts WebSocket connections at `/ws`. On connection, sends all widget HTML. After every state mutation, pushes only changed widget fragments via htmx out-of-band swaps. Also handles turn lifecycle signals (thinking indicator, input disabling).

The server handles two types of inbound WebSocket messages: `user_input` (enqueued for the runner's main loop) and `widget_event` (dispatched directly to the target widget's `update()` method via `Context.handle_widget_event()`, bypassing the LLM and action log protocol). After a widget event, changed HTML is pushed immediately.

## CLI

**Source:** `src/calipso/cli.py`

An async entry point registered as the `calipso` console script in `pyproject.toml`. Creates the model, assembles the widget tree into a `Context`, starts the `DashboardServer`, and loops reading user input from the server's WebSocket queue.

Each turn's raw HTTP request/response payloads are saved to `prompts/NNNN.json`, captured via httpx event hooks on the underlying HTTP client.

## SPA

**Source:** `src/calipso/static/index.html`

A single-page htmx application served by the dashboard server. Uses the htmx WebSocket extension to connect to `/ws`. The layout follows a VS Code-style pattern: a narrow activity bar on the left with icon tabs (one per non-conversation widget), a collapsible side panel (320px) that shows the selected widget's content, and a chat area (ConversationLog + input bar) filling the remaining space. Each widget div lives inside a stable wrapper container; the server's `hx-swap-oob` updates replace the inner widget div without disturbing the wrapper's visibility state. Activity bar tabs show a blue notification badge when their widget receives an update while not currently active (the first WebSocket message — initial state — is excluded). A thinking indicator with animated dots appears during turns while the input is disabled. A global `sendWidgetEvent(toolName, args)` JS function sends `widget_event` messages over the WebSocket, allowing widgets to render interactive HTML (buttons, checkboxes, inputs) that trigger their own updates without an LLM round-trip.

## Widgets

Everything in the agent's context is a widget — an Elm-inspired component with state, view functions (generators yielding messages or tool definitions, plus HTML for the browser dashboard), and update handlers. Widgets compose via `yield from`. Each widget also has a `widget_id()` (stable kebab-case HTML ID), a `view_html()` method that renders its state as markdown-to-HTML via the shared `render_md()` function, and an optional `frontend_tools()` method that declares which tools can be invoked directly from the browser (default: none).

**Base class:** `src/calipso/widget.py`

### Implemented

| Widget | Source | State | DSL (tools) | Renders |
|---|---|---|---|---|
| **SystemPrompt** | `src/calipso/widgets/system_prompt.py` | None | None | Identity and workspace framing text |
| **AgentsMd** | `src/calipso/widgets/agents_md.py` | None | None | Behavioral instructions loaded from `AGENTS.md` on disk (silently skipped if missing) |
| **Goal** | `src/calipso/widgets/goal.py` | Current objective text | `set_goal`, `clear_goal` (both frontend-callable) | Goal text as `## Goal` panel with inline edit input and clear button |
| **TaskList** | `src/calipso/widgets/task_list.py` | Tasks with statuses (`pending`, `in_progress`, `done`) | `create_task`, `update_task_status`\*, `remove_task`\* | Compact checklist as `## Tasks` panel with interactive checkboxes and remove buttons |
| **ConversationLog** | `src/calipso/widgets/conversation_log.py` | Turns with segmented messages + protocol state | `action_log_start`, `action_log_end` | Action protocol rules + conversation history; summarized segments render summary + tool call/return messages, unsummarized render full messages |
| **CodeExplorer** | `src/calipso/widgets/code_explorer.py` | Open files with cached parse trees, query results per file | `open_file`, `close_file`\*, `query`, `query_all` | Open files list + tree-sitter query results (signatures + `[...REDACTED...]` body summaries) |
| **FileExplorer** | `src/calipso/widgets/file_explorer.py` | Current directory listing, open file path + content | `list_directory`, `read_file`, `close_read_file`\* | Directory listing + open file content; rejects `.py` files (handled by CodeExplorer) |
| **Context** | `src/calipso/widgets/context.py` | system_prompt + children (state panels) + conversation_log + HTML cache | None | Composes: system prompt first, conversation log second, state panels last (wrapped in `CURRENT STATE` markers as user messages), dispatches tool calls and frontend widget events, detects changed widgets via `changed_html()` |

\* = frontend-callable (invocable from the browser without LLM involvement)

### Planned

| Widget | State | DSL | Renders |
|---|---|---|---|
| **Pytest** | Test results | None (reactive) | `TESTS PASSING` or failure details |
