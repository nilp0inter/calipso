# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Calipso

A context engineering library and AI coding agent where **everything in the context is a widget** — an Elm-inspired component with a frozen model (dataclass), a pure update function (`(model, msg) -> (model, result_str)`), and free view functions (generators yielding messages or tool definitions, plus HTML). Two anticorruption layers (`from_llm` and `from_ui`) convert external events into typed Msg values before update. Widgets are created via factory functions that return a `WidgetHandle` — the uniform interface holding a model reference + function table. Uses [Pydantic AI](https://ai.pydantic.dev/)'s `Model` layer for provider-agnostic LLM calls but owns the agentic loop and prompt composition. Views compose via `yield from` (List monad join); the root `Context` produces the final prompt. A live browser dashboard (htmx SPA over WebSocket) visualizes widget state in real time and serves as the user input channel.

## Commands

All commands use `task` (go-task). The dev environment is a Nix flake activated by direnv (`.envrc`).

```bash
task setup              # uv sync
task tests:unit         # pytest (runs setup first)
task lint               # ruff check src/ tests/
task format             # ruff format src/ tests/
task format:check       # ruff format --check
task run                # run the agent (needs OPENROUTER_API_KEY)
task docs:build         # build mdBook docs
task docs:serve         # live-reload docs preview
```

Run a single test:
```bash
uv run pytest tests/test_widgets.py::TestConversationLog
```

## Architecture

- **Model setup** (`src/calipso/model.py`): configures the Pydantic AI `Model` instance (OpenRouter provider). No `Agent` — we call `Model.request()` directly.
- **Summarizer** (`src/calipso/summarizer.py`): Pydantic AI `Agent` using a cheap model (`google/gemini-3.1-flash-lite-preview`) that takes comment-stripped code and returns signatures with `[...REDACTED...]` body descriptions. Used by the CodeExplorer widget.
- **Runner** (`src/calipso/runner.py`): thin agentic loop. Materializes context views, calls `Model.request()`, dispatches tool calls. Accepts an `on_update` callback for pushing live updates.
- **Dashboard server** (`src/calipso/server.py`): aiohttp HTTP + WebSocket server. Serves the htmx SPA at `/`, pushes per-widget HTML diffs via `hx-swap-oob` over WebSocket, receives user input and frontend widget events.
- **SPA** (`src/calipso/static/index.html`): htmx single-page app with WebSocket extension. VS Code-style layout: a narrow activity bar on the left with icon tabs (one per widget), a collapsible side panel showing the selected widget, and a chat area (ConversationLog + input) occupying the remaining space. Tabs show notification badges when their widget receives updates while not active. No build step. Exposes `sendWidgetEvent()` for browser-initiated widget updates.
- **CLI** (`src/calipso/cli.py`): async entry point. Creates model, assembles widget tree, starts `DashboardServer`, reads input from WebSocket queue. Registered as `calipso` console script.
- **Widgets** (`src/calipso/widgets/`): composable units that render into the model's context and the browser. Each widget module defines: a frozen `Model` dataclass, a `Msg` union (frozen dataclasses), a pure `update(model, msg) -> (model, str)` function, view functions (`view_messages(model)`, `view_tools(model)`, `view_html(model)`), and two anticorruption layers — `from_llm` (async, converts LLM tool calls to Msgs, may do I/O) and `from_ui` (sync, converts browser events to Msgs). A `create_xxx()` factory function returns a `WidgetHandle`. I/O resources (e.g., CodeExplorer's tree-sitter parser and summarizer agent) are captured in `from_llm` closures, not stored in the model. HTML is rendered via `render_md()` (markdown to safe HTML). The `WidgetHandle` exposes `dispatch_llm()`, `dispatch_ui()`, `send(msg)` (direct Msg dispatch), and `.model` (read access). The root `Context` composes all children via `yield from`, dispatches tool calls via `dispatch_llm()`, tracks HTML changes via `changed_html()`, and routes frontend events via `handle_widget_event()` (sync).

## Testing conventions

- `models.ALLOW_MODEL_REQUESTS = False` is set at module level in test files to prevent real API calls.
- Tests use `pytest-anyio` — test functions are `async def` with `pytestmark = pytest.mark.anyio`.
- Use `TestModel(custom_output_text=...)` or `FunctionModel` for deterministic outputs.
- Pure `update` functions can be tested directly: `model, result = update(GoalModel(), SetGoal(goal="x"))`. Handle-level tests use `dispatch_llm()` (async) or `dispatch_ui()` (sync). View-only tests call view functions directly on models (synchronous). Access widget state via `handle.model`.

## Infrastructure

- **Python 3.12+**, managed by **uv** (never use pip directly).
- **Nix flake** provides two devshells: `default` (local dev via direnv) and `ci` (minimal, used in GitHub Actions as `nix develop .#ci --command task ...`).
- **Ruff** for both linting and formatting (rules: E, F, I, W).
- **CI** runs format:check → lint → tests:unit → docs:build, with path-based filtering to skip unchanged areas.
