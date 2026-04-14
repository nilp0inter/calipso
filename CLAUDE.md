# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Calipso

A context engineering library and AI coding agent where **everything in the context is a widget** — an Elm-inspired component with state, view functions (generators yielding messages or tool definitions via `view_messages()` / `view_tools()`, plus HTML via `view_html()`), and update handlers. Uses [Pydantic AI](https://ai.pydantic.dev/)'s `Model` layer for provider-agnostic LLM calls but owns the agentic loop and prompt composition. Widgets compose via `yield from` (List monad join); the root `Context` widget produces the final prompt. A live browser dashboard (htmx SPA over WebSocket) visualizes widget state in real time and serves as the user input channel.

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
- **SPA** (`src/calipso/static/index.html`): htmx single-page app with WebSocket extension. Three-column grid: sidebar for state widgets, code panel (FileExplorer + CodeExplorer), main area for conversation with input bar. No build step. Exposes `sendWidgetEvent()` for browser-initiated widget updates.
- **CLI** (`src/calipso/cli.py`): async entry point. Creates model, assembles widget tree, starts `DashboardServer`, reads input from WebSocket queue. Registered as `calipso` console script.
- **Widgets** (`src/calipso/widgets/`): composable units that render into the model's context and the browser. Each is a `Widget` subclass with `view_messages()`, `view_tools()`, `view_html()`, `async update()`, and `frontend_tools()`. The `update()` method is async to support widgets that need LLM calls (e.g., CodeExplorer's summarizer). HTML is rendered via `render_md()` (markdown to safe HTML). Widgets can declare frontend-callable tools (via `frontend_tools()`) that the browser can invoke directly without LLM involvement. The root `Context` widget composes all children via `yield from`, tracks HTML changes via `changed_html()`, and routes frontend events via `handle_widget_event()`.

## Testing conventions

- `models.ALLOW_MODEL_REQUESTS = False` is set at module level in test files to prevent real API calls.
- Tests use `pytest-anyio` — test functions are `async def` with `pytestmark = pytest.mark.anyio`.
- Use `TestModel(custom_output_text=...)` or `FunctionModel` for deterministic outputs.
- Widget tests that call `update()` are async (since `update()` is `async def`). View-only tests remain synchronous. Runner tests are async.

## Infrastructure

- **Python 3.12+**, managed by **uv** (never use pip directly).
- **Nix flake** provides two devshells: `default` (local dev via direnv) and `ci` (minimal, used in GitHub Actions as `nix develop .#ci --command task ...`).
- **Ruff** for both linting and formatting (rules: E, F, I, W).
- **CI** runs format:check → lint → tests:unit → docs:build, with path-based filtering to skip unchanged areas.
