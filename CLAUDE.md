# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Calipso

A context engineering library and AI coding agent where **everything in the context is a widget** — an Elm-inspired component with state, view functions (generators yielding messages or tool definitions via `view_messages()` / `view_tools()`), and update handlers. Uses [Pydantic AI](https://ai.pydantic.dev/)'s `Model` layer for provider-agnostic LLM calls but owns the agentic loop and prompt composition. Widgets compose via `yield from` (List monad join); the root `Context` widget produces the final prompt.

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
- **Runner** (`src/calipso/runner.py`): thin agentic loop. Materializes context views, calls `Model.request()`, dispatches tool calls.
- **CLI** (`src/calipso/cli.py`): REPL entry point. Creates model, assembles widget tree into a `Context`, runs the turn loop. Registered as `calipso` console script.
- **Widgets** (`src/calipso/widgets/`): composable units that render into the model's context. They are "widgets" (Elm Architecture: state/view/update). Each is a `Widget` subclass with `view_messages()`, `view_tools()`, and `update()`. The root `Context` widget composes all children via `yield from`.

## Testing conventions

- `models.ALLOW_MODEL_REQUESTS = False` is set at module level in test files to prevent real API calls.
- Tests use `pytest-anyio` — test functions are `async def` with `pytestmark = pytest.mark.anyio`.
- Use `TestModel(custom_output_text=...)` or `FunctionModel` for deterministic outputs.
- Widget tests are synchronous (test view rendering and update logic directly). Runner tests are async.

## Infrastructure

- **Python 3.12+**, managed by **uv** (never use pip directly).
- **Nix flake** provides two devshells: `default` (local dev via direnv) and `ci` (minimal, used in GitHub Actions as `nix develop .#ci --command task ...`).
- **Ruff** for both linting and formatting (rules: E, F, I, W).
- **CI** runs format:check → lint → tests:unit → docs:build, with path-based filtering to skip unchanged areas.
