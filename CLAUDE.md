# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Calipso

An AI coding agent built on [Pydantic AI](https://ai.pydantic.dev/) where **everything in the prompt is a capability** — a Pydantic AI `AbstractCapability` subclass with state, rendered text (`get_instructions()`), optional DSL commands (`get_toolset()`), and optional lifecycle hooks. The project is early-stage; only a `SystemPrompt` capability exists so far.

## Commands

All commands use `task` (go-task). The dev environment is a Nix flake activated by direnv (`.envrc`).

```bash
task setup              # uv sync
task tests:unit         # pytest (runs setup first)
task lint               # ruff check src/ tests/
task format             # ruff format src/ tests/
task format:check       # ruff format --check
task run                # run the agent (needs ANTHROPIC_API_KEY)
task docs:build         # build mdBook docs
task docs:serve         # live-reload docs preview
```

Run a single test:
```bash
uv run pytest tests/test_agent.py::test_agent_responds
```

## Architecture

- **Agent singleton** (`src/calipso/agent.py`): module-level `Agent` instance with `defer_model_check=True` so it imports without an API key. Tests override it with `agent.override(model=TestModel())` — never construct a new Agent.
- **CLI** (`src/calipso/cli.py`): minimal entry point calling `agent.run_sync()`. Registered as `calipso` console script.
- **Capabilities**: composable units that render into the agent's prompt. Use Pydantic AI's terminology — they are "capabilities" (not "widgets" or any other name). Each is an `AbstractCapability` subclass.

## Testing conventions

- `models.ALLOW_MODEL_REQUESTS = False` is set at module level in test files to prevent real API calls.
- Tests use `pytest-anyio` — test functions are `async def` with `pytestmark = pytest.mark.anyio`.
- Use `TestModel(custom_output_text=...)` for deterministic outputs.

## Infrastructure

- **Python 3.12+**, managed by **uv** (never use pip directly).
- **Nix flake** provides two devshells: `default` (local dev via direnv) and `ci` (minimal, used in GitHub Actions as `nix develop .#ci --command task ...`).
- **Ruff** for both linting and formatting (rules: E, F, I, W).
- **CI** runs format:check → lint → tests:unit → docs:build, with path-based filtering to skip unchanged areas.
