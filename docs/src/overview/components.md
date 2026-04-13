# Components

## Agent

**Source:** `src/calipso/agent.py`

The central component. A Pydantic AI `Agent` instance configured with:

- **Model:** `anthropic:claude-haiku-3-5`
- **`defer_model_check=True`** — allows the agent module to be imported without an API key present (important for testing)
- **`capabilities=[]`** — a list of `AbstractCapability` subclasses that provide instructions, tools, and hooks

The agent is a module-level singleton. Tests override it via `agent.override(model=TestModel())` rather than constructing a new one.

## CLI

**Source:** `src/calipso/cli.py`

A minimal entry point registered as the `calipso` console script in `pyproject.toml`. Calls `agent.run_sync()` and prints the output. No argument parsing yet.

## Capabilities

Everything in the agent's prompt is a capability — a Pydantic AI `AbstractCapability` subclass. A capability has state, renders text into the prompt via `get_instructions()`, and optionally accepts DSL commands via `get_toolset()`.

### Implemented

| Capability | Source | State | DSL | Renders |
|---|---|---|---|---|
| **System Prompt** | `src/calipso/agent.py` | None | None | Static instruction text |
| **TaskList** | `src/calipso/capabilities/task_list.py` | Tasks with statuses (`pending`, `in_progress`, `done`) | `create_task`, `update_task_status`, `remove_task` | Compact checklist with status icons (`[ ]`, `[~]`, `[x]`) |
| **Goal** | `src/calipso/capabilities/goal.py` | Current objective text | `set_goal`, `clear_goal` | The specific goal the agent is working toward |

### Planned

| Capability | State | DSL | Renders |
|---|---|---|---|
| **Pytest** | Test results | None (reactive via hooks) | `TESTS PASSING` or `TESTS FAILING:\n<first failure>` |
| **Code Explorer** | LSP-sourced structural view | Subset of LSP commands (go to definition, list symbols, expand module) | Module/class/function overview |
| **Code Summary** | Compact description from a cheap model | Select target (file, function, class) and detail level | Dense summary of what the code does |
| **Short-term Memory** | Compacted recent actions | None (auto-updated via hooks) | Dense summary of what the agent just did |
