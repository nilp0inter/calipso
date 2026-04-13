# Project Structure

```
calipso/
├── src/calipso/              # Python package
│   ├── agent.py              # Pydantic AI agent definition + SystemPrompt
│   ├── cli.py                # CLI entry point
│   └── capabilities/         # Capability modules
│       ├── task_list.py      # TaskList capability
│       └── goal.py           # Goal capability
├── tests/                    # Test suite (pytest + anyio)
├── docs/src/                 # mdBook documentation source
├── flake.nix                 # Nix devShell (CI + dev tiers)
├── Taskfile.yml              # Task automation
├── pyproject.toml            # Python project metadata and dependencies
└── .github/workflows/        # CI pipeline
```

## Why it's organized this way

- **`src/` layout** — the Python package lives under `src/calipso/` to avoid import confusion between the package and the repo root. This is the standard layout recommended by the Python packaging ecosystem.
- **Tests outside the package** — `tests/` is a top-level directory, not inside `src/`. Tests are not shipped with the package.
- **`capabilities/` subdirectory** — each capability lives in its own module under `src/calipso/capabilities/`. The `SystemPrompt` capability remains in `agent.py` since it is the core prompt with no tools or state, but DSL-bearing capabilities get their own files to keep the agent module lean.
