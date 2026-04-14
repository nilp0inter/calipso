# Project Structure

```
calipso/
├── src/calipso/              # Python package
│   ├── widget.py             # Widget base class / protocol
│   ├── runner.py             # Agentic loop (Model.request())
│   ├── model.py              # Model/provider setup
│   ├── cli.py                # CLI entry point
│   └── widgets/              # Widget modules
│       ├── system_prompt.py  # Static text widget
│       ├── goal.py           # Goal widget
│       ├── task_list.py      # TaskList widget
│       ├── action_log.py     # ActionLog widget
│       ├── conversation.py   # Conversation widget
│       └── context.py        # Root context widget (composes all)
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
- **`widgets/` subdirectory** — each widget lives in its own module under `src/calipso/widgets/`. The `Context` widget (root composition) lives here alongside leaf widgets. `widget.py` at the package root defines the base class.
- **`runner.py` and `model.py` at the package root** — the runner (agentic loop) and model setup are thin modules that don't warrant their own subdirectory. The runner depends on the Context widget; `model.py` configures the Pydantic AI `Model` instance.
