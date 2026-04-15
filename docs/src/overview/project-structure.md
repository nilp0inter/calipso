# Project Structure

```
calipso/
├── src/calipso/              # Python package
│   ├── widget.py             # Widget base class / protocol + render_md()
│   ├── runner.py             # Agentic loop (Model.request())
│   ├── model.py              # Model/provider setup
│   ├── summarizer.py         # Code summarizer agent (cheap LLM)
│   ├── server.py             # DashboardServer (aiohttp HTTP + WebSocket)
│   ├── cli.py                # Async CLI entry point
│   ├── static/               # Browser SPA assets
│   │   └── index.html        # htmx dashboard (served by DashboardServer)
│   └── widgets/              # Widget modules
│       ├── system_prompt.py  # Identity + workspace framing
│       ├── agents_md.py      # Behavioral instructions from AGENTS.md
│       ├── goal.py           # Goal widget
│       ├── task_list.py      # TaskList widget
│       ├── code_explorer.py  # Tree-sitter code reading + summarization
│       ├── file_explorer.py  # Filesystem navigation + non-Python file reading
│       ├── conversation_log.py # Conversation + step protocol
│       └── context.py        # Root context widget (composes all)
├── tests/                    # Test suite (pytest + anyio)
├── docs/src/                 # mdBook documentation source
├── AGENTS.md                 # Agent behavioral instructions (loaded by AgentsMd widget)
├── flake.nix                 # Nix devShell (CI + dev tiers)
├── Taskfile.yml              # Task automation
├── pyproject.toml            # Python project metadata and dependencies
└── .github/workflows/        # CI pipeline
```

## Why it's organized this way

- **`src/` layout** — the Python package lives under `src/calipso/` to avoid import confusion between the package and the repo root. This is the standard layout recommended by the Python packaging ecosystem.
- **Tests outside the package** — `tests/` is a top-level directory, not inside `src/`. Tests are not shipped with the package.
- **`widgets/` subdirectory** — each widget lives in its own module under `src/calipso/widgets/`. The `Context` widget (root composition) lives here alongside leaf widgets. `widget.py` at the package root defines the base class.
- **`runner.py`, `model.py`, and `summarizer.py` at the package root** — the runner (agentic loop), model setup, and code summarizer are thin modules that don't warrant their own subdirectory. The runner depends on the Context widget; `model.py` configures the Pydantic AI `Model` instance; `summarizer.py` creates a cheap Pydantic AI agent for code body summarization.
- **`server.py` at the package root** — the `DashboardServer` (aiohttp) serves the browser SPA and manages WebSocket connections for live widget visualization and user input.
- **`static/` directory** — contains the single-page htmx application (`index.html`). No build step; htmx and extensions are loaded from CDN.
