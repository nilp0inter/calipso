# Taskfile Commands

All automation is in `Taskfile.yml` at the repo root. Run `task --list` to see available tasks.

## Available tasks

| Task | Description |
|---|---|
| `task setup` | Install Python dependencies via `uv sync` |
| `task format` | Auto-format `src/` and `tests/` with ruff |
| `task format:check` | Check formatting without modifying (used in CI) |
| `task lint` | Lint `src/` and `tests/` with ruff |
| `task tests:unit` | Run pytest (depends on `setup`) |
| `task docs:build` | Build mdBook documentation to `docs/book/` |
| `task docs:serve` | Serve docs locally with live reload |
| `task run` | Run the calipso agent (depends on `setup`) |

## Conventions

- **`silent: true`** — go-task's own diagnostics are suppressed so only real command output appears.
- **GitHub Actions grouping** — each task's output is wrapped in `::group::` / `::endgroup::` markers, creating collapsible sections in CI logs.
- **`deps:` for prerequisites** — `tests:unit` and `run` depend on `setup`, ensuring dependencies are installed before execution.
