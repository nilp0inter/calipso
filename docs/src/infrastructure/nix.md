# Nix & Dev Environment

## Two-tier devShell

The `flake.nix` defines two shells:

| Shell | Invocation | Purpose |
|---|---|---|
| `ci` | `nix develop .#ci` | Minimal set for CI — keeps the Nix cache closure small |
| `default` | `nix develop` or `direnv allow` | Extends CI shell with dev-only tools |

Both shells share a `shellHook` that sets `LD_LIBRARY_PATH` to include `libstdc++`, which is required by pydantic-core's native extensions on NixOS.

## Packages

### CI packages (always available)

| Package | Role |
|---|---|
| `python312` | Python 3.12 runtime |
| `uv` | Python package manager |
| `ruff` | Linter and formatter |
| `go-task` | Task runner |
| `mdbook` | Documentation builder |

### Dev-only packages

None yet. As the project grows, dev-only tools (e.g., LSP servers, debugging tools) will be added here.

## Local setup

```bash
# Option A: direnv (recommended)
direnv allow

# Option B: manual
nix develop
```

After entering the shell, run `task setup` to install Python dependencies into the `.venv/`.
