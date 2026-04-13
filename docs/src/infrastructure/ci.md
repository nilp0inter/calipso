# CI / GitHub Actions

A single workflow (`.github/workflows/ci.yml`) runs all checks in one job on `ubuntu-latest`.

## Trigger

Pushes to `main` that touch source code, tests, docs, project config, or CI config itself. Path filtering avoids running CI on irrelevant commits.

## Step order

Steps run in fast-fail order — cheapest checks first:

1. **Check formatting** — `task format:check`
2. **Lint** — `task lint`
3. **Unit tests** — `task tests:unit`
4. **Build docs** — `task docs:build` (only when `docs/` changes)

Code steps are gated on a `code` path filter; docs steps on a `docs` filter. Both use `dorny/paths-filter`.

## Nix in CI

- Nix is installed via `DeterminateSystems/determinate-nix-action`
- The store is cached with `nix-community/cache-nix-action` (8 GB max, keyed on `flake.lock` hash)
- A prefetch step (`nix develop .#ci --command true`) warms the shell so noisy store-path output doesn't pollute task logs
- Every real step runs as `nix develop .#ci --command task <name>`
