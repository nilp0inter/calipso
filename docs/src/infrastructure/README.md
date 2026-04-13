# Infrastructure

Calipso uses three pillars for reproducible development and CI:

1. **Nix flake** — provides all tools via a two-tier devShell (CI-minimal and full dev)
2. **go-task** — all automation is defined in `Taskfile.yml`
3. **GitHub Actions** — CI runs tasks inside the Nix CI shell

These pillars reinforce each other: the flake provides all tools, the Taskfile orchestrates them, and CI runs `nix develop .#ci --command task <name>` for every step.

See the individual pages for details:

- [Nix & Dev Environment](./nix.md)
- [Taskfile Commands](./taskfile.md)
- [CI / GitHub Actions](./ci.md)
