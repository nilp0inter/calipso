# Calipso

Calipso is an AI coding agent that uses **capabilities** — small stateful programs whose output is rendered directly into the agent's prompt. The agent manipulates capabilities via domain-specific languages (DSLs), and the capabilities' current state is always visible to the agent without explicit queries.

Built on [Pydantic AI](https://ai.pydantic.dev/), Calipso runs as a CLI application (`calipso`) backed by Anthropic's Claude. Each capability is a Pydantic AI `AbstractCapability` subclass that provides instructions, tools, and lifecycle hooks.

## Current status

The project is in its early stages. The agent responds to a simple greeting prompt. The DSL-bearing capabilities are not yet implemented.

## Audience

| Reader | Start here |
|---|---|
| New contributor | [Architecture Overview](./overview/README.md), then [Project Structure](./overview/project-structure.md) |
| Operator / DevOps | [Infrastructure](./infrastructure/README.md) |

## Quick start

```bash
# Enter the dev shell (or use direnv)
nix develop

# Install dependencies
task setup

# Run the agent (requires ANTHROPIC_API_KEY)
task run

# Run tests
task tests:unit
```
