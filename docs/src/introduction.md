# Calipso

Calipso is a context engineering library and AI coding agent. It uses **widgets** — Elm-inspired stateful components whose output is rendered directly into the model's context. The agent manipulates widgets via domain-specific languages (DSLs), and the widgets' current state is always visible to the agent without explicit queries.

Built on [Pydantic AI](https://ai.pydantic.dev/)'s `Model` layer for provider-agnostic LLM communication, Calipso owns the agentic loop and prompt composition. Widgets compose via `yield from` (List monad join), and the root `Context` widget produces the final flat prompt the model sees.

## Current status

The agent has a CLI entry point, a runner (agentic loop), and eight widgets: `SystemPrompt`, `AgentsMd`, `Goal`, `ConversationLog` (manages tasks *and* conversation history together), `CodeExplorer`, `FileExplorer`, `TestSuite`, and `TokenUsage`. The widget architecture is functional and tested. The `CodeExplorer` widget provides tree-sitter-based code reading with LLM-powered summarization — the agent sees signatures and descriptions, never raw code. The `ConversationLog` enforces a **task protocol**: every non-task, non-goal tool call must happen while a task is `in_progress`, and when the LLM closes a task, the full span collapses to the task's LLM-authored memory list (reclaiming prompt budget while keeping the durable findings).

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

# Run the agent (requires OPENROUTER_API_KEY)
task run

# Run tests
task tests:unit
```
