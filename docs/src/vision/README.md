# Vision & Strategy

## The core idea

Everything in the prompt is a capability.

A capability is a program with internal state that renders text into the agent's prompt. Some capabilities are trivial — a system prompt capability has no state and always renders the same text. Others are complex — a code explorer capability maintains a view of the codebase and accepts LSP-like commands to navigate it.

The agent's entire context is composed of capabilities. There is no distinction between "the system prompt" and "a tool's output" — they are all capabilities with different levels of complexity.

Each capability is a Pydantic AI `AbstractCapability` subclass. It provides instructions via `get_instructions()`, tools via `get_toolset()`, and reacts to events via lifecycle hooks.

## The spectrum of capabilities

### Static: System Prompt

The simplest capability. No state, no DSL. It always renders the same text into the prompt. This is the degenerate case that shows every prompt component is a capability.

### Reactive: Pytest

Has state (test results) but the agent doesn't directly control when it updates. It renders compactly:

```
TESTS PASSING
```

or:

```
TESTS FAILING:
test_login_redirects - AssertionError: expected 302, got 200
```

Updates are triggered by lifecycle hooks (e.g. `after_tool_execute`) rather than DSL commands.

### Interactive: Code Explorer

Maintains a structural overview of the codebase — modules, classes, functions — sourced from an LSP. The agent navigates it via a DSL that is a subset of LSP commands (go to definition, list symbols, expand module). The capability re-renders the current view after each command.

This is how the agent understands the shape of the code without reading every file.

### Derived: Code Summary

Takes raw code (stripped of comments) and feeds it to a fast, cheap model that returns a compact description of what the code does. This is the primary way the agent learns about the specifics of code — not by reading source directly, but by reading summaries that fit the token budget.

The DSL controls which code is summarized (file, function, class) and at what level of detail.

### Organizational: Tasklist

A list of tasks with their statuses. The agent uses the DSL to create, update, and complete tasks. The capability renders the current list into the prompt so the agent always knows what's done and what remains.

### Directional: Goal

The current specific goal the agent is working toward. Keeps the agent focused across turns. The DSL allows updating the goal as sub-goals are completed or the plan changes.

### Contextual: Short-term Memory

A textual, compacted summary of what the agent just did. Since conversation history can grow large and repetitive, this capability distills recent actions into a dense summary that stays in the prompt. The agent always has context on its recent work without re-reading the full history.

Updated automatically via `after_model_request` hooks rather than DSL commands.

## Why this model

The capability model unifies everything the agent sees into a single abstraction. Instead of a mix of system prompts, tool responses, message history, and injected context — each managed differently — there are just capabilities. Each one:

1. Has state (possibly trivial)
2. Renders text into the prompt via `get_instructions()`
3. Optionally accepts DSL commands via `get_toolset()`
4. Optionally reacts to events via lifecycle hooks

This makes the agent's context composable, auditable, and token-efficient. Capabilities compete for prompt space and must earn their tokens through compact, high-signal rendering.

## Design principles

- **Everything is a capability.** The system prompt, the test results, the code view, the task list — all capabilities with varying complexity.
- **DSLs constrain manipulation.** The agent issues structured commands, not arbitrary text. Each capability defines only the operations that make sense for it.
- **Minimal rendering.** Capabilities render the least text needed to convey their state. Token budget is scarce; every character in the prompt must earn its place.
- **Composability.** Multiple capabilities coexist in a prompt. A coding session might show a goal, a code explorer, a pytest capability, a tasklist, and short-term memory simultaneously.
