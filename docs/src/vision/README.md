# Vision & Strategy

## The core idea

Everything in the context is a widget.

A widget is an Elm Architecture component (state, view, update) that renders into the model's context. Some widgets are trivial — a system prompt widget has no state and always renders the same text. Others are complex — a code explorer widget maintains a view of the codebase and accepts LSP-like commands to navigate it.

The agent's entire context is composed of widgets. There is no distinction between "the system prompt" and "a tool's output" — they are all widgets with different levels of complexity. The conversation itself is a widget that manages user/assistant turns and compacts old exchanges.

Each widget provides view functions — generators that yield messages (`view_messages()`) or tool definitions (`view_tools()`). Widgets compose via `yield from`, which naturally flattens nested iterators (List monad join). The root `Context` widget composes all children into the final prompt.

## The spectrum of widgets

### Static: System Prompt

The simplest widget. No state, no DSL. It always renders the same text into the prompt. This is the degenerate case that shows every prompt component is a widget.

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

Updates are triggered by the runner after relevant tool calls, not by DSL commands.

### Interactive: Code Explorer

Maintains a structural overview of the codebase — modules, classes, functions — sourced from an LSP. The agent navigates it via a DSL that is a subset of LSP commands (go to definition, list symbols, expand module). The widget re-renders the current view after each command.

This is how the agent understands the shape of the code without reading every file.

### Derived: Code Summary

Takes raw code (stripped of comments) and feeds it to a fast, cheap model that returns a compact description of what the code does. This is the primary way the agent learns about the specifics of code — not by reading source directly, but by reading summaries that fit the token budget.

The DSL controls which code is summarized (file, function, class) and at what level of detail.

### Organizational: Tasklist

A list of tasks with their statuses. The agent uses the DSL to create, update, and complete tasks. The widget renders the current list into the prompt so the agent always knows what's done and what remains.

### Directional: Goal

The current specific goal the agent is working toward. Keeps the agent focused across turns. The DSL allows updating the goal as sub-goals are completed or the plan changes.

### Conversational: Conversation

Manages the user/assistant message turns. Since conversation history can grow large and repetitive, this widget compacts older turns into dense summaries. The agent always has context on its recent work without re-reading the full history.

Compaction is a view decision — the widget always has the full turn state internally, but `view_messages()` renders older turns as summaries and recent turns in full.

## Why this model

The widget model unifies everything the agent sees into a single abstraction. Instead of a mix of system prompts, tool responses, message history, and injected context — each managed differently — there are just widgets. Each one:

1. Has state (possibly trivial)
2. Renders via view functions — generators yielding messages or tool definitions
3. Optionally accepts DSL commands via tools (which are themselves just another view)
4. Updates state when the runner dispatches tool calls

This makes the agent's context composable, auditable, and token-efficient. Widgets compete for prompt space and must earn their tokens through compact, high-signal rendering. Compaction is never a mutation of message history — it is a rendering decision made by each widget's view function.

## Design principles

- **Everything is a widget.** The system prompt, the test results, the code view, the task list, the conversation — all widgets with varying complexity.
- **Tools are views.** `view_tools()` returns `Iterator[ToolDefinition]`, composed the same way as messages. There is no separate mechanism for tools.
- **Composition via `yield from`.** Widgets nest and compose. A parent widget calls child views with `yield from`, which flattens one level. The root Context produces a single flat list of messages and tools.
- **View is a pure function of state.** Compaction is a rendering choice, not a history mutation. The widget always has full state; the view decides what to show.
- **DSLs constrain manipulation.** The agent issues structured commands, not arbitrary text. Each widget defines only the operations that make sense for it.
- **Minimal rendering.** Widgets render the least text needed to convey their state. Token budget is scarce; every character in the context must earn its place.
