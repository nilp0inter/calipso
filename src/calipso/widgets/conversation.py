"""Conversation widget — manages user/assistant message turns with compaction."""

from collections.abc import Iterator
from dataclasses import dataclass, field

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)

from calipso.widget import Widget


@dataclass
class Turn:
    """A single conversational exchange."""

    user_message: str
    messages: list[ModelMessage] = field(default_factory=list)


@dataclass
class Conversation(Widget):
    turns: list[Turn] = field(default_factory=list)
    compaction_summaries: list[str] = field(default_factory=list)
    compaction_threshold: int = 10

    def add_user_message(self, text: str) -> None:
        """Start a new turn with a user message."""
        self.turns.append(Turn(user_message=text))

    def add_response(self, response: ModelResponse) -> None:
        """Record a model response in the current turn."""
        if self.turns:
            self.turns[-1].messages.append(response)

    def add_tool_results(self, request: ModelRequest) -> None:
        """Record tool return results in the current turn."""
        if self.turns:
            self.turns[-1].messages.append(request)

    def view_messages(self) -> Iterator[ModelMessage]:
        for summary in self.compaction_summaries:
            yield ModelRequest(
                parts=[SystemPromptPart(content=f"[Earlier context] {summary}")]
            )

        for turn in self.turns:
            yield ModelRequest(parts=[UserPromptPart(content=turn.user_message)])
            yield from turn.messages

    def _compact_if_needed(self) -> None:
        """Compact older turns into summaries when threshold is exceeded."""
        if len(self.turns) <= self.compaction_threshold:
            return

        turns_to_compact = len(self.turns) - self.compaction_threshold
        old_turns = self.turns[:turns_to_compact]

        lines = []
        for turn in old_turns:
            lines.append(f"User: {turn.user_message}")
            for msg in turn.messages:
                if isinstance(msg, ModelResponse):
                    for part in msg.parts:
                        if isinstance(part, TextPart) and part.content:
                            lines.append(f"Assistant: {part.content[:200]}")
                        elif isinstance(part, ToolCallPart):
                            lines.append(f"  Tool: {part.tool_name}")

        self.compaction_summaries.append("\n".join(lines))
        self.turns = self.turns[turns_to_compact:]
