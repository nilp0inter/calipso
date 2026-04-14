"""ConversationLog widget — manages conversation turns with action log protocol.

Merges the old ActionLog and Conversation widgets. Messages are partitioned
into segments. Each segment has an optional model-provided summary: summarized
segments render as their summary text, unsummarized segments render their full
messages. This prevents replaying raw tool call/result messages for completed
actions.
"""

import html as html_mod
from collections.abc import Iterator
from dataclasses import dataclass, field

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.tools import ToolDefinition

from calipso.widget import Widget, render_md


@dataclass
class Segment:
    """A contiguous run of messages, optionally summarized."""

    messages: list[ModelMessage] = field(default_factory=list)
    summary: str | None = None


@dataclass
class Turn:
    """A single conversational exchange starting with a user message."""

    user_message: str
    segments: list[Segment] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.segments:
            self.segments.append(Segment())


_RULES = (
    "## Action Protocol\n"
    "Every tool use must be wrapped in an action:\n"
    "1. Call action_log_start with a description of what you will do.\n"
    "2. Either call your tool(s) (one tool type per action, may repeat)"
    " OR respond to the user with text.\n"
    "3. Call action_log_end with a summary of what happened.\n"
    "Calling tools outside an action will be rejected.\n"
    "Empty actions (start then immediately end) will be rejected."
)


@dataclass
class ConversationLog(Widget):
    turns: list[Turn] = field(default_factory=list)
    _active_action: str | None = field(init=False, repr=False, default=None)
    _allowed_tool: str | None = field(init=False, repr=False, default=None)
    _action_tool_count: int = field(init=False, repr=False, default=0)
    _tool_defs: list[ToolDefinition] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._tool_defs = [
            ToolDefinition(
                name="action_log_start",
                description=(
                    "Start a new action log entry. Call this before executing any tool."
                ),
                parameters_json_schema={
                    "type": "object",
                    "properties": {
                        "description": {
                            "type": "string",
                            "description": (
                                "What you are about to do, in imperative mood."
                            ),
                        },
                    },
                    "required": ["description"],
                },
            ),
            ToolDefinition(
                name="action_log_end",
                description="Finish the current action log entry.",
                parameters_json_schema={
                    "type": "object",
                    "properties": {
                        "result": {
                            "type": "string",
                            "description": "A summary of what happened.",
                        },
                    },
                    "required": ["result"],
                },
            ),
        ]

    def _current_segment(self) -> Segment:
        return self.turns[-1].segments[-1]

    def add_user_message(self, text: str) -> None:
        """Start a new turn with a user message."""
        self.turns.append(Turn(user_message=text))

    def add_response(self, response: ModelResponse, segment: Segment) -> None:
        """Record a model response in the given segment."""
        segment.messages.append(response)

    def add_tool_results(self, request: ModelRequest, segment: Segment) -> None:
        """Record tool return results in the given segment."""
        segment.messages.append(request)

    def view_messages(self) -> Iterator[ModelMessage]:
        yield ModelRequest(parts=[SystemPromptPart(content=_RULES)])

        for turn in self.turns:
            yield ModelRequest(parts=[UserPromptPart(content=turn.user_message)])
            for segment in turn.segments:
                if segment.summary is not None:
                    yield ModelRequest(
                        parts=[SystemPromptPart(content=segment.summary)]
                    )
                else:
                    yield from segment.messages

    def view_tools(self) -> Iterator[ToolDefinition]:
        yield from self._tool_defs

    def view_html(self) -> str:
        if not self.turns:
            content = "<p><em>No messages yet</em></p>"
        else:
            parts = []
            for turn in self.turns:
                user_html = render_md(turn.user_message)
                parts.append(
                    f'<div class="msg user"><strong>You:</strong> {user_html}</div>'
                )
                for segment in turn.segments:
                    if segment.summary is not None:
                        parts.append(
                            f'<div class="msg summary">'
                            f"{html_mod.escape(segment.summary)}</div>"
                        )
                    else:
                        for msg in segment.messages:
                            parts.extend(self._render_message(msg))
            content = "".join(parts)
        return (
            f'<div id="{self.widget_id()}" class="widget">'
            f"<h3>Conversation</h3>{content}</div>"
        )

    @staticmethod
    def _render_message(msg: ModelMessage) -> list[str]:
        parts: list[str] = []
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, TextPart):
                    rendered = render_md(part.content)
                    parts.append(
                        f'<div class="msg assistant">'
                        f"<strong>Calipso:</strong> {rendered}"
                        f"</div>"
                    )
                elif isinstance(part, ToolCallPart):
                    args = html_mod.escape(str(part.args_as_dict()))
                    parts.append(
                        f'<div class="msg tool-call">'
                        f"<code>{html_mod.escape(part.tool_name)}({args})</code>"
                        f"</div>"
                    )
        elif isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, ToolReturnPart):
                    parts.append(
                        f'<div class="msg tool-result">'
                        f"<code>→ {html_mod.escape(str(part.content))}</code>"
                        f"</div>"
                    )
        return parts

    def check_protocol(self, tool_name: str) -> str | None:
        """Check if a tool call is allowed under the action log protocol.

        Returns an error message if the call violates the protocol, or None
        if it's allowed.
        """
        if tool_name == "action_log_start":
            return None
        if tool_name == "action_log_end":
            if self._active_action is None:
                return (
                    "Cannot call action_log_end without an active action. "
                    "Call action_log_start first."
                )
            if self._action_tool_count == 0:
                return (
                    "Cannot call action_log_end without doing anything. "
                    "Either call a tool or respond to the user first, "
                    "then end the action."
                )
            return None
        if self._active_action is None:
            return (
                f"Cannot execute '{tool_name}' without an active action log entry. "
                "Call action_log_start first."
            )
        if self._allowed_tool is None:
            return None
        if tool_name != self._allowed_tool:
            return (
                f"Cannot execute '{tool_name}' during this action log entry. "
                f"Only '{self._allowed_tool}' is allowed until you call action_log_end."
            )
        return None

    def track_tool(self, tool_name: str) -> None:
        """Track that a non-action-log tool was called (locks the allowed tool)."""
        if self._active_action is not None:
            self._action_tool_count += 1
            if self._allowed_tool is None:
                self._allowed_tool = tool_name

    def update(self, tool_name: str, args: dict) -> str:
        if tool_name == "action_log_start":
            self._active_action = args["description"]
            self._allowed_tool = None
            self._action_tool_count = 0
            return f"Action started: {args['description']}"

        if tool_name == "action_log_end":
            # Build a structured summary and assign it to the current segment
            summary = (
                f"[Action: {self._active_action}] "
                f"Called {self._action_tool_count} tool(s). "
                f"Result: {args['result']}"
            )
            if self.turns:
                self._current_segment().summary = summary
                self.turns[-1].segments.append(Segment())
            self._active_action = None
            self._allowed_tool = None
            self._action_tool_count = 0
            return "Action logged."

        raise NotImplementedError(f"ConversationLog does not handle tool '{tool_name}'")
