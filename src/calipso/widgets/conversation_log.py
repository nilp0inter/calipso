"""ConversationLog widget — manages conversation turns with action log protocol.

Messages are partitioned into segments. Each segment has an optional
model-provided summary: summarized segments render as their summary text,
unsummarized segments render their full messages. This prevents replaying
raw tool call/result messages for completed actions.
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

from calipso.cmd import Cmd, none, tool_result
from calipso.widget import WidgetHandle, create_widget, render_md

# --- Supporting types ---


@dataclass
class Segment:
    """A contiguous run of messages, optionally summarized.

    Mutable by design — messages are appended during a turn,
    and summary is set when the action ends.
    """

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


# --- Model ---


@dataclass
class ConversationLogModel:
    """Model for the ConversationLog widget.

    Not frozen: Turn and Segment are mutable containers (lists of messages
    are appended during a turn). The update function mutates in place and
    returns the same model reference. This is a pragmatic choice — the
    conversation history is an append-only log where immutable copies
    would be wasteful.
    """

    turns: list[Turn] = field(default_factory=list)
    active_action: str | None = None
    allowed_tool: str | None = None
    action_tool_count: int = 0


# --- Messages ---


@dataclass(frozen=True)
class UserMessageReceived:
    text: str


@dataclass(frozen=True)
class ResponseReceived:
    response: ModelResponse
    segment: Segment


@dataclass(frozen=True)
class ToolResultsReceived:
    request: ModelRequest
    segment: Segment


@dataclass(frozen=True)
class ActionLogStart:
    description: str


@dataclass(frozen=True)
class ActionLogEnd:
    result: str


@dataclass(frozen=True)
class ToolTracked:
    tool_name: str


ConversationLogMsg = (
    UserMessageReceived
    | ResponseReceived
    | ToolResultsReceived
    | ActionLogStart
    | ActionLogEnd
    | ToolTracked
)


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


# --- Update ---


def update(
    model: ConversationLogModel, msg: ConversationLogMsg
) -> tuple[ConversationLogModel, Cmd]:
    match msg:
        case UserMessageReceived(text=text):
            model.turns.append(Turn(user_message=text))
            return model, none

        case ResponseReceived(response=response, segment=segment):
            segment.messages.append(response)
            return model, none

        case ToolResultsReceived(request=request, segment=segment):
            segment.messages.append(request)
            return model, none

        case ActionLogStart(description=desc):
            model.active_action = desc
            model.allowed_tool = None
            model.action_tool_count = 0
            return model, tool_result(f"Action started: {desc}")

        case ActionLogEnd(result=result):
            summary = (
                f"[Action: {model.active_action}] "
                f"Called {model.action_tool_count} tool(s). "
                f"Result: {result}"
            )
            if model.turns:
                current_segment(model).summary = summary
                model.turns[-1].segments.append(Segment())
            model.active_action = None
            model.allowed_tool = None
            model.action_tool_count = 0
            return model, tool_result("Action logged.")

        case ToolTracked(tool_name=name):
            if model.active_action is not None:
                model.action_tool_count += 1
                if model.allowed_tool is None:
                    model.allowed_tool = name
            return model, none


# --- Pure query functions ---


def check_protocol(model: ConversationLogModel, tool_name: str) -> str | None:
    """Check if a tool call is allowed under the action log protocol.

    Returns an error message if the call violates the protocol, or None
    if it's allowed.
    """
    if tool_name == "action_log_start":
        return None
    if tool_name == "action_log_end":
        if model.active_action is None:
            return (
                "Cannot call action_log_end without an active action. "
                "Call action_log_start first."
            )
        if model.action_tool_count == 0:
            return (
                "Cannot call action_log_end without doing anything. "
                "Either call a tool or respond to the user first, "
                "then end the action."
            )
        return None
    if model.active_action is None:
        return (
            f"Cannot execute '{tool_name}' without an active action log entry. "
            "Call action_log_start first."
        )
    if model.allowed_tool is None:
        return None
    if tool_name != model.allowed_tool:
        return (
            f"Cannot execute '{tool_name}' during this action log entry. "
            f"Only '{model.allowed_tool}' is allowed until you call action_log_end."
        )
    return None


def current_segment(model: ConversationLogModel) -> Segment:
    """Return the current (latest) segment of the latest turn."""
    return model.turns[-1].segments[-1]


# --- Views ---

_TOOL_DEFS = [
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
                    "description": ("What you are about to do, in imperative mood."),
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


def view_messages(model: ConversationLogModel) -> Iterator[ModelMessage]:
    yield ModelRequest(parts=[SystemPromptPart(content=_RULES)])

    for turn in model.turns:
        yield ModelRequest(parts=[UserPromptPart(content=turn.user_message)])
        for segment in turn.segments:
            if segment.summary is not None:
                yield ModelRequest(parts=[SystemPromptPart(content=segment.summary)])
                for msg in segment.messages:
                    if isinstance(msg, ModelResponse):
                        tool_parts = [
                            p for p in msg.parts if isinstance(p, ToolCallPart)
                        ]
                        if tool_parts:
                            yield ModelResponse(
                                parts=tool_parts,
                                model_name=msg.model_name,
                            )
                    elif isinstance(msg, ModelRequest):
                        tool_parts = [
                            p for p in msg.parts if isinstance(p, ToolReturnPart)
                        ]
                        if tool_parts:
                            yield ModelRequest(parts=tool_parts)
            else:
                yield from segment.messages


def view_tools(model: ConversationLogModel) -> Iterator[ToolDefinition]:
    yield from _TOOL_DEFS


def view_html(model: ConversationLogModel) -> str:
    if not model.turns:
        content = "<p><em>No messages yet</em></p>"
    else:
        parts = []
        for turn in model.turns:
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
                    for msg in segment.messages:
                        parts.extend(_render_tool_parts(msg))
                else:
                    for msg in segment.messages:
                        parts.extend(_render_message(msg))
        content = "".join(parts)
    return (
        f'<div id="widget-conversation-log" class="widget">'
        f"<h3>Conversation</h3>{content}</div>"
    )


def _render_tool_parts(msg: ModelMessage) -> list[str]:
    """Render only tool call/result parts from a message."""
    parts: list[str] = []
    if isinstance(msg, ModelResponse):
        for part in msg.parts:
            if isinstance(part, ToolCallPart):
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


# --- Anticorruption layers ---


def from_llm(
    model: ConversationLogModel, tool_name: str, args: dict
) -> ConversationLogMsg:
    match tool_name:
        case "action_log_start":
            return ActionLogStart(description=args["description"])
        case "action_log_end":
            return ActionLogEnd(result=args["result"])
    raise ValueError(f"ConversationLog: unknown tool '{tool_name}'")


# No frontend tools for ConversationLog


# --- Factory ---


def create_conversation_log() -> WidgetHandle:
    return create_widget(
        id="widget-conversation-log",
        model=ConversationLogModel(),
        update=update,
        view_messages=view_messages,
        view_tools=view_tools,
        view_html=view_html,
        from_llm=from_llm,
    )
