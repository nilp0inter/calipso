"""ConversationLog widget — manages conversation turns with step protocol.

Messages are partitioned into segments. Each segment has an optional
model-provided summary: summarized segments render as their summary text,
unsummarized segments render their full messages. This prevents replaying
raw tool call/result messages for completed steps.
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
    and summary is set when the step ends.
    """

    messages: list[ModelMessage] = field(default_factory=list)
    summary: str | None = None
    show_tools: bool = False


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
    active_step: str | None = None
    step_tool_count: int = 0


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
class BeginStep:
    description: str


@dataclass(frozen=True)
class EndStep:
    result: str


@dataclass(frozen=True)
class ToolTracked:
    tool_name: str


@dataclass(frozen=True)
class ToggleSegmentTools:
    turn_index: int
    segment_index: int


ConversationLogMsg = (
    UserMessageReceived
    | ResponseReceived
    | ToolResultsReceived
    | BeginStep
    | EndStep
    | ToolTracked
    | ToggleSegmentTools
)


_RULES = (
    "## Step Protocol — conversation memory management\n"
    "\n"
    "Your conversation history is managed through **steps**. A step is a\n"
    "logical unit of work (e.g. reading a file, setting a goal, running\n"
    "a query). When you close a step with `end_step`, the full history\n"
    "of that step is **replaced by your summary**. This keeps the\n"
    "conversation compact — only your summary survives, so it must be\n"
    "accurate and complete.\n"
    "\n"
    "### Lifecycle\n"
    "1. Call `begin_step` — describe what you are about to do.\n"
    "2. Call your tool(s) — mix freely, as many as needed.\n"
    "3. Read the results. Once you have seen all results, call\n"
    "   `end_step` with a summary of the actions you took and their\n"
    "   outcomes. Describe what you DID, not the current state — the\n"
    "   state panels already show that.\n"
    "\n"
    "### Rules\n"
    "- Every tool call must be inside a step. Calls outside a step"
    " are rejected.\n"
    "- Only one step at a time. Call `end_step` before `begin_step`"
    " again.\n"
    "- `end_step` must be the **first** tool call in a response."
    " You must not call any tool before it in the same response."
    " You may call `begin_step` (and more tools) after it.\n"
    "- `end_step` may appear at most once per response.\n"
    "- Empty steps (begin then immediately end with no tool calls)"
    " are rejected.\n"
    "\n"
    "### Why this matters\n"
    "Your `end_step` summary **replaces** the full conversation for\n"
    "that step. If you call `end_step` in the same response as your\n"
    "tools, you have not seen the results yet — your summary will be\n"
    "wrong, and you will lose information permanently.\n"
    "\n"
    "### Example\n"
    "```\n"
    "Response 1: begin_step({description: 'Read config and set goal'})\n"
    "Response 1: open_file({path: 'config.yaml'})\n"
    "            ← you receive tool results\n"
    "Response 2: set_goal({goal: 'Deploy v2'})\n"
    "            ← you receive tool results\n"
    "Response 3: end_step({result: 'Read config.yaml, found deploy\n"
    "             target is v2. Set goal to Deploy v2.'})\n"
    "```\n"
    "Note: `end_step` is in a **separate response** after seeing all\n"
    "results. The summary captures what actually happened."
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

        case BeginStep(description=desc):
            model.active_step = desc
            model.step_tool_count = 0
            return model, tool_result(f"Step started: {desc}")

        case EndStep(result=result):
            summary = (
                f"[Step: {model.active_step}] "
                f"Called {model.step_tool_count} tool(s). "
                f"Result: {result}"
            )
            if model.turns:
                current_segment(model).summary = summary
                model.turns[-1].segments.append(Segment())
            model.active_step = None
            model.step_tool_count = 0
            return model, tool_result("Step logged.")

        case ToolTracked(tool_name=_):
            if model.active_step is not None:
                model.step_tool_count += 1
            return model, none

        case ToggleSegmentTools(turn_index=ti, segment_index=si):
            model.turns[ti].segments[si].show_tools = (
                not model.turns[ti].segments[si].show_tools
            )
            return model, none


# --- Pure query functions ---


def check_protocol(model: ConversationLogModel, tool_name: str) -> str | None:
    """Check if a tool call is allowed under the step protocol.

    Returns an error message if the call violates the protocol, or None
    if it's allowed.
    """
    if tool_name == "begin_step":
        if model.active_step is not None:
            return (
                "Cannot call begin_step while a step is already "
                "active. Call end_step first."
            )
        return None
    if tool_name == "end_step":
        if model.active_step is None:
            return "Cannot call end_step without an active step. Call begin_step first."
        if model.step_tool_count == 0:
            return (
                "Cannot call end_step without doing anything. "
                "Call at least one tool first, then end the step."
            )
        return None
    if model.active_step is None:
        return f"Cannot execute '{tool_name}' outside a step. Call begin_step first."
    return None


def current_segment(model: ConversationLogModel) -> Segment:
    """Return the current (latest) segment of the latest turn."""
    return model.turns[-1].segments[-1]


# --- Views ---

_TOOL_START = ToolDefinition(
    name="begin_step",
    description=(
        "Begin a new step. Call this before using any tool."
        " Describe what you are about to do."
    ),
    parameters_json_schema={
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "What you are about to do.",
            },
        },
        "required": ["description"],
    },
)

_TOOL_END = ToolDefinition(
    name="end_step",
    description=(
        "End the current step with a summary of what happened."
        " Must be the first tool call in the response."
        " The summary permanently replaces the step's history."
        " Summarize the ACTIONS YOU TOOK and their results —"
        " do not describe the current state of the system."
        " The system state is already visible in the state"
        " panels; the summary records what you DID."
    ),
    parameters_json_schema={
        "type": "object",
        "properties": {
            "result": {
                "type": "string",
                "description": (
                    "Summary of the actions you performed and their"
                    " outcomes. Describe what you did, not what you"
                    " see now. Example: 'Set the goal to Deploy v2'"
                    " — not 'The goal is already Deploy v2'."
                ),
            },
        },
        "required": ["result"],
    },
)

ALL_CONVERSATION_LOG_TOOLS: frozenset[str] = frozenset(
    {_TOOL_START.name, _TOOL_END.name}
)


def view_messages(model: ConversationLogModel) -> Iterator[ModelMessage]:
    yield ModelRequest(parts=[SystemPromptPart(content=_RULES)])

    for turn in model.turns:
        yield ModelRequest(parts=[UserPromptPart(content=turn.user_message)])
        for segment in turn.segments:
            if segment.summary is not None:
                yield ModelRequest(parts=[SystemPromptPart(content=segment.summary)])
                if segment.show_tools:
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
    if model.active_step is None:
        yield _TOOL_START
    elif model.step_tool_count > 0:
        yield _TOOL_END


def view_html(model: ConversationLogModel) -> str:
    if not model.turns:
        content = "<p><em>No messages yet</em></p>"
    else:
        parts = []
        for i, turn in enumerate(model.turns):
            if i > 0:
                parts.append('<hr class="turn-sep">')
            parts.append('<div class="turn">')
            user_html = render_md(turn.user_message)
            parts.append(
                f'<div class="msg sent">'
                f'<span class="bubble-label">You</span>'
                f"{user_html}"
                f"</div>"
            )
            for j, segment in enumerate(turn.segments):
                if segment.summary is not None:
                    tool_html = []
                    for msg in segment.messages:
                        tool_html.extend(_render_tool_parts(msg))
                    open_attr = " open" if segment.show_tools else ""
                    toggle_handler = (
                        f"sendWidgetEvent('toggle_segment_tools',"
                        f"{{'turn_index':{i},'segment_index':{j}}})"
                    )
                    parts.append(
                        f'<details class="tool-group"{open_attr}'
                        f' ontoggle="{toggle_handler}">'
                        f"<summary>"
                        f"{html_mod.escape(segment.summary)}"
                        f"</summary>"
                        f"{''.join(tool_html)}"
                        f"</details>"
                    )
                else:
                    text_parts = []
                    tool_parts = []
                    for msg in segment.messages:
                        t, tl = _split_message(msg)
                        text_parts.extend(t)
                        tool_parts.extend(tl)
                    parts.extend(text_parts)
                    if tool_parts:
                        tool_call_count = sum(
                            1
                            for m in segment.messages
                            if isinstance(m, ModelResponse)
                            for p in m.parts
                            if isinstance(p, ToolCallPart)
                        )
                        label = (
                            "1 tool call"
                            if tool_call_count == 1
                            else f"{tool_call_count} tool calls"
                        )
                        parts.append(
                            '<details class="tool-group">'
                            f"<summary>{label}</summary>"
                            f"{''.join(tool_parts)}"
                            "</details>"
                        )
            parts.append("</div>")
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
                    f'<div class="msg received tool-call">'
                    f'<span class="dir">&#9660;</span> '
                    f"<code>{html_mod.escape(part.tool_name)}"
                    f"({args})</code></div>"
                )
    elif isinstance(msg, ModelRequest):
        for part in msg.parts:
            if isinstance(part, ToolReturnPart):
                parts.append(
                    f'<div class="msg sent tool-result">'
                    f'<span class="dir">&#9650;</span> '
                    f"<code>"
                    f"{html_mod.escape(str(part.content))}"
                    f"</code></div>"
                )
    return parts


def _split_message(
    msg: ModelMessage,
) -> tuple[list[str], list[str]]:
    """Split a message into (text_parts, tool_parts) HTML."""
    text: list[str] = []
    tools: list[str] = []
    if isinstance(msg, ModelResponse):
        for part in msg.parts:
            if isinstance(part, TextPart):
                rendered = render_md(part.content)
                text.append(
                    f'<div class="msg received assistant">'
                    f'<span class="bubble-label">Agent</span>'
                    f"{rendered}</div>"
                )
            elif isinstance(part, ToolCallPart):
                args = html_mod.escape(str(part.args_as_dict()))
                tools.append(
                    f'<div class="msg received tool-call">'
                    f'<span class="dir">&#9660;</span> '
                    f"<code>{html_mod.escape(part.tool_name)}"
                    f"({args})</code></div>"
                )
    elif isinstance(msg, ModelRequest):
        for part in msg.parts:
            if isinstance(part, ToolReturnPart):
                tools.append(
                    f'<div class="msg sent tool-result">'
                    f'<span class="dir">&#9650;</span> '
                    f"<code>"
                    f"{html_mod.escape(str(part.content))}"
                    f"</code></div>"
                )
    return text, tools


# --- Anticorruption layers ---


def from_llm(
    model: ConversationLogModel, tool_name: str, args: dict
) -> ConversationLogMsg:
    match tool_name:
        case "begin_step":
            return BeginStep(description=args["description"])
        case "end_step":
            return EndStep(result=args["result"])
    raise ValueError(f"ConversationLog: unknown tool '{tool_name}'")


def from_ui(
    model: ConversationLogModel, event_name: str, args: dict
) -> ConversationLogMsg | None:
    if event_name == "toggle_segment_tools":
        return ToggleSegmentTools(
            turn_index=int(args["turn_index"]),
            segment_index=int(args["segment_index"]),
        )
    return None


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
        from_ui=from_ui,
        frontend_tools=frozenset({"toggle_segment_tools"}),
    )
