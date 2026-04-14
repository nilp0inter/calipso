"""ActionLog widget — tracks agent actions with protocol enforcement.

The action log enforces a protocol: action_log_start → (one tool type) →
action_log_end. Completed actions are rendered as collapsed summaries in
view_messages() — compaction is a view decision, not history mutation.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
)
from pydantic_ai.tools import ToolDefinition

from calipso.widget import Widget


@dataclass
class LogEntry:
    id: int
    action: str
    result: str


_RULES = (
    "## Action Log\n"
    "Rules:\n"
    "- You MUST call action_log_start before executing any other tool.\n"
    "- After action_log_start, you may call one type of tool one or more times.\n"
    "- You MUST call action_log_end before starting a new action.\n"
    "- Violating these rules will result in an error."
)


@dataclass
class ActionLog(Widget):
    entries: list[LogEntry] = field(default_factory=list)
    _next_id: int = field(init=False, repr=False, default=1)
    _active_action: str | None = field(init=False, repr=False, default=None)
    _allowed_tool: str | None = field(init=False, repr=False, default=None)
    _tool_defs: list[ToolDefinition] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.entries:
            self._next_id = max(e.id for e in self.entries) + 1

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

    def view_messages(self) -> Iterator[ModelMessage]:
        yield ModelRequest(parts=[SystemPromptPart(content=_RULES)])
        for entry in self.entries:
            yield ModelResponse(
                parts=[
                    TextPart(
                        content=(
                            f"[Action #{entry.id}] {entry.action}\n"
                            f"  Result: {entry.result}"
                        )
                    )
                ]
            )

    def view_tools(self) -> Iterator[ToolDefinition]:
        yield from self._tool_defs

    def check_protocol(self, tool_name: str) -> str | None:
        """Check if a tool call is allowed under the action log protocol.

        Returns an error message if the call violates the protocol, or None
        if it's allowed. This is called by the Context for ALL tool calls,
        not just action_log tools.
        """
        if tool_name == "action_log_start":
            return None
        if tool_name == "action_log_end":
            if self._active_action is None:
                return (
                    "Cannot call action_log_end without an active action. "
                    "Call action_log_start first."
                )
            return None
        # Any other tool: must have an active action
        if self._active_action is None:
            return (
                f"Cannot execute '{tool_name}' without an active action log entry. "
                "Call action_log_start first."
            )
        if self._allowed_tool is None:
            return None  # First tool in this action — will be locked in by update
        if tool_name != self._allowed_tool:
            return (
                f"Cannot execute '{tool_name}' during this action log entry. "
                f"Only '{self._allowed_tool}' is allowed until you call action_log_end."
            )
        return None

    def update(self, tool_name: str, args: dict) -> str:
        if tool_name == "action_log_start":
            self._active_action = args["description"]
            self._allowed_tool = None
            return f"Action started: {args['description']}"

        if tool_name == "action_log_end":
            entry = LogEntry(
                id=self._next_id,
                action=self._active_action,  # type: ignore[arg-type]
                result=args["result"],
            )
            self.entries.append(entry)
            self._next_id += 1
            self._active_action = None
            self._allowed_tool = None
            return f"Action #{entry.id} logged."

        raise NotImplementedError(f"ActionLog does not handle tool '{tool_name}'")

    def track_tool(self, tool_name: str) -> None:
        """Track that a non-action-log tool was called (locks the allowed tool)."""
        if self._active_action is not None and self._allowed_tool is None:
            self._allowed_tool = tool_name
