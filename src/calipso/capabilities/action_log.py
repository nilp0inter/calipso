from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.toolsets import FunctionToolset


@dataclass
class LogEntry:
    id: int
    action: str
    result: str


TOOL_NAMES = frozenset({"action_log_start", "action_log_end"})


@dataclass
class ActionLog(AbstractCapability[Any]):
    entries: list[LogEntry] = field(default_factory=list)
    _next_id: int = field(init=False, repr=False, default=1)
    _active_action: str | None = field(init=False, repr=False, default=None)
    _allowed_tool: str | None = field(init=False, repr=False, default=None)
    _needs_compaction: bool = field(init=False, repr=False, default=False)
    _toolset: FunctionToolset[Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.entries:
            self._next_id = max(e.id for e in self.entries) + 1

        self._toolset = FunctionToolset()

        @self._toolset.tool_plain
        def action_log_start(description: str) -> str:
            """Start a new action log entry. Call this before executing any tool.

            Args:
                description: What you are about to do, in imperative mood.
            """
            # State is set in before_tool_execute, not here.
            return f"Action started: {description}"

        @self._toolset.tool_plain
        def action_log_end(result: str) -> str:
            """Finish the current action log entry.

            Args:
                result: A summary of what happened.
            """
            # State is set in before_tool_execute, not here.
            return f"Action #{self._next_id - 1} logged."

    _RULES = (
        "## Action Log\n"
        "Rules:\n"
        "- You MUST call action_log_start before executing any other tool.\n"
        "- After action_log_start, you may call one type of tool "
        "one or more times.\n"
        "- You MUST call action_log_end before starting a new action.\n"
        "- Violating these rules will result in an error."
    )

    def get_instructions(self):
        return self._RULES

    def get_toolset(self) -> FunctionToolset[Any]:
        return self._toolset

    @staticmethod
    def _summary_message(entry: LogEntry) -> ModelResponse:
        text = f"- Task: {entry.action}\n  Result: {entry.result}"
        return ModelResponse(parts=[TextPart(content=text)])

    @staticmethod
    def _find_compaction_ranges(
        msgs: list[ModelRequest | ModelResponse],
    ) -> list[tuple[int, int, str, str]]:
        """Find (start_idx, end_idx, action, result) for completed actions."""
        ranges: list[tuple[int, int, str, str]] = []
        i = 0
        while i < len(msgs):
            msg = msgs[i]
            if isinstance(msg, ModelResponse):
                start_call = None
                for part in msg.parts:
                    if (
                        isinstance(part, ToolCallPart)
                        and part.tool_name == "action_log_start"
                    ):
                        start_call = part
                        break
                if start_call is not None:
                    action = start_call.args_as_dict().get("description", "")
                    for j in range(i + 1, len(msgs)):
                        end_msg = msgs[j]
                        if isinstance(end_msg, ModelRequest):
                            for part in end_msg.parts:
                                if (
                                    isinstance(part, ToolReturnPart)
                                    and part.tool_name == "action_log_end"
                                    and "logged" in part.content
                                ):
                                    result_text = ""
                                    for k in range(j, i - 1, -1):
                                        km = msgs[k]
                                        if isinstance(km, ModelResponse):
                                            for kp in km.parts:
                                                if (
                                                    isinstance(kp, ToolCallPart)
                                                    and kp.tool_name == "action_log_end"
                                                ):
                                                    result_text = kp.args_as_dict().get(
                                                        "result", ""
                                                    )
                                                    break
                                            if result_text:
                                                break
                                    ranges.append((i, j + 1, action, result_text))
                                    i = j + 1
                                    break
                        continue
                    else:
                        i += 1
                        continue
                    continue
            i += 1
        return ranges

    async def before_model_request(
        self,
        ctx: RunContext[Any],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        if not self._needs_compaction:
            return request_context

        self._needs_compaction = False
        msgs = request_context.messages
        # Exclude the last message (the current ModelRequest being sent)
        # from compaction — pydantic-ai requires it to stay.
        ranges = self._find_compaction_ranges(msgs[:-1])

        for start, end, action, result in reversed(ranges):
            entry = LogEntry(id=0, action=action, result=result)
            msgs[start:end] = [self._summary_message(entry)]

        return request_context

    async def before_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call,
        tool_def,
        args,
    ):
        name = call.tool_name

        if name == "action_log_start":
            description = args.get("description", "")
            self._active_action = description
            self._allowed_tool = None
            return args

        if name == "action_log_end":
            if self._active_action is None:
                raise ModelRetry(
                    "Cannot call action_log_end without an active action. "
                    "Call action_log_start first."
                )
            result = args.get("result", "")
            entry = LogEntry(
                id=self._next_id,
                action=self._active_action,
                result=result,
            )
            self.entries.append(entry)
            self._next_id += 1
            self._active_action = None
            self._allowed_tool = None
            self._needs_compaction = True
            return args

        # Any other tool: enforce protocol.
        if self._active_action is None:
            raise ModelRetry(
                f"Cannot execute '{name}' without an active action log entry. "
                "Call action_log_start first."
            )

        if self._allowed_tool is None:
            self._allowed_tool = name
        elif name != self._allowed_tool:
            raise ModelRetry(
                f"Cannot execute '{name}' during this action log entry. "
                f"Only '{self._allowed_tool}' is allowed until you call "
                "action_log_end."
            )

        return args
