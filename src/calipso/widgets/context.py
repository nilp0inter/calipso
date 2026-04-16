"""Context widget — root composition of all widgets into the model's context."""

from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, field, replace

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.tools import ToolDefinition

from calipso.widget import WidgetHandle
from calipso.widgets.conversation_log import (
    ALL_CONVERSATION_LOG_TOOLS,
    ConsumePicks,
    ResponseReceived,
    ToolResultsReceived,
    UserMessageReceived,
    check_protocol,
    current_owning_task_id,
)
from calipso.widgets.token_usage import UsageRecorded

_STATE_BEGIN = "─── CURRENT STATE ───"
_STATE_END = "─── END STATE ───"


@dataclass
class Context:
    """Root compositor that composes all child widgets.

    The runner only interacts with the Context — it doesn't know about
    the tree structure inside. The Context handles:
    - Composing all child views via yield from
    - Dispatching tool calls to the owning widget
    - Enforcing cross-widget protocols (task protocol)

    Layout order:
    1. system_prompt (identity + framing)
    2. conversation_log (task protocol rules + conversation history)
    3. children (state panels, wrapped in markers)
    """

    system_prompt: WidgetHandle
    children: list[WidgetHandle] = field(default_factory=list)
    conversation_log: WidgetHandle = field(default_factory=lambda: None)  # type: ignore[assignment]
    token_usage: WidgetHandle | None = None
    _tool_owners: dict[str, WidgetHandle] = field(
        init=False, repr=False, default_factory=dict
    )
    _protocol_free_tool_names: frozenset[str] = field(
        init=False, repr=False, default_factory=frozenset
    )
    _html_cache: dict[str, str] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._rebuild_tool_owners()

    def _rebuild_tool_owners(self) -> None:
        self._tool_owners.clear()
        protocol_free: set[str] = set()
        for widget in self._all_widgets():
            for tool_def in widget.view_tools():
                self._tool_owners[tool_def.name] = widget
            for name in widget.frontend_tools():
                self._tool_owners.setdefault(name, widget)
            protocol_free.update(widget.protocol_free_tools())
        # Conversation log owns all task tools even when not currently
        # exposed (view_tools is state-dependent). Register so dispatch
        # can find the owner; _currently_exposed_task_tools() gates
        # actual execution.
        for name in ALL_CONVERSATION_LOG_TOOLS:
            self._tool_owners[name] = self.conversation_log
        self._protocol_free_tool_names = frozenset(protocol_free)

    def _currently_exposed_task_tools(self) -> frozenset[str]:
        """Return task tool names currently exposed to the LLM."""
        return frozenset(t.name for t in self.conversation_log.view_tools())

    def _all_widgets(self) -> Iterator[WidgetHandle]:
        yield self.system_prompt
        yield from self.children
        if self.token_usage is not None:
            yield self.token_usage
        yield self.conversation_log

    def view_messages(self) -> Iterator[ModelMessage]:
        yield from self.system_prompt.view_messages()
        yield from self.conversation_log.view_messages()
        yield ModelRequest(parts=[UserPromptPart(content=_STATE_BEGIN)])
        for widget in self.children:
            yield from widget.view_messages()
        yield ModelRequest(parts=[UserPromptPart(content=_STATE_END)])

    def view_tools(self) -> Iterator[ToolDefinition]:
        for widget in self._all_widgets():
            yield from widget.view_tools()

    def changed_html(self) -> list[str]:
        """Return HTML fragments for widgets whose view_html() changed."""
        changed = []
        for widget in self._all_widgets():
            wid = widget.widget_id()
            current = widget.view_html()
            if self._html_cache.get(wid) != current:
                self._html_cache[wid] = current
                changed.append(current)
        return changed

    def all_html(self) -> list[str]:
        """Return HTML for all widgets (for initial page load)."""
        result = []
        for widget in self._all_widgets():
            wid = widget.widget_id()
            html = widget.view_html()
            self._html_cache[wid] = html
            result.append(html)
        return result

    def add_user_message(self, text: str) -> None:
        self.conversation_log.send(UserMessageReceived(text=text))

    async def handle_response(
        self,
        response: ModelResponse,
        on_update: Callable[[], Awaitable[None]] | None = None,
    ) -> list[tuple[str, str]]:
        """Process a model response, dispatch tool calls, record the
        response and any tool results in the conversation log.

        The owning task id is captured **per tool call** — just before
        dispatch — so a response that mixes ``start_task(x)`` with other
        tools is split at the pivot: ``start_task`` itself is tagged
        with the pre-start task id (``None`` or the outer task), while
        any tools that follow are tagged with the newly-started task.

        Returns the list of ``(tool_call_id, result_text)`` pairs for
        use by the runner (which decides whether to loop).
        """
        # Picks set by the previous response were used to compose the
        # request we're handling now — consume them.
        self.conversation_log.send(ConsumePicks())

        # Pre-scan: close_current_task must appear at most once and be first.
        tool_call_parts = [p for p in response.parts if isinstance(p, ToolCallPart)]
        close_count = sum(
            1 for p in tool_call_parts if p.tool_name == "close_current_task"
        )
        reject_close = close_count > 1 or (
            close_count == 1 and tool_call_parts[0].tool_name != "close_current_task"
        )

        tool_results: list[tuple[str, str]] = []
        # owning_task_id captured immediately *before* each tool call's
        # dispatch, so successful start_task transitions form a pivot.
        per_call_tids: list[int | None] = []

        for part in response.parts:
            if not isinstance(part, ToolCallPart):
                continue

            name = part.tool_name
            args = part.args_as_dict()

            per_call_tids.append(current_owning_task_id(self.conversation_log.model))

            # Reject close_current_task if not first or duplicated.
            if reject_close and name == "close_current_task":
                tool_results.append(
                    (
                        part.tool_call_id,
                        "close_current_task must appear exactly once and be"
                        " the first and only tool call in a response. Do"
                        " not call other tools alongside it.",
                    )
                )
                continue

            # Reject task tools that aren't currently exposed.
            if (
                name in ALL_CONVERSATION_LOG_TOOLS
                and name not in self._currently_exposed_task_tools()
            ):
                tool_results.append(
                    (
                        part.tool_call_id,
                        f"Unknown tool: {name}",
                    )
                )
                continue

            # Protocol enforcement.
            error = check_protocol(
                self.conversation_log.model, name, self._protocol_free_tool_names
            )
            if error is not None:
                tool_results.append((part.tool_call_id, error))
                continue

            # Dispatch to owning widget.
            owner = self._tool_owners.get(name)
            if owner is None:
                tool_results.append((part.tool_call_id, f"Unknown tool: {name}"))
                continue

            result = await owner.dispatch_llm(name, args, on_update=on_update)
            tool_results.append((part.tool_call_id, result))

        # Record token usage.
        if self.token_usage is not None:
            usage = response.usage
            self.token_usage.send(
                UsageRecorded(
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cache_read_tokens=usage.cache_read_tokens,
                    cache_write_tokens=usage.cache_write_tokens,
                )
            )

        # Record response + tool_results, split by owning task id.
        self._record_log_items(response, tool_results, per_call_tids)

        return tool_results

    def _record_log_items(
        self,
        response: ModelResponse,
        tool_results: list[tuple[str, str]],
        per_call_tids: list[int | None],
    ) -> None:
        """Split the response and its tool returns by owning task id and
        send interleaved (ResponseReceived, ToolResultsReceived) pairs
        so items with the same task id remain adjacent in the log.

        When there is a single group (no task pivot) the original
        ``response`` object is preserved — callers and tests rely on
        identity of the logged response.
        """
        result_by_id = dict(tool_results)

        # Walk parts in order, building per-owner groups.
        groups: list[tuple[list, list[ToolReturnPart], int | None]] = []
        current_parts: list = []
        current_returns: list[ToolReturnPart] = []
        current_tid: int | None = current_owning_task_id(self.conversation_log.model)
        has_tid = False
        call_idx = 0

        for part in response.parts:
            if isinstance(part, ToolCallPart):
                tid = per_call_tids[call_idx]
                call_idx += 1
                if not has_tid:
                    current_tid = tid
                    has_tid = True
                elif tid != current_tid:
                    groups.append((current_parts, current_returns, current_tid))
                    current_parts = []
                    current_returns = []
                    current_tid = tid
                current_parts.append(part)
                if part.tool_call_id in result_by_id:
                    current_returns.append(
                        ToolReturnPart(
                            tool_name=part.tool_name,
                            content=result_by_id[part.tool_call_id],
                            tool_call_id=part.tool_call_id,
                        )
                    )
            else:
                current_parts.append(part)

        if current_parts:
            groups.append((current_parts, current_returns, current_tid))

        if not groups:
            return

        single = len(groups) == 1
        for parts, returns, tid in groups:
            sub_response = response if single else replace(response, parts=parts)
            self.conversation_log.send(
                ResponseReceived(response=sub_response, owning_task_id=tid)
            )
            if returns:
                self.conversation_log.send(
                    ToolResultsReceived(
                        request=ModelRequest(parts=returns),
                        owning_task_id=tid,
                    )
                )

    async def handle_widget_event(
        self,
        tool_name: str,
        args: dict,
        on_update: Callable[[], Awaitable[None]] | None = None,
    ) -> str | None:
        """Handle a frontend-initiated widget event.

        Bypasses the task protocol. Returns the update result, or None if
        the tool is unknown or not frontend-callable.
        """
        owner = self._tool_owners.get(tool_name)
        if owner is None:
            return None
        return await owner.dispatch_ui(tool_name, args, on_update=on_update)
