"""Context widget — root composition of all widgets into the model's context."""

from collections.abc import Iterator
from dataclasses import dataclass, field

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.tools import ToolDefinition

from calipso.widget import WidgetHandle
from calipso.widgets.conversation_log import (
    ResponseReceived,
    Segment,
    ToolTracked,
    UserMessageReceived,
    check_protocol,
    current_segment,
)

_STATE_BEGIN = "─── CURRENT STATE ───"
_STATE_END = "─── END STATE ───"


@dataclass
class Context:
    """Root compositor that composes all child widgets.

    The runner only interacts with the Context — it doesn't know about
    the tree structure inside. The Context handles:
    - Composing all child views via yield from
    - Dispatching tool calls to the owning widget
    - Enforcing cross-widget protocols (action log)

    Layout order:
    1. system_prompt (identity + framing)
    2. conversation_log (action protocol rules + conversation history)
    3. children (state panels, wrapped in markers)
    """

    system_prompt: WidgetHandle
    children: list[WidgetHandle] = field(default_factory=list)
    conversation_log: WidgetHandle = field(default_factory=lambda: None)  # type: ignore[assignment]
    _tool_owners: dict[str, WidgetHandle] = field(
        init=False, repr=False, default_factory=dict
    )
    _html_cache: dict[str, str] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._rebuild_tool_owners()

    def _rebuild_tool_owners(self) -> None:
        self._tool_owners.clear()
        for widget in self._all_widgets():
            for tool_def in widget.view_tools():
                self._tool_owners[tool_def.name] = widget

    def _all_widgets(self) -> Iterator[WidgetHandle]:
        yield self.system_prompt
        yield from self.children
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
        self, response: ModelResponse
    ) -> tuple[list[tuple[str, str]], Segment]:
        """Process a model response, dispatch tool calls, return tool results.

        Returns (tool_results, segment) where tool_results is a list of
        (tool_call_id, result_text) pairs and segment is the pinned segment
        that the response and tool results should be recorded in.
        """
        conv_model = self.conversation_log.model
        segment = current_segment(conv_model)

        tool_results: list[tuple[str, str]] = []

        for part in response.parts:
            if not isinstance(part, ToolCallPart):
                continue

            name = part.tool_name
            args = part.args_as_dict()

            # Protocol enforcement
            error = check_protocol(self.conversation_log.model, name)
            if error is not None:
                tool_results.append((part.tool_call_id, error))
                continue

            # Dispatch to owning widget
            owner = self._tool_owners.get(name)
            if owner is None:
                tool_results.append((part.tool_call_id, f"Unknown tool: {name}"))
                continue

            result = await owner.dispatch_llm(name, args)
            tool_results.append((part.tool_call_id, result))

            # Track non-action-log tools for protocol enforcement
            if owner is not self.conversation_log:
                self.conversation_log.send(ToolTracked(tool_name=name))

        # Record the response in the pinned segment
        self.conversation_log.send(ResponseReceived(response=response, segment=segment))

        return tool_results, segment

    async def handle_widget_event(self, tool_name: str, args: dict) -> str | None:
        """Handle a frontend-initiated widget event.

        Bypasses the action log protocol. Returns the update result,
        or None if the tool is unknown or not frontend-callable.
        """
        owner = self._tool_owners.get(tool_name)
        if owner is None:
            return None
        return await owner.dispatch_ui(tool_name, args)
