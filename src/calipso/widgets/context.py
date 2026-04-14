"""Context widget — root composition of all widgets into the model's context."""

from collections.abc import Iterator
from dataclasses import dataclass, field

from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    ToolCallPart,
)
from pydantic_ai.tools import ToolDefinition

from calipso.widget import Widget
from calipso.widgets.action_log import ActionLog
from calipso.widgets.conversation import Conversation


@dataclass
class Context(Widget):
    """Root widget that composes all child widgets.

    The runner only interacts with the Context — it doesn't know about
    the tree structure inside. The Context handles:
    - Composing all child views via yield from
    - Dispatching tool calls to the owning widget
    - Enforcing cross-widget protocols (action log)
    """

    children: list[Widget] = field(default_factory=list)
    conversation: Conversation = field(default_factory=Conversation)
    action_log: ActionLog | None = field(default=None, repr=False)
    _tool_owners: dict[str, Widget] = field(
        init=False, repr=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        self._rebuild_tool_owners()

    def _rebuild_tool_owners(self) -> None:
        self._tool_owners.clear()
        for widget in self._all_widgets():
            for tool_def in widget.view_tools():
                self._tool_owners[tool_def.name] = widget

    def _all_widgets(self) -> Iterator[Widget]:
        yield from self.children
        if self.action_log is not None:
            yield self.action_log
        yield self.conversation

    def view_messages(self) -> Iterator[ModelMessage]:
        for widget in self.children:
            yield from widget.view_messages()
        if self.action_log is not None:
            yield from self.action_log.view_messages()
        yield from self.conversation.view_messages()

    def view_tools(self) -> Iterator[ToolDefinition]:
        for widget in self._all_widgets():
            yield from widget.view_tools()

    def add_user_message(self, text: str) -> None:
        self.conversation.add_user_message(text)

    def handle_response(self, response: ModelResponse) -> list[tuple[str, str]]:
        """Process a model response, dispatch tool calls, return tool results.

        Returns a list of (tool_call_id, result_text) pairs for tool calls,
        or an empty list if the response was text-only.
        """
        tool_results: list[tuple[str, str]] = []

        for part in response.parts:
            if not isinstance(part, ToolCallPart):
                continue

            name = part.tool_name
            args = part.args_as_dict()

            # Protocol enforcement via action log
            if self.action_log is not None:
                error = self.action_log.check_protocol(name)
                if error is not None:
                    tool_results.append((part.tool_call_id, error))
                    continue

            # Dispatch to owning widget
            owner = self._tool_owners.get(name)
            if owner is None:
                tool_results.append((part.tool_call_id, f"Unknown tool: {name}"))
                continue

            result = owner.update(name, args)
            tool_results.append((part.tool_call_id, result))

            # Track non-action-log tools for protocol enforcement
            if self.action_log is not None and owner is not self.action_log:
                self.action_log.track_tool(name)

        # Record the response in conversation
        self.conversation.add_response(response)

        return tool_results
