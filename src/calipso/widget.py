"""Base widget protocol for Calipso's context engineering architecture.

Widgets are Elm-inspired components (State/View/Update) that compose into
the model's context. Each widget:

- Holds internal state as dataclass fields
- Provides view functions (generators yielding messages, tools, etc.)
- Handles updates (tool calls, events) that mutate state

View functions return Iterator[T] — composition uses ``yield from``
which naturally flattens nested iterators (List monad join).
"""

import re
from collections.abc import Iterator
from dataclasses import dataclass

import markdown as md_lib
from pydantic_ai.messages import ModelMessage
from pydantic_ai.tools import ToolDefinition

_md = md_lib.Markdown(extensions=["fenced_code", "tables"])


def render_md(text: str) -> str:
    """Convert markdown text to safe HTML.

    Raw HTML in the input is escaped before markdown processing to
    prevent XSS — markdown syntax still works normally.
    """
    import html

    safe_text = html.escape(text)
    _md.reset()
    return _md.convert(safe_text)


@dataclass
class Widget:
    """Base class for all widgets.

    Subclasses override view and update methods as needed.
    Not all widgets need all methods — a static widget may only
    implement view_messages(), while an interactive widget will
    also implement view_tools() and update().
    """

    def widget_id(self) -> str:
        """Stable HTML element ID for this widget (kebab-case from class name)."""
        name = re.sub(r"(?<!^)(?=[A-Z])", "-", type(self).__name__).lower()
        return f"widget-{name}"

    def view_messages(self) -> Iterator[ModelMessage]:
        """Yield messages that represent this widget in the context."""
        return iter(())

    def view_tools(self) -> Iterator[ToolDefinition]:
        """Yield tool definitions that this widget exposes."""
        return iter(())

    def view_html(self) -> str:
        """Return an HTML fragment for this widget's browser panel."""
        return f'<div id="{self.widget_id()}"></div>'

    def update(self, tool_name: str, args: dict) -> str:
        """Handle a tool call directed at this widget.

        Returns the tool's response string.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not handle tool '{tool_name}'"
        )
