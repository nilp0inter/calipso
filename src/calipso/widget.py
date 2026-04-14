"""Base widget protocol for Calipso's context engineering architecture.

Widgets are Elm-inspired components (Model/Update/View) that compose into
the model's context. Each widget is created via ``create_widget`` which
produces a ``WidgetHandle``:

- **Model**: a frozen dataclass holding internal state
- **Update**: a pure function ``(model, msg) -> (model, result_str)``
- **Views**: free functions ``model -> Iterator[T]`` or ``model -> str``
- **Anticorruption layers**: ``from_llm`` (async) and ``from_ui`` (sync)
  translate external events into typed Msg values

View functions return Iterator[T] — composition uses ``yield from``
which naturally flattens nested iterators (List monad join).
"""

from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

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


# ---------------------------------------------------------------------------
# Default no-op functions for create_widget
# ---------------------------------------------------------------------------


def _no_messages(_model: Any) -> Iterator[ModelMessage]:
    return iter(())


def _no_tools(_model: Any) -> Iterator[ToolDefinition]:
    return iter(())


def _default_view_html(widget_id: str) -> Callable[[Any], str]:
    def _view_html(_model: Any) -> str:
        return f'<div id="{widget_id}"></div>'

    return _view_html


async def _no_llm(_model: Any, _tool_name: str, _args: dict) -> Any:
    raise NotImplementedError("This widget does not handle LLM tool calls")


def _no_ui(_model: Any, _event_name: str, _args: dict) -> Any | None:
    return None


def _no_update(_model: Any, _msg: Any) -> tuple[Any, str]:
    raise NotImplementedError("This widget does not handle updates")


# ---------------------------------------------------------------------------
# WidgetHandle — the uniform interface Context works with
# ---------------------------------------------------------------------------


@dataclass
class WidgetHandle:
    """A composed widget: model reference + function table.

    Context interacts only with WidgetHandle instances. The handle wraps
    a mutable model reference and delegates to pure functions for update
    and views.
    """

    id: str
    _model: Any = field(repr=False)
    _update_fn: Callable[[Any, Any], tuple[Any, str]] = field(repr=False)
    _view_messages_fn: Callable[[Any], Iterator[ModelMessage]] = field(repr=False)
    _view_tools_fn: Callable[[Any], Iterator[ToolDefinition]] = field(repr=False)
    _view_html_fn: Callable[[Any], str] = field(repr=False)
    _from_llm_fn: Callable[[Any, str, dict], Awaitable[Any]] = field(repr=False)
    _from_ui_fn: Callable[[Any, str, dict], Any | None] = field(repr=False)
    _frontend_tool_names: frozenset[str] = field(repr=False)

    # -- Public model access ------------------------------------------------

    @property
    def model(self) -> Any:
        """Read access to the current model.

        Elm-style: parent can inspect child state.
        """
        return self._model

    # -- Public interface (same names Context already calls) ----------------

    def widget_id(self) -> str:
        return self.id

    def view_messages(self) -> Iterator[ModelMessage]:
        return self._view_messages_fn(self._model)

    def view_tools(self) -> Iterator[ToolDefinition]:
        return self._view_tools_fn(self._model)

    def view_html(self) -> str:
        return self._view_html_fn(self._model)

    def frontend_tools(self) -> frozenset[str]:
        return self._frontend_tool_names

    # -- Dispatch methods ---------------------------------------------------

    async def dispatch_llm(self, tool_name: str, args: dict) -> str:
        """Dispatch an LLM tool call: from_llm -> update.

        ValueError raised by from_llm is caught and returned as the
        tool result string (validation errors at the boundary).
        """
        try:
            msg = await self._from_llm_fn(self._model, tool_name, args)
        except ValueError as e:
            return str(e)
        self._model, result = self._update_fn(self._model, msg)
        return result

    def dispatch_ui(self, tool_name: str, args: dict) -> str | None:
        """Dispatch a browser event: from_ui -> update.

        Returns None if the tool is not frontend-callable or not recognized.
        """
        if tool_name not in self._frontend_tool_names:
            return None
        msg = self._from_ui_fn(self._model, tool_name, args)
        if msg is None:
            return None
        self._model, result = self._update_fn(self._model, msg)
        return result

    def send(self, msg: Any) -> str:
        """Send a Msg directly, bypassing anticorruption layers.

        Standard Elm pattern: parent forwards messages to children.
        """
        self._model, result = self._update_fn(self._model, msg)
        return result


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def create_widget(
    *,
    id: str,
    model: Any,
    update: Callable[[Any, Any], tuple[Any, str]] = _no_update,
    view_messages: Callable[[Any], Iterator[ModelMessage]] = _no_messages,
    view_tools: Callable[[Any], Iterator[ToolDefinition]] = _no_tools,
    view_html: Callable[[Any], str] | None = None,
    from_llm: Callable[[Any, str, dict], Awaitable[Any]] = _no_llm,
    from_ui: Callable[[Any, str, dict], Any | None] = _no_ui,
    frontend_tools: frozenset[str] = frozenset(),
) -> WidgetHandle:
    """Create a WidgetHandle from an initial model and function table."""
    return WidgetHandle(
        id=id,
        _model=model,
        _update_fn=update,
        _view_messages_fn=view_messages,
        _view_tools_fn=view_tools,
        _view_html_fn=view_html or _default_view_html(id),
        _from_llm_fn=from_llm,
        _from_ui_fn=from_ui,
        _frontend_tool_names=frontend_tools,
    )
