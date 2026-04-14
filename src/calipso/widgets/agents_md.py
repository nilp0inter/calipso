"""AgentsMd widget — loads behavioral instructions from AGENTS.md or CLAUDE.md."""

import html as html_mod
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from pydantic_ai.messages import ModelMessage, ModelRequest, SystemPromptPart
from pydantic_ai.tools import ToolDefinition

from calipso.widget import Widget, render_md

_CANDIDATES = ("AGENTS.md", "CLAUDE.md")


@dataclass
class AgentsMd(Widget):
    """Loads behavioral instructions from AGENTS.md or CLAUDE.md (first found)."""

    directory: Path = field(default_factory=lambda: Path.cwd())
    loaded_path: str | None = field(init=False, repr=False, default=None)
    content: str | None = field(init=False, repr=False, default=None)
    error: str | None = field(init=False, repr=False, default=None)
    _tool_defs: list[ToolDefinition] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._tool_defs = [
            ToolDefinition(
                name="reload_agents_md",
                description=(
                    "Reload behavioral instructions from AGENTS.md or CLAUDE.md."
                ),
                parameters_json_schema={"type": "object", "properties": {}},
            ),
        ]
        self._load()

    def _load(self) -> None:
        for name in _CANDIDATES:
            p = self.directory / name
            try:
                text = p.read_text()
            except (FileNotFoundError, OSError):
                continue
            if text.strip():
                self.loaded_path = str(p)
                self.content = text
                self.error = None
                return
        self.loaded_path = None
        self.content = None
        self.error = f"Neither AGENTS.md nor CLAUDE.md found in {self.directory}"

    def view_messages(self) -> Iterator[ModelMessage]:
        if self.content:
            yield ModelRequest(parts=[SystemPromptPart(content=self.content)])

    def view_tools(self) -> Iterator[ToolDefinition]:
        yield from self._tool_defs

    def view_html(self) -> str:
        if self.error:
            warning = (
                '<p class="agents-md-warning">'
                f"<em>{html_mod.escape(self.error)}</em></p>"
            )
            body = warning
        else:
            filename = Path(self.loaded_path).name if self.loaded_path else ""
            body = f"<p><strong>{html_mod.escape(filename)}</strong></p>" + render_md(
                self.content or ""
            )

        reload_btn = (
            "<button onclick=\"sendWidgetEvent('reload_agents_md', {})\""
            ' class="btn-remove" title="Reload">'
            "Reload</button>"
        )
        return (
            f'<div id="{self.widget_id()}" class="widget">'
            f"<h3>AGENTS.md {reload_btn}</h3>{body}</div>"
        )

    async def update(self, tool_name: str, args: dict) -> str:
        if tool_name == "reload_agents_md":
            self._load()
            if self.error:
                return self.error
            return f"Loaded: {self.loaded_path}"
        raise NotImplementedError(f"AgentsMd does not handle tool '{tool_name}'")

    def frontend_tools(self) -> set[str]:
        return {"reload_agents_md"}
