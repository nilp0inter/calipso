"""AgentsMd widget — loads behavioral instructions from an AGENTS.md file."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from pydantic_ai.messages import ModelMessage, ModelRequest, SystemPromptPart

from calipso.widget import Widget, render_md


@dataclass
class AgentsMd(Widget):
    path: Path = Path("AGENTS.md")

    def view_messages(self) -> Iterator[ModelMessage]:
        try:
            text = self.path.read_text()
        except FileNotFoundError:
            return
        if text.strip():
            yield ModelRequest(parts=[SystemPromptPart(content=text)])

    def view_html(self) -> str:
        try:
            text = self.path.read_text()
        except FileNotFoundError:
            text = ""
        if text.strip():
            content = render_md(text)
        else:
            content = "<em>Not found</em>"
        return (
            f'<div id="{self.widget_id()}" class="widget">'
            f"<h3>AGENTS.md</h3>{content}</div>"
        )
