"""AgentsMd widget — loads behavioral instructions from an AGENTS.md file."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from pydantic_ai.messages import ModelMessage, ModelRequest, SystemPromptPart

from calipso.widget import Widget


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
