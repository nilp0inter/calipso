"""Static system prompt widget — the trivial case."""

from collections.abc import Iterator
from dataclasses import dataclass

from pydantic_ai.messages import ModelMessage, ModelRequest, SystemPromptPart

from calipso.widget import Widget


@dataclass
class SystemPrompt(Widget):
    text: str = "You are Calipso, an AI coding assistant."

    def view_messages(self) -> Iterator[ModelMessage]:
        yield ModelRequest(parts=[SystemPromptPart(content=self.text)])
