from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability

from calipso.capabilities import Goal, TaskList


@dataclass
class SystemPrompt(AbstractCapability[Any]):
    text: str = (
        "You are Calipso, a friendly AI assistant."
        " Say hello and introduce yourself briefly."
    )

    def get_instructions(self):
        return self.text


agent = Agent(
    "anthropic:claude-haiku-3-5",
    defer_model_check=True,
    capabilities=[
        SystemPrompt(),
        TaskList(),
        Goal(),
    ],
)
