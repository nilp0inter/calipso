from dataclasses import dataclass, field
from typing import Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import FunctionToolset


@dataclass
class Goal(AbstractCapability[Any]):
    text: str | None = None
    _toolset: FunctionToolset[Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._toolset = FunctionToolset()

        @self._toolset.tool_plain
        def set_goal(goal: str) -> str:
            """Set or update the current goal."""
            self.text = goal
            return f"Goal set: {goal}"

        @self._toolset.tool_plain
        def clear_goal() -> str:
            """Clear the current goal."""
            self.text = None
            return "Goal cleared"

    def _render(self) -> str:
        if self.text is None:
            return "Goal: No goal set"
        return f"Goal: {self.text}"

    def get_instructions(self):
        return self._render

    def get_toolset(self) -> FunctionToolset[Any]:
        return self._toolset
