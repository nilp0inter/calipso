"""Goal widget — keeps the agent focused on a specific objective."""

from collections.abc import Iterator
from dataclasses import dataclass, field

from pydantic_ai.messages import ModelMessage, ModelRequest, SystemPromptPart
from pydantic_ai.tools import ToolDefinition

from calipso.widget import Widget


@dataclass
class Goal(Widget):
    text: str | None = None
    _tool_defs: list[ToolDefinition] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._tool_defs = [
            ToolDefinition(
                name="set_goal",
                description="Set or update the current goal.",
                parameters_json_schema={
                    "type": "object",
                    "properties": {
                        "goal": {
                            "type": "string",
                            "description": "The goal text.",
                        },
                    },
                    "required": ["goal"],
                },
            ),
            ToolDefinition(
                name="clear_goal",
                description="Clear the current goal.",
                parameters_json_schema={"type": "object", "properties": {}},
            ),
        ]

    def view_messages(self) -> Iterator[ModelMessage]:
        if self.text is None:
            text = "## Goal\nNo goal set"
        else:
            text = f"## Goal\n{self.text}"
        yield ModelRequest(parts=[SystemPromptPart(content=text)])

    def view_tools(self) -> Iterator[ToolDefinition]:
        yield from self._tool_defs

    def update(self, tool_name: str, args: dict) -> str:
        if tool_name == "set_goal":
            self.text = args["goal"]
            return f"Goal set: {self.text}"
        if tool_name == "clear_goal":
            self.text = None
            return "Goal cleared"
        raise NotImplementedError(f"Goal does not handle tool '{tool_name}'")
