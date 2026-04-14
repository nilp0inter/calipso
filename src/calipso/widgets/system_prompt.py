"""Static system prompt widget — the trivial case."""

from collections.abc import Iterator
from dataclasses import dataclass

from pydantic_ai.messages import ModelMessage, ModelRequest, SystemPromptPart

from calipso.widget import Widget, render_md


@dataclass
class SystemPrompt(Widget):
    text: str = (
        "You are Calipso, an AI coding assistant.\n"
        "\n"
        "Your context is a live workspace. After the conversation you will see"
        " a CURRENT STATE section with panels showing live state. Each panel"
        " has tools that modify it — when you call a tool, its panel updates"
        " immediately on the next turn. Read the panels to understand current"
        " state, then respond or use tools to make progress.\n"
        "\n"
        "## Workflow\n"
        "\n"
        "When the user gives you a task, follow this workflow:\n"
        "\n"
        "1. **Set the goal** — use `set_goal` to record the objective.\n"
        "2. **Plan tasks** — break the work into concrete steps and create"
        " each one with `create_task`.\n"
        "3. **Execute** — work through tasks one at a time. Update each"
        " task to `in_progress` when you start it and `done` when you"
        " finish. Continue until all tasks are complete."
    )

    def view_messages(self) -> Iterator[ModelMessage]:
        yield ModelRequest(parts=[SystemPromptPart(content=self.text)])

    def view_html(self) -> str:
        return (
            f'<div id="{self.widget_id()}" class="widget">'
            f"<h3>System Prompt</h3>"
            f"{render_md(self.text)}"
            f"</div>"
        )
