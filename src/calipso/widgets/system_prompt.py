"""SystemPrompt widget — static identity and framing text."""

from collections.abc import Iterator
from dataclasses import dataclass

from pydantic_ai.messages import ModelMessage, ModelRequest, SystemPromptPart

from calipso.widget import WidgetHandle, create_widget, render_md

_DEFAULT_TEXT = (
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


# --- Model ---


@dataclass(frozen=True)
class SystemPromptModel:
    text: str = _DEFAULT_TEXT


# --- Views ---


def view_messages(model: SystemPromptModel) -> Iterator[ModelMessage]:
    yield ModelRequest(parts=[SystemPromptPart(content=model.text)])


def view_html(model: SystemPromptModel) -> str:
    return (
        '<div id="widget-system-prompt" class="widget">'
        f"<h3>System Prompt</h3>"
        f"{render_md(model.text)}"
        "</div>"
    )


# --- Factory ---


def create_system_prompt(text: str = _DEFAULT_TEXT) -> WidgetHandle:
    return create_widget(
        id="widget-system-prompt",
        model=SystemPromptModel(text=text),
        view_messages=view_messages,
        view_html=view_html,
    )
