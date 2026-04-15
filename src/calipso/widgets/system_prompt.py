"""SystemPrompt widget — identity and framing text, editable from the UI."""

import html as html_mod
from collections.abc import Iterator
from dataclasses import dataclass, replace

from pydantic_ai.messages import ModelMessage, ModelRequest, SystemPromptPart

from calipso.cmd import Cmd, none
from calipso.widget import WidgetHandle, create_widget

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


# --- Messages ---


@dataclass(frozen=True)
class SetSystemPrompt:
    text: str


@dataclass(frozen=True)
class ResetSystemPrompt:
    pass


SystemPromptMsg = SetSystemPrompt | ResetSystemPrompt


# --- Update (pure) ---


def update(
    model: SystemPromptModel, msg: SystemPromptMsg
) -> tuple[SystemPromptModel, Cmd]:
    match msg:
        case SetSystemPrompt(text=text):
            return replace(model, text=text), none
        case ResetSystemPrompt():
            return replace(model, text=_DEFAULT_TEXT), none


# --- Views ---


def view_messages(model: SystemPromptModel) -> Iterator[ModelMessage]:
    yield ModelRequest(parts=[SystemPromptPart(content=model.text)])


def view_html(model: SystemPromptModel) -> str:
    escaped = html_mod.escape(model.text, quote=True)
    form = (
        '<div class="system-prompt-edit">'
        f'<textarea class="system-prompt-textarea" rows="10">{escaped}</textarea>'
        ' <button onclick="'
        "sendWidgetEvent('set_system_prompt',"
        " {text: this.parentElement.querySelector('textarea').value})"
        '">Save</button>'
        ' <button onclick="'
        "sendWidgetEvent('reset_system_prompt', {})"
        '">Reset</button>'
        "</div>"
    )
    return (
        '<div id="widget-system-prompt" class="widget">'
        f"<h3>System Prompt</h3>"
        f"{form}"
        "</div>"
    )


# --- Anticorruption layers ---


def from_ui(
    model: SystemPromptModel, event_name: str, args: dict
) -> SystemPromptMsg | None:
    match event_name:
        case "set_system_prompt":
            return SetSystemPrompt(text=args["text"])
        case "reset_system_prompt":
            return ResetSystemPrompt()
    return None


# --- Factory ---


def create_system_prompt(text: str = _DEFAULT_TEXT) -> WidgetHandle:
    return create_widget(
        id="widget-system-prompt",
        model=SystemPromptModel(text=text),
        update=update,
        view_messages=view_messages,
        view_html=view_html,
        from_ui=from_ui,
        frontend_tools=frozenset({"set_system_prompt", "reset_system_prompt"}),
    )
