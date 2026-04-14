"""Goal widget — keeps the agent focused on a specific objective."""

import html as html_mod
from collections.abc import Iterator
from dataclasses import dataclass, replace

from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.tools import ToolDefinition

from calipso.widget import WidgetHandle, create_widget, render_md

# --- Model ---


@dataclass(frozen=True)
class GoalModel:
    text: str | None = None


# --- Messages ---


@dataclass(frozen=True)
class SetGoal:
    goal: str


@dataclass(frozen=True)
class ClearGoal:
    pass


GoalMsg = SetGoal | ClearGoal


# --- Update (pure) ---


def update(model: GoalModel, msg: GoalMsg) -> tuple[GoalModel, str]:
    match msg:
        case SetGoal(goal=goal):
            return replace(model, text=goal), f"Goal set: {goal}"
        case ClearGoal():
            return replace(model, text=None), "Goal cleared"


# --- Views ---

_TOOL_DEFS = [
    ToolDefinition(
        name="set_goal",
        description=(
            "Set the overall goal for the conversation. "
            "Use once at the start to establish the objective. "
            "Do not change it mid-conversation."
        ),
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
        description="Clear the goal only after it has been fully fulfilled.",
        parameters_json_schema={"type": "object", "properties": {}},
    ),
]


def view_messages(model: GoalModel) -> Iterator[ModelMessage]:
    if model.text is None:
        text = "## Goal\nNo goal set"
    else:
        text = f"## Goal\n{model.text}"
    yield ModelRequest(parts=[UserPromptPart(content=text)])


def view_tools(model: GoalModel) -> Iterator[ToolDefinition]:
    yield from _TOOL_DEFS


def view_html(model: GoalModel) -> str:
    if model.text is None:
        content = "<em>No goal set</em>"
    else:
        content = render_md(model.text)
    escaped = html_mod.escape(model.text or "", quote=True)
    form = (
        '<div class="goal-edit">'
        '<input type="text" class="goal-input"'
        f' value="{escaped}"'
        ' placeholder="Set a goal..."'
        " onkeydown=\"if(event.key==='Enter'){"
        "sendWidgetEvent('set_goal',"
        '{goal:this.value});}"'
        ">"
        ' <button onclick="'
        "sendWidgetEvent('clear_goal', {})"
        '">Clear</button>'
        "</div>"
    )
    return f'<div id="widget-goal" class="widget"><h3>Goal</h3>{content}{form}</div>'


# --- Anticorruption layers ---


async def from_llm(model: GoalModel, tool_name: str, args: dict) -> GoalMsg:
    match tool_name:
        case "set_goal":
            return SetGoal(goal=args["goal"])
        case "clear_goal":
            return ClearGoal()
    raise ValueError(f"Goal: unknown tool '{tool_name}'")


def from_ui(model: GoalModel, event_name: str, args: dict) -> GoalMsg | None:
    match event_name:
        case "set_goal":
            return SetGoal(goal=args["goal"])
        case "clear_goal":
            return ClearGoal()
    return None


# --- Factory ---


def create_goal(text: str | None = None) -> WidgetHandle:
    return create_widget(
        id="widget-goal",
        model=GoalModel(text=text),
        update=update,
        view_messages=view_messages,
        view_tools=view_tools,
        view_html=view_html,
        from_llm=from_llm,
        from_ui=from_ui,
        frontend_tools=frozenset({"set_goal", "clear_goal"}),
    )
