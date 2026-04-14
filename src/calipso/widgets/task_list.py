"""TaskList widget — tracks tasks with statuses."""

import html as html_mod
from collections.abc import Iterator
from dataclasses import dataclass, replace
from enum import StrEnum

from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.tools import ToolDefinition

from calipso.widget import WidgetHandle, create_widget


class TaskStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"


_STATUS_ICONS = {
    TaskStatus.PENDING: "[ ]",
    TaskStatus.IN_PROGRESS: "[~]",
    TaskStatus.DONE: "[x]",
}


# --- Model ---


@dataclass(frozen=True)
class Task:
    id: int
    description: str
    status: TaskStatus = TaskStatus.PENDING


@dataclass(frozen=True)
class TaskListModel:
    tasks: tuple[Task, ...] = ()
    next_id: int = 1


# --- Messages ---


@dataclass(frozen=True)
class CreateTask:
    description: str


@dataclass(frozen=True)
class UpdateTaskStatus:
    task_id: int
    status: TaskStatus


@dataclass(frozen=True)
class RemoveTask:
    task_id: int


TaskListMsg = CreateTask | UpdateTaskStatus | RemoveTask


# --- Update (pure) ---


def update(model: TaskListModel, msg: TaskListMsg) -> tuple[TaskListModel, str]:
    match msg:
        case CreateTask(description=desc):
            task = Task(id=model.next_id, description=desc)
            return (
                replace(model, tasks=(*model.tasks, task), next_id=model.next_id + 1),
                f"Created task {task.id}: {task.description}",
            )
        case UpdateTaskStatus(task_id=tid, status=new_status):
            new_tasks = []
            found = False
            for task in model.tasks:
                if task.id == tid:
                    new_tasks.append(replace(task, status=new_status))
                    found = True
                else:
                    new_tasks.append(task)
            if not found:
                return model, f"Task {tid} not found"
            return (
                replace(model, tasks=tuple(new_tasks)),
                f"Task {tid} status updated to {new_status}",
            )
        case RemoveTask(task_id=tid):
            new_tasks = tuple(t for t in model.tasks if t.id != tid)
            if len(new_tasks) == len(model.tasks):
                return model, f"Task {tid} not found"
            return replace(model, tasks=new_tasks), f"Removed task {tid}"


# --- Views ---

_TOOL_DEFS = [
    ToolDefinition(
        name="create_task",
        description="Create a new task with the given description.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "The task description.",
                },
            },
            "required": ["description"],
        },
    ),
    ToolDefinition(
        name="update_task_status",
        description=(
            "Update a task's status. Valid statuses: pending, in_progress, done."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "task_id": {"type": "integer", "description": "The task ID."},
                "status": {
                    "type": "string",
                    "description": "New status.",
                    "enum": ["pending", "in_progress", "done"],
                },
            },
            "required": ["task_id", "status"],
        },
    ),
    ToolDefinition(
        name="remove_task",
        description="Remove a task by its ID.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "task_id": {"type": "integer", "description": "The task ID."},
            },
            "required": ["task_id"],
        },
    ),
]


def view_messages(model: TaskListModel) -> Iterator[ModelMessage]:
    if not model.tasks:
        text = "## Tasks\n(none)"
    else:
        lines = ["## Tasks"]
        for task in model.tasks:
            icon = _STATUS_ICONS[task.status]
            lines.append(f"{icon} {task.id}. {task.description}")
        text = "\n".join(lines)
    yield ModelRequest(parts=[UserPromptPart(content=text)])


def view_tools(model: TaskListModel) -> Iterator[ToolDefinition]:
    yield from _TOOL_DEFS


def view_html(model: TaskListModel) -> str:
    if not model.tasks:
        items = "<p><em>No tasks</em></p>"
    else:
        lines = []
        for task in model.tasks:
            checked = " checked" if task.status == TaskStatus.DONE else ""
            indeterminate = (
                " data-indeterminate" if task.status == TaskStatus.IN_PROGRESS else ""
            )
            desc = html_mod.escape(task.description)
            toggle = (
                "sendWidgetEvent("
                "'update_task_status', "
                f"{{task_id: {task.id}, "
                "status: this.checked ? 'done' : 'pending'})"
            )
            remove = f"sendWidgetEvent('remove_task', {{task_id: {task.id}}})"
            lines.append(
                f'<li class="task-{task.status}">'
                f"<label>"
                f'<input type="checkbox"'
                f"{checked}{indeterminate}"
                f' onchange="{toggle}">'
                f" {desc}"
                f"</label>"
                f' <button onclick="{remove}"'
                f' class="btn-remove"'
                f' title="Remove task">x</button>'
                f"</li>"
            )
        items = "<ul>" + "".join(lines) + "</ul>"
    return f'<div id="widget-task-list" class="widget"><h3>Tasks</h3>{items}</div>'


# --- Anticorruption layers ---


async def from_llm(model: TaskListModel, tool_name: str, args: dict) -> TaskListMsg:
    match tool_name:
        case "create_task":
            return CreateTask(description=args["description"])
        case "update_task_status":
            try:
                status = TaskStatus(args["status"])
            except ValueError:
                valid = ", ".join(s.value for s in TaskStatus)
                raise ValueError(f"Invalid status '{args['status']}'. Valid: {valid}")
            return UpdateTaskStatus(task_id=args["task_id"], status=status)
        case "remove_task":
            return RemoveTask(task_id=args["task_id"])
    raise ValueError(f"TaskList: unknown tool '{tool_name}'")


def from_ui(model: TaskListModel, event_name: str, args: dict) -> TaskListMsg | None:
    match event_name:
        case "update_task_status":
            try:
                status = TaskStatus(args["status"])
            except ValueError:
                return None
            return UpdateTaskStatus(task_id=args["task_id"], status=status)
        case "remove_task":
            return RemoveTask(task_id=args["task_id"])
    return None


# --- Factory ---


def create_task_list() -> WidgetHandle:
    return create_widget(
        id="widget-task-list",
        model=TaskListModel(),
        update=update,
        view_messages=view_messages,
        view_tools=view_tools,
        view_html=view_html,
        from_llm=from_llm,
        from_ui=from_ui,
        frontend_tools=frozenset({"update_task_status", "remove_task"}),
    )
