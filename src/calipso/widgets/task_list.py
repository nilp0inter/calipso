"""TaskList widget — tracks tasks with statuses."""

import html as html_mod
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum

from pydantic_ai.messages import ModelMessage, ModelRequest, SystemPromptPart
from pydantic_ai.tools import ToolDefinition

from calipso.widget import Widget


class TaskStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"


_STATUS_ICONS = {
    TaskStatus.PENDING: "[ ]",
    TaskStatus.IN_PROGRESS: "[~]",
    TaskStatus.DONE: "[x]",
}


@dataclass
class Task:
    id: int
    description: str
    status: TaskStatus = TaskStatus.PENDING


@dataclass
class TaskList(Widget):
    tasks: list[Task] = field(default_factory=list)
    _next_id: int = field(init=False, repr=False, default=1)
    _tool_defs: list[ToolDefinition] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.tasks:
            self._next_id = max(t.id for t in self.tasks) + 1

        self._tool_defs = [
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
                    "Update a task's status. "
                    "Valid statuses: pending, in_progress, done."
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

    def view_messages(self) -> Iterator[ModelMessage]:
        if not self.tasks:
            text = "## Tasks\n(none)"
        else:
            lines = ["## Tasks"]
            for task in self.tasks:
                icon = _STATUS_ICONS[task.status]
                lines.append(f"{icon} {task.id}. {task.description}")
            text = "\n".join(lines)
        yield ModelRequest(parts=[SystemPromptPart(content=text)])

    def view_tools(self) -> Iterator[ToolDefinition]:
        yield from self._tool_defs

    def view_html(self) -> str:
        if not self.tasks:
            items = "<p><em>No tasks</em></p>"
        else:
            lines = []
            for task in self.tasks:
                checked = " checked" if task.status == TaskStatus.DONE else ""
                indeterminate = (
                    " data-indeterminate"
                    if task.status == TaskStatus.IN_PROGRESS
                    else ""
                )
                desc = html_mod.escape(task.description)
                lines.append(
                    f'<li class="task-{task.status}">'
                    f"<label>"
                    f'<input type="checkbox" disabled{checked}{indeterminate}>'
                    f" {desc}"
                    f"</label>"
                    f"</li>"
                )
            items = "<ul>" + "".join(lines) + "</ul>"
        return (
            f'<div id="{self.widget_id()}" class="widget"><h3>Tasks</h3>{items}</div>'
        )

    def update(self, tool_name: str, args: dict) -> str:
        if tool_name == "create_task":
            task = Task(id=self._next_id, description=args["description"])
            self.tasks.append(task)
            self._next_id += 1
            return f"Created task {task.id}: {task.description}"

        if tool_name == "update_task_status":
            task_id = args["task_id"]
            try:
                new_status = TaskStatus(args["status"])
            except ValueError:
                valid = ", ".join(s.value for s in TaskStatus)
                return f"Invalid status '{args['status']}'. Valid: {valid}"
            for task in self.tasks:
                if task.id == task_id:
                    task.status = new_status
                    return f"Task {task_id} status updated to {new_status}"
            return f"Task {task_id} not found"

        if tool_name == "remove_task":
            task_id = args["task_id"]
            for i, task in enumerate(self.tasks):
                if task.id == task_id:
                    self.tasks.pop(i)
                    return f"Removed task {task_id}"
            return f"Task {task_id} not found"

        raise NotImplementedError(f"TaskList does not handle tool '{tool_name}'")
