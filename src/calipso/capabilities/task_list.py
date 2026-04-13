from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import FunctionToolset


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
class TaskList(AbstractCapability[Any]):
    tasks: list[Task] = field(default_factory=list)
    _next_id: int = field(init=False, repr=False, default=1)
    _toolset: FunctionToolset[Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.tasks:
            self._next_id = max(t.id for t in self.tasks) + 1

        self._toolset = FunctionToolset()

        @self._toolset.tool_plain
        def create_task(description: str) -> str:
            """Create a new task with the given description."""
            task = Task(id=self._next_id, description=description)
            self.tasks.append(task)
            self._next_id += 1
            return f"Created task {task.id}: {description}"

        @self._toolset.tool_plain
        def update_task_status(task_id: int, status: str) -> str:
            """Update a task's status. Valid statuses: pending, in_progress, done."""
            try:
                new_status = TaskStatus(status)
            except ValueError:
                valid = ", ".join(s.value for s in TaskStatus)
                return f"Invalid status '{status}'. Valid statuses: {valid}"

            for task in self.tasks:
                if task.id == task_id:
                    task.status = new_status
                    return f"Task {task_id} status updated to {status}"

            return f"Task {task_id} not found"

        @self._toolset.tool_plain
        def remove_task(task_id: int) -> str:
            """Remove a task by its ID."""
            for i, task in enumerate(self.tasks):
                if task.id == task_id:
                    self.tasks.pop(i)
                    return f"Removed task {task_id}"

            return f"Task {task_id} not found"

    def _render(self) -> str:
        if not self.tasks:
            return "Tasks: (none)"

        lines = ["Tasks:"]
        for task in self.tasks:
            icon = _STATUS_ICONS[task.status]
            lines.append(f"  {icon} {task.id}. {task.description}")
        return "\n".join(lines)

    def get_instructions(self):
        return self._render

    def get_toolset(self) -> FunctionToolset[Any]:
        return self._toolset
