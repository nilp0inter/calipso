"""ConversationLog widget — task-partitioned conversation history.

The conversation is partitioned by **tasks**. Each LogItem belongs either
to no task (`owning_task_id is None`) or to a specific task. A task is a
unit of work with status ``PENDING → IN_PROGRESS → DONE``. At most one
task can be ``IN_PROGRESS`` at any time.

While a task is ``IN_PROGRESS``, all messages (user input, model
responses, tool results) are tagged with its id. When the LLM closes the
task (``close_current_task``), the full span is collapsed for future
prompts: only the task description and the LLM-authored **memories**
survive in the LLM's view. The browser can still expand the span
visually, and the LLM can request one-shot re-expansion via
``task_pick(task_id)``.
"""

import html as html_mod
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.tools import ToolDefinition

from calipso.cmd import Cmd, Initiator, for_initiator, none
from calipso.widget import WidgetHandle, create_widget, render_md

# --- Supporting types ---


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
    """A unit of work. Only one task is IN_PROGRESS at a time.

    Mutable: memories are appended during the in_progress phase,
    ui_expanded flips on UI toggles.
    """

    id: int
    description: str
    status: TaskStatus = TaskStatus.PENDING
    memories: list[str] = field(default_factory=list)
    ui_expanded: bool = False  # visual-only; does not affect view_messages


@dataclass
class LogItem:
    """One chronological message. Exactly one payload field is set."""

    user_message: str | None = None
    response: ModelResponse | None = None
    tool_results: ModelRequest | None = None
    owning_task_id: int | None = None


# --- Model ---


@dataclass
class ConversationLogModel:
    """Model for the ConversationLog widget.

    Not frozen: ``log`` and ``tasks[*].memories`` are append-heavy. The
    update function mutates in place and returns the same model reference.
    """

    log: list[LogItem] = field(default_factory=list)
    tasks: dict[int, Task] = field(default_factory=dict)
    task_order: list[int] = field(default_factory=list)
    next_id: int = 1
    active_task_id: int | None = None
    picks_for_next_request: frozenset[int] = frozenset()


# --- Messages ---


@dataclass(frozen=True)
class CreateTask:
    description: str
    initiator: Initiator


@dataclass(frozen=True)
class StartTask:
    task_id: int
    initiator: Initiator


@dataclass(frozen=True)
class TaskMemoryAppend:
    text: str
    target_task_id: int | None  # None = current active task (LLM)
    initiator: Initiator


@dataclass(frozen=True)
class CloseCurrentTask:
    initiator: Initiator


@dataclass(frozen=True)
class PickTask:
    task_id: int
    initiator: Initiator


@dataclass(frozen=True)
class RemoveTask:
    task_id: int
    initiator: Initiator


@dataclass(frozen=True)
class EditMemory:
    task_id: int
    index: int
    new_text: str


@dataclass(frozen=True)
class RemoveMemory:
    task_id: int
    index: int


@dataclass(frozen=True)
class ToggleTaskExpanded:
    task_id: int


@dataclass(frozen=True)
class UpdateTaskStatus:
    """UI catch-all for checkbox-driven PENDING↔DONE transitions."""

    task_id: int
    status: TaskStatus
    initiator: Initiator


# Internal (sent via .send()):


@dataclass(frozen=True)
class UserMessageReceived:
    text: str


@dataclass(frozen=True)
class ResponseReceived:
    response: ModelResponse
    owning_task_id: int | None


@dataclass(frozen=True)
class ToolResultsReceived:
    request: ModelRequest
    owning_task_id: int | None


@dataclass(frozen=True)
class ConsumePicks:
    """Clear picks_for_next_request. Sent by Context at the start of
    handle_response after a request/response round-trip has consumed them.
    """


ConversationLogMsg = (
    CreateTask
    | StartTask
    | TaskMemoryAppend
    | CloseCurrentTask
    | PickTask
    | RemoveTask
    | EditMemory
    | RemoveMemory
    | ToggleTaskExpanded
    | UpdateTaskStatus
    | UserMessageReceived
    | ResponseReceived
    | ToolResultsReceived
    | ConsumePicks
)


# --- Task protocol rules (shown to the LLM as a system prompt) ---


_TASK_PROTOCOL_RULES = (
    "## Task Protocol — conversation memory management\n"
    "\n"
    "Your conversation is partitioned into **tasks**. Every tool call"
    " you make (except task management and goal management) must happen"
    " while a task is `in_progress`. When you close a task with"
    " `close_current_task`, the full span of that task — every tool"
    " call, every tool result, every intermediate response — is"
    " **replaced** for future prompts by the task's **memories**.\n"
    "\n"
    "### Lifecycle\n"
    "1. `create_task(description)` — plan a unit of work (or reuse a"
    " pending one).\n"
    "2. `start_task(task_id)` — mark it `in_progress`. Only one task"
    " can be `in_progress` at a time.\n"
    "3. Do the work — call any tools you need, freely interleaved.\n"
    "4. `task_memory(text)` — **call this aggressively** to record"
    " anything you will need later: findings, file paths, API shapes,"
    " decisions, partial results, dead ends you already explored."
    " Call it multiple times during the task. Memories are"
    " **APPEND-ONLY while the task is in_progress** — after close,"
    " you cannot add more.\n"
    "5. `close_current_task()` — in a response by itself, as the"
    " first and only tool call. Requires at least one memory.\n"
    "\n"
    "### MEMORIES ARE YOUR ONLY DURABLE RECORD\n"
    "After `close_current_task`, the task's conversation becomes a"
    " collapsed block containing only the memories you saved. The raw"
    " tool calls, tool results, and your intermediate thinking are"
    " hidden from future prompts. You cannot undo this. **Save"
    " memories aggressively** — anything relevant to the overall"
    " goal, any non-obvious finding, any decision you made. If you"
    " realise later that you need the hidden detail, you can call"
    " `task_pick(task_id)` to re-expand the full log for one single"
    " request only.\n"
    "\n"
    "### Rules\n"
    "- Every non-task, non-goal tool call must be inside an"
    " `in_progress` task. Calls outside are rejected.\n"
    "- Only one task `in_progress` at a time. Call"
    " `close_current_task` before `start_task` again.\n"
    "- `close_current_task` must be the **first and only** tool call"
    " in its response. Do not call other tools in the same response."
    " You need to have seen all tool results before closing so your"
    " memories are accurate.\n"
    "- `close_current_task` is rejected if the task has zero"
    " memories — you must record at least one memory before"
    " closing.\n"
    "- Tasks with status `in_progress` or `done` cannot be removed."
    " They are frozen records.\n"
    "- `task_pick(task_id)` expands a done task's full log for the"
    " **next request only**, then collapses again.\n"
    "\n"
    "### Example\n"
    "```\n"
    "Response 1: create_task({description: 'Audit the auth middleware'})\n"
    "            start_task({task_id: 1})\n"
    "            open_file({path: 'src/auth/middleware.py'})\n"
    "            ← you receive tool results\n"
    "Response 2: task_memory({text: 'Middleware at src/auth/middleware.py"
    " lines 12-48. Uses jwt.decode() without verifying aud claim.'})\n"
    "            task_memory({text: 'Session store is Redis, keys"
    ' prefixed "sess:".\'})\n'
    "Response 3: close_current_task({})\n"
    "```\n"
    "Note that `close_current_task` is in a response by itself, after"
    " all memories have been saved."
)


# --- Update ---


def _err(init: Initiator, text: str) -> Cmd:
    """Return an error tool_result for LLM, CmdNone for UI."""
    return for_initiator(init, text)


def update(
    model: ConversationLogModel, msg: ConversationLogMsg
) -> tuple[ConversationLogModel, Cmd]:
    match msg:
        case CreateTask(description=desc, initiator=init):
            task = Task(id=model.next_id, description=desc)
            model.tasks[task.id] = task
            model.task_order.append(task.id)
            model.next_id += 1
            return model, for_initiator(init, f"Created task {task.id}: {desc}")

        case StartTask(task_id=tid, initiator=init):
            if model.active_task_id is not None:
                return model, _err(
                    init,
                    f"Cannot start task {tid}: task {model.active_task_id} is"
                    " already in_progress. Call close_current_task first.",
                )
            task = model.tasks.get(tid)
            if task is None:
                return model, _err(init, f"Task {tid} not found")
            if task.status != TaskStatus.PENDING:
                return model, _err(
                    init,
                    f"Task {tid} cannot be started: status is {task.status.value}"
                    f" (only pending tasks can be started).",
                )
            task.status = TaskStatus.IN_PROGRESS
            model.active_task_id = tid
            return model, for_initiator(init, f"Started task {tid}: {task.description}")

        case TaskMemoryAppend(text=text, target_task_id=target, initiator=init):
            # LLM always targets the active task (target is None for LLM).
            # UI may target any task directly.
            if target is None:
                target = model.active_task_id
            if target is None:
                return model, _err(
                    init,
                    "Cannot append memory: no active task.",
                )
            task = model.tasks.get(target)
            if task is None:
                return model, _err(init, f"Task {target} not found")
            # LLM: only allowed on the current in_progress task.
            if init is Initiator.LLM and task.status != TaskStatus.IN_PROGRESS:
                return model, _err(
                    init,
                    f"Cannot append memory to task {target}: memories are"
                    " frozen for the LLM once a task is closed. Only"
                    " the currently in_progress task can receive new"
                    " memories from the LLM.",
                )
            task.memories.append(text)
            return model, for_initiator(init, f"Memory recorded on task {target}.")

        case CloseCurrentTask(initiator=init):
            tid = model.active_task_id
            if tid is None:
                return model, _err(init, "No task is in_progress.")
            task = model.tasks[tid]
            if not task.memories:
                return model, _err(
                    init,
                    "Cannot close the current task without any memory."
                    " Call task_memory(...) first, recording anything"
                    " you need to remember — once the task is closed"
                    " you cannot add more.",
                )
            task.status = TaskStatus.DONE
            model.active_task_id = None
            return model, for_initiator(init, f"Closed task {tid}.")

        case PickTask(task_id=tid, initiator=init):
            task = model.tasks.get(tid)
            if task is None:
                return model, _err(init, f"Task {tid} not found")
            if task.status != TaskStatus.DONE:
                return model, _err(
                    init,
                    f"Task {tid} is not done (status: {task.status.value})."
                    " Only done tasks can be picked for expansion.",
                )
            model.picks_for_next_request = model.picks_for_next_request | {tid}
            return model, for_initiator(
                init,
                f"Task {tid} will be expanded for the next request only.",
            )

        case RemoveTask(task_id=tid, initiator=init):
            task = model.tasks.get(tid)
            if task is None:
                return model, _err(init, f"Task {tid} not found")
            if task.status != TaskStatus.PENDING:
                return model, _err(
                    init,
                    f"Cannot remove task {tid}: only pending tasks can be"
                    " removed. In-progress and done tasks are frozen.",
                )
            del model.tasks[tid]
            model.task_order = [x for x in model.task_order if x != tid]
            return model, for_initiator(init, f"Removed task {tid}.")

        case EditMemory(task_id=tid, index=i, new_text=text):
            task = model.tasks.get(tid)
            if task is None or i < 0 or i >= len(task.memories):
                return model, none
            task.memories[i] = text
            return model, none

        case RemoveMemory(task_id=tid, index=i):
            task = model.tasks.get(tid)
            if task is None or i < 0 or i >= len(task.memories):
                return model, none
            del task.memories[i]
            return model, none

        case ToggleTaskExpanded(task_id=tid):
            task = model.tasks.get(tid)
            if task is None:
                return model, none
            task.ui_expanded = not task.ui_expanded
            return model, none

        case UpdateTaskStatus(task_id=tid, status=new_status, initiator=init):
            task = model.tasks.get(tid)
            if task is None:
                return model, _err(init, f"Task {tid} not found")
            # Transitions into/out of IN_PROGRESS go through start/close logic.
            if new_status == TaskStatus.IN_PROGRESS:
                if task.status == TaskStatus.IN_PROGRESS:
                    return model, for_initiator(
                        init, f"Task {tid} already in_progress."
                    )
                return update(model, StartTask(task_id=tid, initiator=init))
            if (
                task.status == TaskStatus.IN_PROGRESS
                and new_status != TaskStatus.IN_PROGRESS
            ):
                # Attempt to close via UI — must have memories; UI skirts the
                # "must have memory" rule only for PENDING→DONE direct.
                if not task.memories:
                    return model, _err(
                        init,
                        f"Cannot move task {tid} out of in_progress without"
                        " at least one memory. Add a memory first.",
                    )
                task.status = new_status
                model.active_task_id = None
                return model, for_initiator(init, f"Task {tid} → {new_status.value}.")
            task.status = new_status
            return model, for_initiator(init, f"Task {tid} → {new_status.value}.")

        case UserMessageReceived(text=text):
            model.log.append(
                LogItem(user_message=text, owning_task_id=model.active_task_id)
            )
            return model, none

        case ResponseReceived(response=response, owning_task_id=tid):
            model.log.append(LogItem(response=response, owning_task_id=tid))
            return model, none

        case ToolResultsReceived(request=request, owning_task_id=tid):
            model.log.append(LogItem(tool_results=request, owning_task_id=tid))
            return model, none

        case ConsumePicks():
            model.picks_for_next_request = frozenset()
            return model, none


# --- Pure query functions ---


def check_protocol(
    model: ConversationLogModel,
    tool_name: str,
    protocol_free_tools: frozenset[str],
) -> str | None:
    """Check if a tool call is allowed under the task protocol.

    Returns an error message if the call violates the protocol, or None
    if it's allowed.
    """
    # Task-management tools — always callable.
    if tool_name in ("create_task", "task_pick", "remove_task"):
        return None
    if tool_name == "start_task":
        if model.active_task_id is not None:
            return (
                f"Cannot start a task while task {model.active_task_id}"
                " is in_progress. Call close_current_task first."
            )
        return None
    if tool_name == "close_current_task":
        if model.active_task_id is None:
            return "No task is in_progress; nothing to close."
        task = model.tasks.get(model.active_task_id)
        if task is None or not task.memories:
            return (
                "Cannot close the current task without at least one"
                " memory. Call task_memory(...) first, recording"
                " anything you need to remember — once the task is"
                " closed you cannot add more."
            )
        return None
    if tool_name == "task_memory":
        if model.active_task_id is None:
            return (
                "Cannot append memory outside a task. Start a task"
                " first with start_task(task_id)."
            )
        return None
    # Any other tool — allowed iff a task is active or tool is protocol-free.
    if model.active_task_id is None and tool_name not in protocol_free_tools:
        return (
            f"Cannot execute '{tool_name}' outside a task. Start a task"
            " first with start_task(task_id)."
        )
    return None


def current_owning_task_id(model: ConversationLogModel) -> int | None:
    """Return the active task id (the task that will own newly arriving log items)."""
    return model.active_task_id


# --- Tool definitions ---


_TOOL_CREATE_TASK = ToolDefinition(
    name="create_task",
    description=(
        "Create a new pending task with the given description."
        " Does not start the task — call start_task(task_id)"
        " afterwards to begin working on it."
    ),
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
)

_TOOL_START_TASK = ToolDefinition(
    name="start_task",
    description=(
        "Mark a pending task as in_progress. Only one task can be"
        " in_progress at a time. While in_progress, all your"
        " conversation (tool calls, tool results, intermediate"
        " responses) is recorded against this task; on close, it"
        " collapses to the memory list you saved."
    ),
    parameters_json_schema={
        "type": "object",
        "properties": {
            "task_id": {"type": "integer", "description": "The task ID."},
        },
        "required": ["task_id"],
    },
)

_TOOL_TASK_MEMORY = ToolDefinition(
    name="task_memory",
    description=(
        "Record a memory on the current in_progress task. Call this"
        " AGGRESSIVELY during a task — anything relevant to the"
        " overall goal, any surprising finding, any decision, any"
        " key file path or API shape. Memories are your ONLY durable"
        " record of the task. After close, you cannot add more."
    ),
    parameters_json_schema={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": (
                    "The memory to record. Be concrete. Be complete."
                    " Assume the tool-call transcript will be erased."
                ),
            },
        },
        "required": ["text"],
    },
)

_TOOL_CLOSE_CURRENT_TASK = ToolDefinition(
    name="close_current_task",
    description=(
        "Close the in_progress task. Must be the FIRST AND ONLY tool"
        " call in its response. Requires at least one memory recorded"
        " on the task. After close, the task's conversation collapses"
        " to the memory list — the raw tool calls/results are no"
        " longer visible to you. Only call this once you have"
        " recorded every memory you need."
    ),
    parameters_json_schema={"type": "object", "properties": {}},
)

_TOOL_TASK_PICK = ToolDefinition(
    name="task_pick",
    description=(
        "Expand the full log of a done task for the NEXT REQUEST"
        " ONLY. On the request after next, the task collapses again"
        " to its memory block. Use this when the memories don't"
        " cover something you now need from that task."
    ),
    parameters_json_schema={
        "type": "object",
        "properties": {
            "task_id": {"type": "integer", "description": "The task ID."},
        },
        "required": ["task_id"],
    },
)

_TOOL_REMOVE_TASK = ToolDefinition(
    name="remove_task",
    description=(
        "Remove a pending task. Only pending tasks can be removed —"
        " in-progress and done tasks are frozen."
    ),
    parameters_json_schema={
        "type": "object",
        "properties": {
            "task_id": {"type": "integer", "description": "The task ID."},
        },
        "required": ["task_id"],
    },
)


ALL_CONVERSATION_LOG_TOOLS: frozenset[str] = frozenset(
    {
        _TOOL_CREATE_TASK.name,
        _TOOL_START_TASK.name,
        _TOOL_TASK_MEMORY.name,
        _TOOL_CLOSE_CURRENT_TASK.name,
        _TOOL_TASK_PICK.name,
        _TOOL_REMOVE_TASK.name,
    }
)


def view_tools(model: ConversationLogModel) -> Iterator[ToolDefinition]:
    # Always available.
    yield _TOOL_CREATE_TASK
    # remove_task — only if any pending tasks exist.
    if any(t.status == TaskStatus.PENDING for t in model.tasks.values()):
        yield _TOOL_REMOVE_TASK
    # task_pick — only if any done tasks exist.
    if any(t.status == TaskStatus.DONE for t in model.tasks.values()):
        yield _TOOL_TASK_PICK

    if model.active_task_id is None:
        # start_task — only if any pending tasks exist.
        if any(t.status == TaskStatus.PENDING for t in model.tasks.values()):
            yield _TOOL_START_TASK
    else:
        yield _TOOL_TASK_MEMORY
        task = model.tasks.get(model.active_task_id)
        if task is not None and task.memories:
            yield _TOOL_CLOSE_CURRENT_TASK


# --- Collapsed memory rendering ---


def _format_collapsed_block(task: Task, picked: bool) -> str:
    lines = [
        f"# Task: {task.description}",
        f"## Task ID: {task.id}",
        "## Task summary",
    ]
    if task.memories:
        for i, mem in enumerate(task.memories, start=1):
            lines.append(f"### Memory {i}")
            lines.append(mem)
    else:
        lines.append("_(no memories recorded)_")
    lines.append("")
    if picked:
        lines.append(
            "## Notes: You are viewing this task's full log for this"
            " request only; it collapses again on the next request."
        )
    else:
        lines.append(
            "## Notes: This is a collapsed representation of an already"
            " finished task. The details are not shown. If you need"
            " temporary access to this task's details, please execute"
            f" task_pick(task_id={task.id})."
        )
    return "\n".join(lines)


def view_messages(model: ConversationLogModel) -> Iterator[ModelMessage]:
    yield ModelRequest(parts=[SystemPromptPart(content=_TASK_PROTOCOL_RULES)])

    # Walk log grouping by owning_task_id.
    i = 0
    n = len(model.log)
    while i < n:
        group_tid = model.log[i].owning_task_id
        j = i
        while j < n and model.log[j].owning_task_id == group_tid:
            j += 1
        group = model.log[i:j]
        i = j

        if group_tid is None:
            yield from _yield_raw(group)
            continue

        task = model.tasks.get(group_tid)
        if task is None:
            # Task was somehow removed — fall back to raw (defensive).
            yield from _yield_raw(group)
            continue

        if task.status == TaskStatus.IN_PROGRESS:
            yield from _yield_raw(group)
            continue

        # DONE (or PENDING — shouldn't happen, but defensive).
        picked = task.id in model.picks_for_next_request
        yield ModelRequest(
            parts=[SystemPromptPart(content=_format_collapsed_block(task, picked))]
        )
        if picked:
            yield from _yield_raw(group)

    # "## Open tasks" block listing pending + in_progress.
    open_tasks = [
        model.tasks[tid]
        for tid in model.task_order
        if model.tasks[tid].status in (TaskStatus.PENDING, TaskStatus.IN_PROGRESS)
    ]
    if open_tasks:
        lines = ["## Open tasks"]
        for task in open_tasks:
            icon = _STATUS_ICONS[task.status]
            lines.append(f"{icon} {task.id}. {task.description}")
        yield ModelRequest(parts=[UserPromptPart(content="\n".join(lines))])


def _yield_raw(group: list[LogItem]) -> Iterator[ModelMessage]:
    for item in group:
        if item.user_message is not None:
            yield ModelRequest(parts=[UserPromptPart(content=item.user_message)])
        elif item.response is not None:
            yield item.response
        elif item.tool_results is not None:
            yield item.tool_results


# --- HTML view ---


def view_html(model: ConversationLogModel) -> str:
    parts = ['<div class="conversation-stream">']
    parts.extend(_render_log(model))
    parts.append("</div>")
    parts.append(_render_task_panel(model))
    return (
        '<div id="widget-conversation-log" class="widget">'
        f"<h3>Conversation</h3>{''.join(parts)}</div>"
    )


def _render_task_panel(model: ConversationLogModel) -> str:
    """Create-task input + pending tasks overview.

    Only PENDING tasks appear here — in-progress and done tasks are
    rendered inline in the conversation stream.
    """
    pending_tasks = [
        model.tasks[tid]
        for tid in model.task_order
        if model.tasks[tid].status == TaskStatus.PENDING
    ]
    rows = [_render_pending_task_row(task, model) for task in pending_tasks]
    rows_html = "".join(rows) if rows else "<li><em>No pending tasks</em></li>"
    create_input = (
        '<div class="task-add">'
        '<input type="text" class="task-input"'
        ' placeholder="Add a task..."'
        " onkeydown=\"if(event.key==='Enter'){"
        "sendWidgetEvent('create_task',"
        "{description:this.value});this.value='';}\""
        ">"
        "</div>"
    )
    return (
        '<div class="task-panel">'
        f"{create_input}"
        f'<ul class="open-tasks">{rows_html}</ul>'
        "</div>"
    )


def _render_pending_task_row(task: Task, model: ConversationLogModel) -> str:
    icon = html_mod.escape(_STATUS_ICONS[task.status])
    desc = html_mod.escape(task.description)
    controls: list[str] = []
    if model.active_task_id is None:
        controls.append(
            '<button class="btn-start"'
            f" onclick=\"sendWidgetEvent('start_task',{{task_id:{task.id}}})\""
            ">Start</button>"
        )
    controls.append(
        '<button class="btn-remove"'
        f" onclick=\"sendWidgetEvent('remove_task',{{task_id:{task.id}}})\""
        ' title="Remove task">x</button>'
    )
    controls_html = " ".join(controls)
    return (
        f'<li class="task-{task.status.value}">'
        f'<span class="task-row-head">'
        f'<span class="task-icon">{icon}</span>'
        f' <span class="task-id">#{task.id}</span>'
        f" {desc}"
        f' <span class="task-controls">{controls_html}</span>'
        "</span>"
        "</li>"
    )


def _render_memory_block(task: Task) -> str:
    """Render the memory list + add/edit/remove affordances."""
    items = []
    for i, mem in enumerate(task.memories):
        mem_escaped = html_mod.escape(mem, quote=True)
        remove_js = f"sendWidgetEvent('remove_memory',{{task_id:{task.id},index:{i}}})"
        edit_js = (
            "sendWidgetEvent('edit_memory',"
            f"{{task_id:{task.id},index:{i},new_text:this.value}})"
        )
        items.append(
            '<li class="memory-item">'
            f'<input type="text" class="memory-input" value="{mem_escaped}"'
            f' onchange="{edit_js}">'
            f' <button class="btn-remove" onclick="{remove_js}"'
            ' title="Remove memory">x</button>'
            "</li>"
        )
    items_html = "".join(items)
    add_js = (
        "if(event.key==='Enter'){sendWidgetEvent('task_memory',"
        f"{{text:this.value,task_id:{task.id}}});"
        "this.value='';}"
    )
    add_input = (
        '<input type="text" class="memory-add"'
        ' placeholder="Add a memory..."'
        f' onkeydown="{add_js}">'
    )
    return (
        '<div class="memory-block">'
        f'<ul class="memories">{items_html}</ul>'
        f"{add_input}"
        "</div>"
    )


def _render_log(model: ConversationLogModel) -> list[str]:
    out: list[str] = []
    i = 0
    n = len(model.log)
    while i < n:
        group_tid = model.log[i].owning_task_id
        j = i
        while j < n and model.log[j].owning_task_id == group_tid:
            j += 1
        group = model.log[i:j]
        i = j

        if group_tid is None:
            out.extend(_render_group_raw(group))
            continue

        task = model.tasks.get(group_tid)
        if task is None:
            out.extend(_render_group_raw(group))
            continue

        if task.status == TaskStatus.IN_PROGRESS:
            out.append(_render_in_progress_block(task))
            out.extend(_render_group_raw(group))
            continue

        # DONE
        out.append(_render_done_task_block(task, group, model))
    return out


def _render_in_progress_block(task: Task) -> str:
    """Render the in-progress task banner, memories, and close button."""
    desc = html_mod.escape(task.description)
    mem_count = len(task.memories)
    mem_noun = "memory" if mem_count == 1 else "memories"
    close_disabled = "" if task.memories else " disabled"
    close_btn = (
        '<button class="btn-close"'
        f"{close_disabled}"
        " onclick=\"sendWidgetEvent('close_current_task',{})\""
        ">Close task</button>"
    )
    banner = (
        '<div class="in-progress-banner">'
        '<span class="task-row-head">'
        f"Task #{task.id}: {desc} ({mem_count} {mem_noun})"
        f' <span class="task-controls">{close_btn}</span>'
        "</span>"
        "</div>"
    )
    return banner + _render_memory_block(task)


def _render_done_task_block(
    task: Task, group: list[LogItem], model: ConversationLogModel
) -> str:
    open_attr = " open" if task.ui_expanded else ""
    expected = "true" if task.ui_expanded else "false"
    toggle_handler = (
        f"if(this.open!==({expected}))"
        f"sendWidgetEvent('toggle_task_expanded',"
        f"{{task_id:{task.id}}})"
    )
    pick_marker = (
        ' <span class="pick-marker" title="Expanded for next LLM request">🔍</span>'
        if task.id in model.picks_for_next_request
        else ""
    )
    summary = (
        f"<summary>Task #{task.id}: {html_mod.escape(task.description)}"
        f" — {len(task.memories)} memor"
        f"{'y' if len(task.memories) == 1 else 'ies'}"
        f"{pick_marker}"
        "</summary>"
    )
    raw_html = "".join(_render_group_raw(group))
    memories_html = _render_memory_block(task)
    return (
        f'<details class="task-done"{open_attr}'
        f' ontoggle="{toggle_handler}">'
        f"{summary}"
        f'<div class="done-body">'
        f"{raw_html}"
        f"{memories_html}"
        "</div>"
        "</details>"
    )


def _render_group_raw(group: list[LogItem]) -> list[str]:
    out: list[str] = []
    pending_tools: list[str] = []
    pending_calls = 0
    for item in group:
        if item.user_message is not None:
            out.extend(_flush_tools(pending_tools, pending_calls))
            pending_calls = 0
            out.append(
                '<div class="msg sent">'
                '<span class="bubble-label">You</span>'
                f"{render_md(item.user_message)}"
                "</div>"
            )
        elif item.response is not None:
            text_parts, tool_parts = _split_response(item.response)
            out.extend(_flush_tools(pending_tools, pending_calls))
            pending_calls = 0
            out.extend(text_parts)
            pending_tools.extend(tool_parts)
            pending_calls += len(tool_parts)
        elif item.tool_results is not None:
            pending_tools.extend(_render_tool_returns(item.tool_results))
    out.extend(_flush_tools(pending_tools, pending_calls))
    return out


def _flush_tools(pending: list[str], call_count: int) -> list[str]:
    if not pending:
        return []
    noun = "call" if call_count == 1 else "calls"
    out = [
        '<details class="tool-group">'
        f"<summary>{call_count} tool {noun}</summary>"
        f"{''.join(pending)}"
        "</details>"
    ]
    pending.clear()
    return out


def _split_response(response: ModelResponse) -> tuple[list[str], list[str]]:
    text_parts: list[str] = []
    tool_parts: list[str] = []
    for part in response.parts:
        if isinstance(part, TextPart):
            text_parts.append(
                '<div class="msg received assistant">'
                '<span class="bubble-label">Agent</span>'
                f"{render_md(part.content)}"
                "</div>"
            )
        elif isinstance(part, ToolCallPart):
            args = html_mod.escape(str(part.args_as_dict()))
            tool_parts.append(
                '<div class="msg received tool-call">'
                '<span class="dir">&#9660;</span> '
                f"<code>{html_mod.escape(part.tool_name)}({args})</code></div>"
            )
    return text_parts, tool_parts


def _render_tool_returns(req: ModelRequest) -> list[str]:
    out: list[str] = []
    for part in req.parts:
        if isinstance(part, ToolReturnPart):
            out.append(
                '<div class="msg sent tool-result">'
                '<span class="dir">&#9650;</span> '
                f"<code>{html_mod.escape(str(part.content))}</code></div>"
            )
    return out


# --- Anticorruption layers ---


def from_llm(
    model: ConversationLogModel, tool_name: str, args: dict
) -> ConversationLogMsg:
    match tool_name:
        case "create_task":
            return CreateTask(description=args["description"], initiator=Initiator.LLM)
        case "start_task":
            return StartTask(task_id=int(args["task_id"]), initiator=Initiator.LLM)
        case "task_memory":
            return TaskMemoryAppend(
                text=args["text"], target_task_id=None, initiator=Initiator.LLM
            )
        case "close_current_task":
            return CloseCurrentTask(initiator=Initiator.LLM)
        case "task_pick":
            return PickTask(task_id=int(args["task_id"]), initiator=Initiator.LLM)
        case "remove_task":
            return RemoveTask(task_id=int(args["task_id"]), initiator=Initiator.LLM)
    raise ValueError(f"ConversationLog: unknown tool '{tool_name}'")


def from_ui(
    model: ConversationLogModel, event_name: str, args: dict
) -> ConversationLogMsg | None:
    match event_name:
        case "create_task":
            desc = args.get("description", "").strip()
            if not desc:
                return None
            return CreateTask(description=desc, initiator=Initiator.UI)
        case "start_task":
            return StartTask(task_id=int(args["task_id"]), initiator=Initiator.UI)
        case "task_memory":
            text = args.get("text", "").strip()
            if not text:
                return None
            target = args.get("task_id")
            return TaskMemoryAppend(
                text=text,
                target_task_id=int(target) if target is not None else None,
                initiator=Initiator.UI,
            )
        case "close_current_task":
            return CloseCurrentTask(initiator=Initiator.UI)
        case "task_pick":
            return PickTask(task_id=int(args["task_id"]), initiator=Initiator.UI)
        case "remove_task":
            return RemoveTask(task_id=int(args["task_id"]), initiator=Initiator.UI)
        case "edit_memory":
            text = args.get("new_text", "")
            return EditMemory(
                task_id=int(args["task_id"]),
                index=int(args["index"]),
                new_text=text,
            )
        case "remove_memory":
            return RemoveMemory(task_id=int(args["task_id"]), index=int(args["index"]))
        case "toggle_task_expanded":
            return ToggleTaskExpanded(task_id=int(args["task_id"]))
        case "update_task_status":
            try:
                status = TaskStatus(args["status"])
            except ValueError:
                return None
            return UpdateTaskStatus(
                task_id=int(args["task_id"]),
                status=status,
                initiator=Initiator.UI,
            )
    return None


# --- Factory ---


def create_conversation_log() -> WidgetHandle:
    return create_widget(
        id="widget-conversation-log",
        model=ConversationLogModel(),
        update=update,
        view_messages=view_messages,
        view_tools=view_tools,
        view_html=view_html,
        from_llm=from_llm,
        from_ui=from_ui,
        frontend_tools=frozenset(
            {
                "create_task",
                "start_task",
                "task_memory",
                "close_current_task",
                "task_pick",
                "remove_task",
                "edit_memory",
                "remove_memory",
                "toggle_task_expanded",
                "update_task_status",
            }
        ),
    )
