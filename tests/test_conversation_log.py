"""Tests for the merged ConversationLog widget (task protocol)."""

import pytest
from pydantic_ai import models
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)

from calipso.widgets.conversation_log import (
    ConsumePicks,
    ConversationLogModel,
    ResponseReceived,
    TaskStatus,
    ToolResultsReceived,
    UserMessageReceived,
    check_protocol,
    create_conversation_log,
    current_owning_task_id,
    view_tools,
)

models.ALLOW_MODEL_REQUESTS = False
pytestmark = pytest.mark.anyio

FREE: frozenset[str] = frozenset({"set_goal", "clear_goal"})


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    async def test_create_start_memory_close_happy_path(self):
        w = create_conversation_log()
        await w.dispatch_llm("create_task", {"description": "Explore"})
        assert w.model.tasks[1].status == TaskStatus.PENDING

        await w.dispatch_llm("start_task", {"task_id": 1})
        assert w.model.active_task_id == 1
        assert w.model.tasks[1].status == TaskStatus.IN_PROGRESS

        await w.dispatch_llm("task_memory", {"text": "First note"})
        await w.dispatch_llm("task_memory", {"text": "Second note"})
        assert w.model.tasks[1].memories == ["First note", "Second note"]

        result = await w.dispatch_llm("close_current_task", {})
        assert "Closed" in result
        assert w.model.active_task_id is None
        assert w.model.tasks[1].status == TaskStatus.DONE

    async def test_start_rejected_while_another_active(self):
        w = create_conversation_log()
        await w.dispatch_llm("create_task", {"description": "A"})
        await w.dispatch_llm("create_task", {"description": "B"})
        await w.dispatch_llm("start_task", {"task_id": 1})
        result = await w.dispatch_llm("start_task", {"task_id": 2})
        assert "in_progress" in result.lower() or "close_current_task" in result

    async def test_start_rejected_on_done_task(self):
        w = create_conversation_log()
        await w.dispatch_llm("create_task", {"description": "X"})
        await w.dispatch_llm("start_task", {"task_id": 1})
        await w.dispatch_llm("task_memory", {"text": "n"})
        await w.dispatch_llm("close_current_task", {})
        result = await w.dispatch_llm("start_task", {"task_id": 1})
        assert "pending" in result.lower() or "cannot be started" in result.lower()

    async def test_close_rejected_without_memory(self):
        w = create_conversation_log()
        await w.dispatch_llm("create_task", {"description": "X"})
        await w.dispatch_llm("start_task", {"task_id": 1})
        result = await w.dispatch_llm("close_current_task", {})
        assert "memory" in result.lower()
        assert w.model.active_task_id == 1


# ---------------------------------------------------------------------------
# Remove / frozen tasks
# ---------------------------------------------------------------------------


class TestRemove:
    async def test_remove_pending_succeeds(self):
        w = create_conversation_log()
        await w.dispatch_llm("create_task", {"description": "X"})
        result = await w.dispatch_llm("remove_task", {"task_id": 1})
        assert "Removed" in result
        assert 1 not in w.model.tasks

    async def test_remove_in_progress_rejected(self):
        w = create_conversation_log()
        await w.dispatch_llm("create_task", {"description": "X"})
        await w.dispatch_llm("start_task", {"task_id": 1})
        result = await w.dispatch_llm("remove_task", {"task_id": 1})
        assert "pending" in result.lower()
        assert 1 in w.model.tasks

    async def test_remove_done_rejected(self):
        w = create_conversation_log()
        await w.dispatch_llm("create_task", {"description": "X"})
        await w.dispatch_llm("start_task", {"task_id": 1})
        await w.dispatch_llm("task_memory", {"text": "n"})
        await w.dispatch_llm("close_current_task", {})
        result = await w.dispatch_llm("remove_task", {"task_id": 1})
        assert "pending" in result.lower()

    async def test_remove_pending_via_ui(self):
        w = create_conversation_log()
        await w.dispatch_llm("create_task", {"description": "X"})
        await w.dispatch_ui("remove_task", {"task_id": 1})
        assert 1 not in w.model.tasks


# ---------------------------------------------------------------------------
# check_protocol
# ---------------------------------------------------------------------------


class TestCheckProtocol:
    def test_create_task_always_allowed(self):
        w = create_conversation_log()
        assert check_protocol(w.model, "create_task", FREE) is None

    def test_task_pick_always_allowed(self):
        w = create_conversation_log()
        assert check_protocol(w.model, "task_pick", FREE) is None

    def test_other_tool_rejected_without_task(self):
        w = create_conversation_log()
        err = check_protocol(w.model, "open_file", FREE)
        assert err is not None
        assert "outside a task" in err

    def test_other_tool_allowed_with_task(self):
        w = create_conversation_log()
        w.model.active_task_id = 1
        w.model.tasks[1] = _make_task(1, TaskStatus.IN_PROGRESS)
        assert check_protocol(w.model, "open_file", FREE) is None

    def test_goal_tool_allowed_without_task(self):
        w = create_conversation_log()
        assert check_protocol(w.model, "set_goal", FREE) is None

    def test_task_memory_rejected_without_task(self):
        w = create_conversation_log()
        err = check_protocol(w.model, "task_memory", FREE)
        assert err is not None

    def test_close_current_task_rejected_without_memory(self):
        w = create_conversation_log()
        w.model.active_task_id = 1
        w.model.tasks[1] = _make_task(1, TaskStatus.IN_PROGRESS)
        err = check_protocol(w.model, "close_current_task", FREE)
        assert err is not None
        assert "memory" in err.lower()

    def test_close_current_task_allowed_with_memory(self):
        w = create_conversation_log()
        w.model.active_task_id = 1
        t = _make_task(1, TaskStatus.IN_PROGRESS)
        t.memories.append("n")
        w.model.tasks[1] = t
        assert check_protocol(w.model, "close_current_task", FREE) is None

    def test_start_task_rejected_while_active(self):
        w = create_conversation_log()
        w.model.active_task_id = 1
        w.model.tasks[1] = _make_task(1, TaskStatus.IN_PROGRESS)
        err = check_protocol(w.model, "start_task", FREE)
        assert err is not None


def _make_task(i: int, status: TaskStatus):
    from calipso.widgets.conversation_log import Task

    return Task(id=i, description=f"task {i}", status=status)


# ---------------------------------------------------------------------------
# view_tools gating
# ---------------------------------------------------------------------------


class TestViewToolsGating:
    def test_empty_state(self):
        names = {t.name for t in view_tools(ConversationLogModel())}
        # Only create_task is exposed (no pending, no done, no active).
        assert names == {"create_task"}

    def test_with_pending_no_active(self):
        m = ConversationLogModel()
        m.tasks[1] = _make_task(1, TaskStatus.PENDING)
        m.task_order = [1]
        names = {t.name for t in view_tools(m)}
        assert names == {"create_task", "remove_task", "start_task"}

    def test_with_active_no_memory(self):
        m = ConversationLogModel()
        m.tasks[1] = _make_task(1, TaskStatus.IN_PROGRESS)
        m.task_order = [1]
        m.active_task_id = 1
        names = {t.name for t in view_tools(m)}
        assert names == {"create_task", "task_memory"}

    def test_with_active_and_memory(self):
        m = ConversationLogModel()
        t = _make_task(1, TaskStatus.IN_PROGRESS)
        t.memories.append("n")
        m.tasks[1] = t
        m.task_order = [1]
        m.active_task_id = 1
        names = {t.name for t in view_tools(m)}
        assert names == {"create_task", "task_memory", "close_current_task"}

    def test_with_done_exposes_pick(self):
        m = ConversationLogModel()
        m.tasks[1] = _make_task(1, TaskStatus.DONE)
        m.task_order = [1]
        names = {t.name for t in view_tools(m)}
        assert "task_pick" in names


# ---------------------------------------------------------------------------
# view_messages — rendering
# ---------------------------------------------------------------------------


class TestViewMessages:
    def test_empty_has_rules_only(self):
        w = create_conversation_log()
        msgs = list(w.view_messages())
        assert len(msgs) == 1
        part = msgs[0].parts[0]
        assert isinstance(part, SystemPromptPart)
        assert "Task Protocol" in part.content

    def test_user_message_yielded(self):
        w = create_conversation_log()
        w.send(UserMessageReceived(text="Hi"))
        msgs = list(w.view_messages())
        user_msgs = [
            m
            for m in msgs
            if isinstance(m, ModelRequest)
            and any(
                isinstance(p, UserPromptPart) and p.content == "Hi" for p in m.parts
            )
        ]
        assert len(user_msgs) == 1

    async def test_in_progress_task_renders_raw(self):
        w = create_conversation_log()
        await w.dispatch_llm("create_task", {"description": "X"})
        await w.dispatch_llm("start_task", {"task_id": 1})
        # Simulate a response recorded while in_progress.
        resp = ModelResponse(parts=[TextPart(content="thinking...")])
        w.send(ResponseReceived(response=resp, owning_task_id=1))
        msgs = list(w.view_messages())
        assert resp in msgs

    async def test_done_task_collapsed_by_default(self):
        w = create_conversation_log()
        await w.dispatch_llm("create_task", {"description": "Audit"})
        await w.dispatch_llm("start_task", {"task_id": 1})
        resp = ModelResponse(parts=[TextPart(content="intermediate")])
        w.send(ResponseReceived(response=resp, owning_task_id=1))
        await w.dispatch_llm("task_memory", {"text": "The answer is 42."})
        await w.dispatch_llm("close_current_task", {})

        msgs = list(w.view_messages())
        # The raw response should not appear.
        assert resp not in msgs
        # The collapsed block should appear as a system prompt.
        collapsed = [
            p.content
            for m in msgs
            if isinstance(m, ModelRequest)
            for p in m.parts
            if isinstance(p, SystemPromptPart) and "# Task: Audit" in p.content
        ]
        assert len(collapsed) == 1
        assert "The answer is 42." in collapsed[0]
        assert "task_pick(task_id=1)" in collapsed[0]

    async def test_picked_done_task_shows_full_log(self):
        w = create_conversation_log()
        await w.dispatch_llm("create_task", {"description": "Audit"})
        await w.dispatch_llm("start_task", {"task_id": 1})
        resp = ModelResponse(parts=[TextPart(content="intermediate")])
        w.send(ResponseReceived(response=resp, owning_task_id=1))
        await w.dispatch_llm("task_memory", {"text": "note"})
        await w.dispatch_llm("close_current_task", {})
        await w.dispatch_llm("task_pick", {"task_id": 1})

        msgs = list(w.view_messages())
        # Full raw response should now appear alongside the collapsed block.
        assert resp in msgs
        collapsed = [
            p.content
            for m in msgs
            if isinstance(m, ModelRequest)
            for p in m.parts
            if isinstance(p, SystemPromptPart) and "# Task: Audit" in p.content
        ]
        assert len(collapsed) == 1
        assert "expanded" in collapsed[0].lower() or "full log" in collapsed[0].lower()

    async def test_open_tasks_block_lists_pending_and_in_progress(self):
        w = create_conversation_log()
        await w.dispatch_llm("create_task", {"description": "Todo A"})
        await w.dispatch_llm("create_task", {"description": "Todo B"})
        await w.dispatch_llm("start_task", {"task_id": 1})

        msgs = list(w.view_messages())
        open_blocks = [
            p.content
            for m in msgs
            if isinstance(m, ModelRequest)
            for p in m.parts
            if isinstance(p, UserPromptPart) and "Open tasks" in p.content
        ]
        assert len(open_blocks) == 1
        block = open_blocks[0]
        assert "Todo A" in block
        assert "Todo B" in block
        assert "[~]" in block  # in_progress icon
        assert "[ ]" in block  # pending icon

    async def test_done_tasks_not_listed_in_open_tasks_block(self):
        w = create_conversation_log()
        await w.dispatch_llm("create_task", {"description": "Done one"})
        await w.dispatch_llm("start_task", {"task_id": 1})
        await w.dispatch_llm("task_memory", {"text": "n"})
        await w.dispatch_llm("close_current_task", {})

        msgs = list(w.view_messages())
        open_blocks = [
            p.content
            for m in msgs
            if isinstance(m, ModelRequest)
            for p in m.parts
            if isinstance(p, UserPromptPart) and "Open tasks" in p.content
        ]
        # No open tasks → no block.
        assert open_blocks == []


# ---------------------------------------------------------------------------
# Picks
# ---------------------------------------------------------------------------


class TestPicks:
    async def test_pick_rejected_on_pending(self):
        w = create_conversation_log()
        await w.dispatch_llm("create_task", {"description": "X"})
        result = await w.dispatch_llm("task_pick", {"task_id": 1})
        assert "not done" in result.lower()
        assert w.model.picks_for_next_request == frozenset()

    async def test_pick_on_done_sets_flag(self):
        w = create_conversation_log()
        await w.dispatch_llm("create_task", {"description": "X"})
        await w.dispatch_llm("start_task", {"task_id": 1})
        await w.dispatch_llm("task_memory", {"text": "n"})
        await w.dispatch_llm("close_current_task", {})
        await w.dispatch_llm("task_pick", {"task_id": 1})
        assert w.model.picks_for_next_request == frozenset({1})

    def test_consume_picks_clears(self):
        w = create_conversation_log()
        w.model.picks_for_next_request = frozenset({1, 2})
        w.send(ConsumePicks())
        assert w.model.picks_for_next_request == frozenset()


# ---------------------------------------------------------------------------
# Log tagging
# ---------------------------------------------------------------------------


class TestLogTagging:
    async def test_mid_task_user_message_tagged(self):
        w = create_conversation_log()
        await w.dispatch_llm("create_task", {"description": "X"})
        await w.dispatch_llm("start_task", {"task_id": 1})
        w.send(UserMessageReceived(text="mid-task input"))
        tid = current_owning_task_id(w.model)
        assert tid == 1
        tagged = [i for i in w.model.log if i.user_message == "mid-task input"]
        assert len(tagged) == 1
        assert tagged[0].owning_task_id == 1

    async def test_free_user_message_untagged(self):
        w = create_conversation_log()
        w.send(UserMessageReceived(text="before any task"))
        tagged = [i for i in w.model.log if i.user_message == "before any task"]
        assert tagged[0].owning_task_id is None

    async def test_tool_results_tagged_with_owning_task(self):
        w = create_conversation_log()
        await w.dispatch_llm("create_task", {"description": "X"})
        await w.dispatch_llm("start_task", {"task_id": 1})
        req = ModelRequest(parts=[UserPromptPart(content="(fake tool return)")])
        w.send(ToolResultsReceived(request=req, owning_task_id=1))
        assert any(i.tool_results is req and i.owning_task_id == 1 for i in w.model.log)


# ---------------------------------------------------------------------------
# UI-only messages
# ---------------------------------------------------------------------------


class TestUIOnly:
    async def test_toggle_task_expanded(self):
        w = create_conversation_log()
        await w.dispatch_llm("create_task", {"description": "X"})
        assert w.model.tasks[1].ui_expanded is False
        await w.dispatch_ui("toggle_task_expanded", {"task_id": 1})
        assert w.model.tasks[1].ui_expanded is True
        await w.dispatch_ui("toggle_task_expanded", {"task_id": 1})
        assert w.model.tasks[1].ui_expanded is False

    async def test_edit_memory(self):
        w = create_conversation_log()
        await w.dispatch_llm("create_task", {"description": "X"})
        await w.dispatch_llm("start_task", {"task_id": 1})
        await w.dispatch_llm("task_memory", {"text": "old text"})
        await w.dispatch_ui(
            "edit_memory", {"task_id": 1, "index": 0, "new_text": "new text"}
        )
        assert w.model.tasks[1].memories == ["new text"]

    async def test_remove_memory(self):
        w = create_conversation_log()
        await w.dispatch_llm("create_task", {"description": "X"})
        await w.dispatch_llm("start_task", {"task_id": 1})
        await w.dispatch_llm("task_memory", {"text": "a"})
        await w.dispatch_llm("task_memory", {"text": "b"})
        await w.dispatch_ui("remove_memory", {"task_id": 1, "index": 0})
        assert w.model.tasks[1].memories == ["b"]

    async def test_ui_memory_allowed_on_done_task(self):
        """UI can edit memories on a done task; LLM cannot."""
        w = create_conversation_log()
        await w.dispatch_llm("create_task", {"description": "X"})
        await w.dispatch_llm("start_task", {"task_id": 1})
        await w.dispatch_llm("task_memory", {"text": "orig"})
        await w.dispatch_llm("close_current_task", {})
        # LLM memory tool check is gated at the protocol level, but the message
        # itself, when dispatched via UI, can still target the done task.
        await w.dispatch_ui("task_memory", {"text": "user addition", "task_id": 1})
        assert "user addition" in w.model.tasks[1].memories

    async def test_frontend_tools_registered(self):
        w = create_conversation_log()
        expected = {
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
        assert w.frontend_tools() == expected
