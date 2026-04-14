"""Tests for individual widgets — view rendering and update logic."""

from pydantic_ai import models
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
)

from calipso.widgets.action_log import ActionLog, LogEntry
from calipso.widgets.goal import Goal
from calipso.widgets.system_prompt import SystemPrompt
from calipso.widgets.task_list import TaskList, TaskStatus

models.ALLOW_MODEL_REQUESTS = False


# --- SystemPrompt ---


class TestSystemPrompt:
    def test_view_messages_yields_system_prompt(self):
        w = SystemPrompt(text="Be helpful.")
        msgs = list(w.view_messages())
        assert len(msgs) == 1
        assert isinstance(msgs[0], ModelRequest)
        assert isinstance(msgs[0].parts[0], SystemPromptPart)
        assert msgs[0].parts[0].content == "Be helpful."

    def test_view_tools_is_empty(self):
        w = SystemPrompt()
        assert list(w.view_tools()) == []


# --- Goal ---


class TestGoal:
    def test_view_no_goal(self):
        w = Goal()
        msgs = list(w.view_messages())
        assert len(msgs) == 1
        assert "No goal set" in msgs[0].parts[0].content

    def test_view_with_goal(self):
        w = Goal(text="Fix the bug")
        msgs = list(w.view_messages())
        assert "Fix the bug" in msgs[0].parts[0].content

    def test_set_goal(self):
        w = Goal()
        result = w.update("set_goal", {"goal": "Deploy v2"})
        assert "Deploy v2" in result
        assert w.text == "Deploy v2"
        msgs = list(w.view_messages())
        assert "Deploy v2" in msgs[0].parts[0].content

    def test_clear_goal(self):
        w = Goal(text="Something")
        result = w.update("clear_goal", {})
        assert "cleared" in result.lower()
        assert w.text is None

    def test_view_tools(self):
        w = Goal()
        tools = list(w.view_tools())
        names = {t.name for t in tools}
        assert names == {"set_goal", "clear_goal"}


# --- TaskList ---


class TestTaskList:
    def test_view_empty(self):
        w = TaskList()
        msgs = list(w.view_messages())
        assert "none" in msgs[0].parts[0].content.lower()

    def test_create_and_view(self):
        w = TaskList()
        w.update("create_task", {"description": "Write tests"})
        w.update("create_task", {"description": "Deploy"})
        msgs = list(w.view_messages())
        content = msgs[0].parts[0].content
        assert "Write tests" in content
        assert "Deploy" in content
        assert "[ ]" in content  # pending icon

    def test_update_status(self):
        w = TaskList()
        w.update("create_task", {"description": "Task A"})
        w.update("update_task_status", {"task_id": 1, "status": "done"})
        assert w.tasks[0].status == TaskStatus.DONE
        msgs = list(w.view_messages())
        assert "[x]" in msgs[0].parts[0].content

    def test_remove_task(self):
        w = TaskList()
        w.update("create_task", {"description": "Temp"})
        result = w.update("remove_task", {"task_id": 1})
        assert "Removed" in result
        assert len(w.tasks) == 0

    def test_invalid_status(self):
        w = TaskList()
        w.update("create_task", {"description": "X"})
        result = w.update("update_task_status", {"task_id": 1, "status": "bad"})
        assert "Invalid" in result

    def test_view_tools(self):
        w = TaskList()
        tools = list(w.view_tools())
        names = {t.name for t in tools}
        assert names == {"create_task", "update_task_status", "remove_task"}


# --- ActionLog ---


class TestActionLog:
    def test_view_messages_empty(self):
        w = ActionLog()
        msgs = list(w.view_messages())
        # Should have the rules as a system prompt
        assert len(msgs) == 1
        assert "Action Log" in msgs[0].parts[0].content

    def test_view_messages_with_entries(self):
        w = ActionLog(entries=[LogEntry(id=1, action="Read file", result="Done")])
        msgs = list(w.view_messages())
        assert len(msgs) == 2  # rules + 1 entry
        assert isinstance(msgs[1], ModelResponse)
        assert "Read file" in msgs[1].parts[0].content

    def test_protocol_start_end(self):
        w = ActionLog()
        w.update("action_log_start", {"description": "Read file"})
        assert w._active_action == "Read file"
        result = w.update("action_log_end", {"result": "File read OK"})
        assert "logged" in result.lower()
        assert len(w.entries) == 1
        assert w._active_action is None

    def test_check_protocol_no_active_action(self):
        w = ActionLog()
        error = w.check_protocol("some_tool")
        assert error is not None
        assert "action_log_start" in error

    def test_check_protocol_allows_start(self):
        w = ActionLog()
        assert w.check_protocol("action_log_start") is None

    def test_check_protocol_end_without_start(self):
        w = ActionLog()
        error = w.check_protocol("action_log_end")
        assert error is not None

    def test_check_protocol_tool_type_lock(self):
        w = ActionLog()
        w.update("action_log_start", {"description": "Do stuff"})
        w.track_tool("tool_a")
        assert w.check_protocol("tool_a") is None
        error = w.check_protocol("tool_b")
        assert error is not None
        assert "tool_a" in error

    def test_view_tools(self):
        w = ActionLog()
        tools = list(w.view_tools())
        names = {t.name for t in tools}
        assert names == {"action_log_start", "action_log_end"}
