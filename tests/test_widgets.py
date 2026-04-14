"""Tests for individual widgets — view rendering and update logic."""

from pathlib import Path

from pydantic_ai import models
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from calipso.widgets.agents_md import AgentsMd
from calipso.widgets.conversation_log import ConversationLog
from calipso.widgets.goal import Goal
from calipso.widgets.system_prompt import SystemPrompt
from calipso.widgets.task_list import TaskList, TaskStatus

models.ALLOW_MODEL_REQUESTS = False


# --- Widget base ---


class TestWidgetBase:
    def test_frontend_tools_empty_by_default(self):
        from calipso.widget import Widget

        w = Widget()
        assert w.frontend_tools() == set()


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


# --- AgentsMd ---


class TestAgentsMd:
    def test_loads_file(self, tmp_path: Path):
        md = tmp_path / "AGENTS.md"
        md.write_text("Be concise.")
        w = AgentsMd(path=md)
        msgs = list(w.view_messages())
        assert len(msgs) == 1
        assert isinstance(msgs[0], ModelRequest)
        assert msgs[0].parts[0].content == "Be concise."

    def test_missing_file_yields_nothing(self, tmp_path: Path):
        w = AgentsMd(path=tmp_path / "nonexistent.md")
        msgs = list(w.view_messages())
        assert msgs == []

    def test_empty_file_yields_nothing(self, tmp_path: Path):
        md = tmp_path / "AGENTS.md"
        md.write_text("   \n  ")
        w = AgentsMd(path=md)
        msgs = list(w.view_messages())
        assert msgs == []

    def test_view_tools_is_empty(self, tmp_path: Path):
        w = AgentsMd(path=tmp_path / "AGENTS.md")
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

    def test_frontend_tools(self):
        w = TaskList()
        assert w.frontend_tools() == {"update_task_status", "remove_task"}


# --- ConversationLog ---


class TestConversationLog:
    def test_view_messages_empty(self):
        w = ConversationLog()
        msgs = list(w.view_messages())
        # Should have the rules as a system prompt
        assert len(msgs) == 1
        assert "Action Protocol" in msgs[0].parts[0].content

    def test_add_user_message_creates_turn(self):
        w = ConversationLog()
        w.add_user_message("Hello")
        assert len(w.turns) == 1
        assert w.turns[0].user_message == "Hello"
        assert len(w.turns[0].segments) == 1

    def test_view_messages_renders_user_message(self):
        w = ConversationLog()
        w.add_user_message("Hello")
        msgs = list(w.view_messages())
        user_msgs = [
            m
            for m in msgs
            if isinstance(m, ModelRequest)
            and any(isinstance(p, UserPromptPart) for p in m.parts)
        ]
        assert len(user_msgs) == 1
        assert user_msgs[0].parts[0].content == "Hello"

    def test_unsummarized_segment_renders_full_messages(self):
        w = ConversationLog()
        w.add_user_message("Hello")
        response = ModelResponse(parts=[TextPart(content="Hi back!")])
        w.add_response(response, w._current_segment())
        msgs = list(w.view_messages())
        assert response in msgs

    def test_action_start_records_active_action(self):
        w = ConversationLog()
        w.add_user_message("Do something")
        assert len(w.turns[0].segments) == 1
        w.update("action_log_start", {"description": "Do the thing"})
        # action_log_start does NOT create a new segment
        assert len(w.turns[0].segments) == 1
        assert w._active_action == "Do the thing"

    def test_action_end_summarizes_and_creates_new_segment(self):
        w = ConversationLog()
        w.add_user_message("Do something")
        w.update("action_log_start", {"description": "Do the thing"})
        # Simulate some messages in the action segment
        seg = w._current_segment()
        w.add_response(ModelResponse(parts=[TextPart(content="working...")]), seg)
        w.update("action_log_end", {"result": "Thing was done successfully"})
        # The current segment (Seg0) should be summarized
        assert "Thing was done successfully" in w.turns[0].segments[0].summary
        # A new segment should exist for subsequent messages
        assert len(w.turns[0].segments) == 2
        assert w._active_action is None

    def test_summarized_segment_renders_summary_and_tool_calls(self):
        w = ConversationLog()
        w.add_user_message("Do something")
        w.update("action_log_start", {"description": "Do the thing"})
        seg = w._current_segment()
        # Response with both text and a tool call
        tool_call = ToolCallPart(
            tool_name="read_file",
            args={"path": "/tmp/x"},
            tool_call_id="tc1",
        )
        inner_response = ModelResponse(
            parts=[TextPart(content="working..."), tool_call]
        )
        w.add_response(inner_response, seg)
        # Tool return
        tool_return = ToolReturnPart(
            tool_name="read_file", content="file contents", tool_call_id="tc1"
        )
        w.add_tool_results(ModelRequest(parts=[tool_return]), seg)
        w.update("action_log_end", {"result": "Done!"})

        msgs = list(w.view_messages())
        # The summary should appear as a system prompt
        summaries = [
            m
            for m in msgs
            if isinstance(m, ModelRequest)
            and any(
                isinstance(p, SystemPromptPart) and "Done!" in p.content
                for p in m.parts
            )
        ]
        assert len(summaries) == 1
        # The original response (with TextPart) should NOT appear as-is
        assert inner_response not in msgs
        # But a ModelResponse with the ToolCallPart SHOULD appear
        tool_call_msgs = [
            m
            for m in msgs
            if isinstance(m, ModelResponse)
            and any(isinstance(p, ToolCallPart) for p in m.parts)
        ]
        assert len(tool_call_msgs) == 1
        assert tool_call_msgs[0].parts == [tool_call]
        # And the ToolReturnPart SHOULD appear
        tool_return_msgs = [
            m
            for m in msgs
            if isinstance(m, ModelRequest)
            and any(isinstance(p, ToolReturnPart) for p in m.parts)
        ]
        assert len(tool_return_msgs) == 1

    def test_summarized_segment_drops_text_parts(self):
        """Text-only responses in summarized segments are dropped entirely."""
        w = ConversationLog()
        w.add_user_message("Do something")
        w.update("action_log_start", {"description": "Do the thing"})
        seg = w._current_segment()
        text_response = ModelResponse(parts=[TextPart(content="thinking...")])
        w.add_response(text_response, seg)
        w.update("action_log_end", {"result": "Done!"})

        msgs = list(w.view_messages())
        # No ModelResponse should appear (text-only response is dropped)
        assert not any(isinstance(m, ModelResponse) for m in msgs)

    def test_view_tools(self):
        w = ConversationLog()
        tools = list(w.view_tools())
        names = {t.name for t in tools}
        assert names == {"action_log_start", "action_log_end"}


class TestConversationLogProtocol:
    def test_check_protocol_allows_start(self):
        w = ConversationLog()
        assert w.check_protocol("action_log_start") is None

    def test_check_protocol_no_active_action(self):
        w = ConversationLog()
        error = w.check_protocol("some_tool")
        assert error is not None
        assert "action_log_start" in error

    def test_check_protocol_end_without_start(self):
        w = ConversationLog()
        error = w.check_protocol("action_log_end")
        assert error is not None

    def test_check_protocol_tool_type_lock(self):
        w = ConversationLog()
        w.add_user_message("test")
        w.update("action_log_start", {"description": "Do stuff"})
        w.track_tool("tool_a")
        assert w.check_protocol("tool_a") is None
        error = w.check_protocol("tool_b")
        assert error is not None
        assert "tool_a" in error

    def test_protocol_rejects_empty_action(self):
        w = ConversationLog()
        w.add_user_message("test")
        w.update("action_log_start", {"description": "Read file"})
        assert w._active_action == "Read file"
        error = w.check_protocol("action_log_end")
        assert error is not None
        assert "without doing anything" in error

    def test_protocol_start_tool_then_end(self):
        w = ConversationLog()
        w.add_user_message("test")
        w.update("action_log_start", {"description": "Read file"})
        w.track_tool("read_file")
        assert w.check_protocol("action_log_end") is None
        result = w.update("action_log_end", {"result": "File read OK"})
        assert "logged" in result.lower()
        assert w._active_action is None
