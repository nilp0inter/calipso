"""Tests for individual widgets — view rendering and update logic."""

from pathlib import Path

import pytest
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
from calipso.widgets.file_explorer import FileExplorer
from calipso.widgets.goal import Goal
from calipso.widgets.system_prompt import SystemPrompt
from calipso.widgets.task_list import TaskList, TaskStatus

models.ALLOW_MODEL_REQUESTS = False
pytestmark = pytest.mark.anyio


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
    def test_loads_agents_md(self, tmp_path: Path):
        (tmp_path / "AGENTS.md").write_text("Be concise.")
        w = AgentsMd(directory=tmp_path)
        msgs = list(w.view_messages())
        assert len(msgs) == 1
        assert isinstance(msgs[0], ModelRequest)
        assert msgs[0].parts[0].content == "Be concise."

    def test_falls_back_to_claude_md(self, tmp_path: Path):
        (tmp_path / "CLAUDE.md").write_text("Be helpful.")
        w = AgentsMd(directory=tmp_path)
        msgs = list(w.view_messages())
        assert len(msgs) == 1
        assert msgs[0].parts[0].content == "Be helpful."

    def test_agents_md_takes_priority(self, tmp_path: Path):
        (tmp_path / "AGENTS.md").write_text("From AGENTS")
        (tmp_path / "CLAUDE.md").write_text("From CLAUDE")
        w = AgentsMd(directory=tmp_path)
        assert w.content == "From AGENTS"

    def test_missing_files_yields_nothing(self, tmp_path: Path):
        w = AgentsMd(directory=tmp_path)
        msgs = list(w.view_messages())
        assert msgs == []
        assert w.error is not None

    def test_empty_file_yields_nothing(self, tmp_path: Path):
        (tmp_path / "AGENTS.md").write_text("   \n  ")
        w = AgentsMd(directory=tmp_path)
        msgs = list(w.view_messages())
        assert msgs == []

    def test_view_tools_has_reload(self, tmp_path: Path):
        w = AgentsMd(directory=tmp_path)
        names = {t.name for t in w.view_tools()}
        assert names == {"reload_agents_md"}

    async def test_reload(self, tmp_path: Path):
        w = AgentsMd(directory=tmp_path)
        assert w.error is not None
        (tmp_path / "AGENTS.md").write_text("New content")
        result = await w.update("reload_agents_md", {})
        assert "Loaded" in result
        assert w.content == "New content"
        assert w.error is None

    def test_frontend_tools(self, tmp_path: Path):
        w = AgentsMd(directory=tmp_path)
        assert w.frontend_tools() == {"reload_agents_md"}


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

    async def test_set_goal(self):
        w = Goal()
        result = await w.update("set_goal", {"goal": "Deploy v2"})
        assert "Deploy v2" in result
        assert w.text == "Deploy v2"
        msgs = list(w.view_messages())
        assert "Deploy v2" in msgs[0].parts[0].content

    async def test_clear_goal(self):
        w = Goal(text="Something")
        result = await w.update("clear_goal", {})
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

    async def test_create_and_view(self):
        w = TaskList()
        await w.update("create_task", {"description": "Write tests"})
        await w.update("create_task", {"description": "Deploy"})
        msgs = list(w.view_messages())
        content = msgs[0].parts[0].content
        assert "Write tests" in content
        assert "Deploy" in content
        assert "[ ]" in content  # pending icon

    async def test_update_status(self):
        w = TaskList()
        await w.update("create_task", {"description": "Task A"})
        await w.update("update_task_status", {"task_id": 1, "status": "done"})
        assert w.tasks[0].status == TaskStatus.DONE
        msgs = list(w.view_messages())
        assert "[x]" in msgs[0].parts[0].content

    async def test_remove_task(self):
        w = TaskList()
        await w.update("create_task", {"description": "Temp"})
        result = await w.update("remove_task", {"task_id": 1})
        assert "Removed" in result
        assert len(w.tasks) == 0

    async def test_invalid_status(self):
        w = TaskList()
        await w.update("create_task", {"description": "X"})
        result = await w.update("update_task_status", {"task_id": 1, "status": "bad"})
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

    async def test_action_start_records_active_action(self):
        w = ConversationLog()
        w.add_user_message("Do something")
        assert len(w.turns[0].segments) == 1
        await w.update("action_log_start", {"description": "Do the thing"})
        # action_log_start does NOT create a new segment
        assert len(w.turns[0].segments) == 1
        assert w._active_action == "Do the thing"

    async def test_action_end_summarizes_and_creates_new_segment(self):
        w = ConversationLog()
        w.add_user_message("Do something")
        await w.update("action_log_start", {"description": "Do the thing"})
        # Simulate some messages in the action segment
        seg = w._current_segment()
        w.add_response(ModelResponse(parts=[TextPart(content="working...")]), seg)
        await w.update("action_log_end", {"result": "Thing was done successfully"})
        # The current segment (Seg0) should be summarized
        assert "Thing was done successfully" in w.turns[0].segments[0].summary
        # A new segment should exist for subsequent messages
        assert len(w.turns[0].segments) == 2
        assert w._active_action is None

    async def test_summarized_segment_renders_summary_and_tool_calls(self):
        w = ConversationLog()
        w.add_user_message("Do something")
        await w.update("action_log_start", {"description": "Do the thing"})
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
        await w.update("action_log_end", {"result": "Done!"})

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

    async def test_summarized_segment_drops_text_parts(self):
        """Text-only responses in summarized segments are dropped entirely."""
        w = ConversationLog()
        w.add_user_message("Do something")
        await w.update("action_log_start", {"description": "Do the thing"})
        seg = w._current_segment()
        text_response = ModelResponse(parts=[TextPart(content="thinking...")])
        w.add_response(text_response, seg)
        await w.update("action_log_end", {"result": "Done!"})

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

    async def test_check_protocol_tool_type_lock(self):
        w = ConversationLog()
        w.add_user_message("test")
        await w.update("action_log_start", {"description": "Do stuff"})
        w.track_tool("tool_a")
        assert w.check_protocol("tool_a") is None
        error = w.check_protocol("tool_b")
        assert error is not None
        assert "tool_a" in error

    async def test_protocol_rejects_empty_action(self):
        w = ConversationLog()
        w.add_user_message("test")
        await w.update("action_log_start", {"description": "Read file"})
        assert w._active_action == "Read file"
        error = w.check_protocol("action_log_end")
        assert error is not None
        assert "without doing anything" in error

    async def test_protocol_start_tool_then_end(self):
        w = ConversationLog()
        w.add_user_message("test")
        await w.update("action_log_start", {"description": "Read file"})
        w.track_tool("read_file")
        assert w.check_protocol("action_log_end") is None
        result = await w.update("action_log_end", {"result": "File read OK"})
        assert "logged" in result.lower()
        assert w._active_action is None


# --- FileExplorer ---


class TestFileExplorer:
    async def test_list_directory(self, tmp_path: Path):
        w = FileExplorer()
        sub = tmp_path / "subdir"
        sub.mkdir()
        (tmp_path / "file.txt").write_text("hello")
        result = await w.update("list_directory", {"path": str(tmp_path)})
        assert "subdir/" in result
        assert "file.txt" in result
        assert w.current_listing is not None

    async def test_list_directory_not_a_dir(self, tmp_path: Path):
        w = FileExplorer()
        result = await w.update("list_directory", {"path": str(tmp_path / "nope")})
        assert "Not a directory" in result

    async def test_read_file(self, tmp_path: Path):
        w = FileExplorer()
        f = tmp_path / "readme.md"
        f.write_text("# Hello")
        result = await w.update("read_file", {"path": str(f)})
        assert result == "# Hello"
        assert w.open_file_path == str(f)
        assert w.open_file_content == "# Hello"

    async def test_read_file_rejects_python(self):
        w = FileExplorer()
        result = await w.update("read_file", {"path": "foo.py"})
        assert "Code Explorer" in result
        assert w.open_file_path is None

    async def test_read_file_not_found(self):
        w = FileExplorer()
        result = await w.update("read_file", {"path": "/no/such/file.txt"})
        assert "File not found" in result

    async def test_close_read_file(self, tmp_path: Path):
        w = FileExplorer()
        f = tmp_path / "data.json"
        f.write_text("{}")
        await w.update("read_file", {"path": str(f)})
        assert w.open_file_path is not None
        result = await w.update("close_read_file", {})
        assert "Closed" in result
        assert w.open_file_path is None
        assert w.open_file_content is None

    async def test_close_read_file_when_none_open(self):
        w = FileExplorer()
        result = await w.update("close_read_file", {})
        assert "No file is open" in result

    def test_view_messages_empty(self):
        w = FileExplorer()
        msgs = list(w.view_messages())
        assert len(msgs) == 1
        assert "No file open" in msgs[0].parts[0].content

    def test_view_messages_with_file(self):
        w = FileExplorer()
        w.open_file_path = "test.txt"
        w.open_file_content = "content here"
        msgs = list(w.view_messages())
        text = msgs[0].parts[0].content
        assert "`test.txt`" in text
        assert "content here" in text

    def test_view_html_empty(self):
        w = FileExplorer()
        html = w.view_html()
        assert "No file open" in html
        assert w.widget_id() in html

    def test_view_html_with_file(self):
        w = FileExplorer()
        w.open_file_path = "test.txt"
        w.open_file_content = "hello"
        html = w.view_html()
        assert "test.txt" in html
        assert "hello" in html
        assert "close_read_file" in html

    def test_view_tools(self):
        w = FileExplorer()
        tools = list(w.view_tools())
        names = {t.name for t in tools}
        assert names == {"list_directory", "read_file", "close_read_file"}

    def test_frontend_tools(self):
        w = FileExplorer()
        assert w.frontend_tools() == {"close_read_file"}
