"""Tests for individual widgets — view rendering and update logic."""

from pathlib import Path

import pytest
from pydantic_ai import models
from pydantic_ai.messages import ModelRequest, SystemPromptPart

from calipso.widgets.agents_md import create_agents_md
from calipso.widgets.file_explorer import create_file_explorer
from calipso.widgets.goal import create_goal
from calipso.widgets.system_prompt import create_system_prompt
from calipso.widgets.test_suite import create_test_suite

models.ALLOW_MODEL_REQUESTS = False
pytestmark = pytest.mark.anyio


# --- Widget base ---


class TestWidgetBase:
    def test_frontend_tools_empty_by_default(self):
        from calipso.widget import create_widget

        w = create_widget(id="test", model=None)
        assert w.frontend_tools() == frozenset()


# --- SystemPrompt ---


class TestSystemPrompt:
    def test_view_messages_yields_system_prompt(self):
        w = create_system_prompt(text="Be helpful.")
        msgs = list(w.view_messages())
        assert len(msgs) == 1
        assert isinstance(msgs[0], ModelRequest)
        assert isinstance(msgs[0].parts[0], SystemPromptPart)
        assert msgs[0].parts[0].content == "Be helpful."

    def test_view_tools_is_empty(self):
        w = create_system_prompt()
        assert list(w.view_tools()) == []


# --- AgentsMd ---


class TestAgentsMd:
    def test_loads_agents_md(self, tmp_path: Path):
        (tmp_path / "AGENTS.md").write_text("Be concise.")
        w = create_agents_md(directory=tmp_path)
        msgs = list(w.view_messages())
        assert len(msgs) == 1
        assert isinstance(msgs[0], ModelRequest)
        assert msgs[0].parts[0].content == "Be concise."

    def test_falls_back_to_claude_md(self, tmp_path: Path):
        (tmp_path / "CLAUDE.md").write_text("Be helpful.")
        w = create_agents_md(directory=tmp_path)
        msgs = list(w.view_messages())
        assert len(msgs) == 1
        assert msgs[0].parts[0].content == "Be helpful."

    def test_agents_md_takes_priority(self, tmp_path: Path):
        (tmp_path / "AGENTS.md").write_text("From AGENTS")
        (tmp_path / "CLAUDE.md").write_text("From CLAUDE")
        w = create_agents_md(directory=tmp_path)
        assert w.model.content == "From AGENTS"

    def test_missing_files_yields_nothing(self, tmp_path: Path):
        w = create_agents_md(directory=tmp_path)
        msgs = list(w.view_messages())
        assert msgs == []
        assert w.model.error is not None

    def test_empty_file_yields_nothing(self, tmp_path: Path):
        (tmp_path / "AGENTS.md").write_text("   \n  ")
        w = create_agents_md(directory=tmp_path)
        msgs = list(w.view_messages())
        assert msgs == []

    def test_view_tools_has_reload(self, tmp_path: Path):
        w = create_agents_md(directory=tmp_path)
        names = {t.name for t in w.view_tools()}
        assert names == {"reload_agents_md"}

    async def test_reload(self, tmp_path: Path):
        w = create_agents_md(directory=tmp_path)
        assert w.model.error is not None
        (tmp_path / "AGENTS.md").write_text("New content")
        result = await w.dispatch_llm("reload_agents_md", {})
        assert "Loaded" in result
        assert w.model.content == "New content"
        assert w.model.error is None

    def test_frontend_tools(self, tmp_path: Path):
        w = create_agents_md(directory=tmp_path)
        assert w.frontend_tools() == frozenset({"reload_agents_md"})


# --- Goal ---


class TestGoal:
    def test_view_no_goal(self):
        w = create_goal()
        msgs = list(w.view_messages())
        assert len(msgs) == 1
        assert "No goal set" in msgs[0].parts[0].content

    def test_view_with_goal(self):
        w = create_goal(text="Fix the bug")
        msgs = list(w.view_messages())
        assert "Fix the bug" in msgs[0].parts[0].content

    async def test_set_goal(self):
        w = create_goal()
        result = await w.dispatch_llm("set_goal", {"goal": "Deploy v2"})
        assert "Deploy v2" in result
        assert w.model.text == "Deploy v2"
        msgs = list(w.view_messages())
        assert "Deploy v2" in msgs[0].parts[0].content

    async def test_clear_goal(self):
        w = create_goal(text="Something")
        result = await w.dispatch_llm("clear_goal", {})
        assert "cleared" in result.lower()
        assert w.model.text is None

    def test_view_tools(self):
        w = create_goal()
        tools = list(w.view_tools())
        names = {t.name for t in tools}
        assert names == {"set_goal", "clear_goal"}


# --- FileExplorer ---


class TestFileExplorer:
    async def test_open_directory(self, tmp_path: Path):
        w = create_file_explorer()
        sub = tmp_path / "subdir"
        sub.mkdir()
        (tmp_path / "file.txt").write_text("hello")
        result = await w.dispatch_llm("open_directory", {"path": str(tmp_path)})
        assert result == f"Opened directory: {tmp_path}"
        assert len(w.model.open_directories) == 1
        d = w.model.open_directories[0]
        assert d.path == str(tmp_path)
        assert "subdir/" in d.listing_text
        assert "file.txt" in d.listing_text

    async def test_open_directory_not_a_dir(self, tmp_path: Path):
        w = create_file_explorer()
        result = await w.dispatch_llm(
            "open_directory", {"path": str(tmp_path / "nope")}
        )
        assert "Not a directory" in result
        assert w.model.open_directories == ()

    async def test_close_directory(self, tmp_path: Path):
        w = create_file_explorer()
        await w.dispatch_llm("open_directory", {"path": str(tmp_path)})
        assert len(w.model.open_directories) == 1
        result = await w.dispatch_llm("close_directory", {"path": str(tmp_path)})
        assert "Closed directory" in result
        assert w.model.open_directories == ()

    async def test_close_directory_when_none_open(self):
        w = create_file_explorer()
        result = await w.dispatch_llm("close_directory", {"path": "nope"})
        assert "No directory is open" in result

    async def test_open_directory_refreshes_existing(self, tmp_path: Path):
        w = create_file_explorer()
        await w.dispatch_llm("open_directory", {"path": str(tmp_path)})
        await w.dispatch_llm("open_directory", {"path": str(tmp_path)})
        assert len(w.model.open_directories) == 1

    async def test_multiple_open_directories(self, tmp_path: Path):
        w = create_file_explorer()
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        await w.dispatch_llm("open_directory", {"path": str(a)})
        await w.dispatch_llm("open_directory", {"path": str(b)})
        assert len(w.model.open_directories) == 2
        msgs = list(w.view_messages())
        text = msgs[0].parts[0].content
        assert str(a) in text
        assert str(b) in text

    async def test_read_file(self, tmp_path: Path):
        w = create_file_explorer()
        f = tmp_path / "readme.md"
        f.write_text("# Hello")
        result = await w.dispatch_llm("read_file", {"path": str(f)})
        assert result == f"Opened: {f}"
        assert w.model.open_files == ((str(f), "# Hello"),)

    async def test_read_file_not_found(self):
        w = create_file_explorer()
        result = await w.dispatch_llm("read_file", {"path": "/no/such/file.txt"})
        assert "File not found" in result

    async def test_close_read_file(self, tmp_path: Path):
        w = create_file_explorer()
        f = tmp_path / "data.json"
        f.write_text("{}")
        await w.dispatch_llm("read_file", {"path": str(f)})
        assert len(w.model.open_files) == 1
        result = await w.dispatch_llm("close_read_file", {"path": str(f)})
        assert "Closed" in result
        assert w.model.open_files == ()

    async def test_close_read_file_when_none_open(self):
        w = create_file_explorer()
        result = await w.dispatch_llm("close_read_file", {"path": "nope.txt"})
        assert "No file is open" in result

    def test_view_messages_empty(self):
        w = create_file_explorer()
        msgs = list(w.view_messages())
        assert len(msgs) == 1
        assert "No directory or file open" in msgs[0].parts[0].content

    def test_view_messages_with_file(self):
        from calipso.cmd import CmdToolResult
        from calipso.widgets.file_explorer import FileExplorerModel, FileRead, update

        model = FileExplorerModel()
        model, cmd = update(model, FileRead(path="test.txt", content="content here"))
        assert isinstance(cmd, CmdToolResult)
        # Verify via view function directly
        from calipso.widgets.file_explorer import view_messages

        msgs = list(view_messages(model))
        text = msgs[0].parts[0].content
        assert "`test.txt`" in text
        assert "content here" in text

    def test_view_html_empty(self):
        w = create_file_explorer()
        html = w.view_html()
        assert "Root" in html
        assert w.widget_id() in html

    def test_view_html_with_file(self):
        from calipso.widgets.file_explorer import (
            FileExplorerModel,
            FileRead,
            update,
            view_html,
        )

        model = FileExplorerModel()
        model, _cmd = update(model, FileRead(path="test.txt", content="hello"))
        html = view_html(model)
        assert "test.txt" in html
        assert "hello" in html
        assert "close_read_file" in html

    def test_view_tools(self):
        w = create_file_explorer()
        tools = list(w.view_tools())
        names = {t.name for t in tools}
        assert names == {
            "open_directory",
            "close_directory",
            "read_file",
            "close_read_file",
        }

    def test_frontend_tools(self):
        w = create_file_explorer()
        assert w.frontend_tools() == frozenset(
            {"open_directory", "close_directory", "read_file", "close_read_file"}
        )


# --- TestSuite ---


class TestTestSuite:
    async def test_configure(self):
        w = create_test_suite()
        result = await w.dispatch_llm(
            "configure_test_runner",
            {"command": "pytest -v", "env_vars": {"CI": "1"}, "timeout": 60},
        )
        assert "configured" in result.lower()
        assert w.model.command == "pytest -v"
        assert w.model.env_vars == (("CI", "1"),)
        assert w.model.timeout == 60

    async def test_run_without_config(self):
        w = create_test_suite()
        result = await w.dispatch_llm("run_tests", {})
        assert "No test command" in result

    async def test_run_tests_pass(self):
        w = create_test_suite()
        await w.dispatch_llm(
            "configure_test_runner",
            {"command": "echo ok", "env_vars": {}, "timeout": 10},
        )
        result = await w.dispatch_llm("run_tests", {})
        assert "passed" in result.lower()
        assert w.model.status == "passed"
        assert "ok" in w.model.stdout

    async def test_run_tests_fail(self):
        w = create_test_suite()
        await w.dispatch_llm(
            "configure_test_runner",
            {"command": "exit 1", "env_vars": {}, "timeout": 10},
        )
        result = await w.dispatch_llm("run_tests", {})
        assert "failed" in result.lower()
        assert w.model.status == "failed"

    async def test_run_tests_timeout(self):
        w = create_test_suite()
        await w.dispatch_llm(
            "configure_test_runner",
            {"command": "sleep 60", "env_vars": {}, "timeout": 1},
        )
        result = await w.dispatch_llm("run_tests", {})
        assert "timed out" in result.lower()
        assert w.model.status == "timeout"

    async def test_reject_concurrent_run(self):
        import asyncio

        w = create_test_suite()
        await w.dispatch_llm(
            "configure_test_runner",
            {"command": "sleep 60", "env_vars": {}, "timeout": 120},
        )
        run_task = asyncio.create_task(w.dispatch_llm("run_tests", {}))
        await asyncio.sleep(0.2)
        result = await w.dispatch_llm("run_tests", {})
        assert "already running" in result.lower()
        # Clean up: cancel the long-running task
        await w.dispatch_ui("cancel_tests", {})
        await run_task

    async def test_cancel_running(self):
        import asyncio

        w = create_test_suite()
        await w.dispatch_llm(
            "configure_test_runner",
            {"command": "sleep 60", "env_vars": {}, "timeout": 120},
        )
        run_task = asyncio.create_task(w.dispatch_llm("run_tests", {}))
        await asyncio.sleep(0.2)
        await w.dispatch_ui("cancel_tests", {})
        await run_task
        assert w.model.status == "cancelled"

    async def test_cancel_when_not_running(self):
        w = create_test_suite()
        result = await w.dispatch_llm("cancel_tests", {})
        assert "No tests" in result or "not" in result.lower()

    def test_view_tools(self):
        w = create_test_suite()
        tools = list(w.view_tools())
        names = {t.name for t in tools}
        assert names == {"configure_test_runner", "run_tests", "cancel_tests"}

    def test_frontend_tools(self):
        w = create_test_suite()
        assert w.frontend_tools() == frozenset(
            {"configure_test_runner", "run_tests", "cancel_tests"}
        )

    def test_view_html_contains_widget_id(self):
        w = create_test_suite()
        html = w.view_html()
        assert "widget-test-suite" in html

    def test_view_messages_unconfigured(self):
        w = create_test_suite()
        msgs = list(w.view_messages())
        assert "No test runner configured" in msgs[0].parts[0].content

    def test_stale_defaults_false(self):
        w = create_test_suite()
        assert w.model.stale is False
