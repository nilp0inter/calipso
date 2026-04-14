"""Tests for widget HTML rendering and Context change detection."""

import pytest
from pydantic_ai import models

from calipso.widgets.agents_md import AgentsMd
from calipso.widgets.context import Context
from calipso.widgets.conversation_log import ConversationLog
from calipso.widgets.goal import Goal
from calipso.widgets.system_prompt import SystemPrompt
from calipso.widgets.task_list import TaskList

models.ALLOW_MODEL_REQUESTS = False
pytestmark = pytest.mark.anyio


# --- widget_id ---


class TestWidgetId:
    def test_system_prompt_id(self):
        assert SystemPrompt().widget_id() == "widget-system-prompt"

    def test_goal_id(self):
        assert Goal().widget_id() == "widget-goal"

    def test_task_list_id(self):
        assert TaskList().widget_id() == "widget-task-list"

    def test_conversation_log_id(self):
        assert ConversationLog().widget_id() == "widget-conversation-log"

    def test_agents_md_id(self):
        assert AgentsMd().widget_id() == "widget-agents-md"


# --- view_html contains correct id ---


class TestViewHtmlIds:
    def test_system_prompt_has_id(self):
        html = SystemPrompt(text="Hello").view_html()
        assert 'id="widget-system-prompt"' in html

    def test_goal_has_id(self):
        html = Goal().view_html()
        assert 'id="widget-goal"' in html

    def test_task_list_has_id(self):
        html = TaskList().view_html()
        assert 'id="widget-task-list"' in html

    def test_conversation_log_has_id(self):
        html = ConversationLog().view_html()
        assert 'id="widget-conversation-log"' in html


# --- view_html content ---


class TestSystemPromptHtml:
    def test_renders_text(self):
        html = SystemPrompt(text="Be helpful.").view_html()
        assert "Be helpful." in html

    def test_escapes_html(self):
        html = SystemPrompt(text="<script>alert(1)</script>").view_html()
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_renders_markdown(self):
        html = SystemPrompt(text="**bold**").view_html()
        assert "<strong>bold</strong>" in html


class TestGoalHtml:
    def test_no_goal(self):
        html = Goal().view_html()
        assert "No goal set" in html

    def test_with_goal(self):
        g = Goal(text="Build a dashboard")
        html = g.view_html()
        assert "Build a dashboard" in html

    def test_escapes_goal(self):
        g = Goal(text="<b>xss</b>")
        html = g.view_html()
        assert "<b>" not in html


class TestTaskListHtml:
    def test_empty(self):
        html = TaskList().view_html()
        assert "No tasks" in html

    async def test_with_tasks(self):
        tl = TaskList()
        await tl.update("create_task", {"description": "Write tests"})
        html = tl.view_html()
        assert "Write tests" in html
        assert "<ul>" in html

    async def test_escapes_task_description(self):
        tl = TaskList()
        await tl.update("create_task", {"description": "<img src=x>"})
        html = tl.view_html()
        assert "<img" not in html


class TestConversationLogHtml:
    def test_empty(self):
        html = ConversationLog().view_html()
        assert "No messages yet" in html

    def test_with_user_message(self):
        cl = ConversationLog()
        cl.add_user_message("Hello")
        html = cl.view_html()
        assert "Hello" in html
        assert "You:" in html


class TestAgentsMdHtml:
    def test_missing_file(self, tmp_path):
        w = AgentsMd(path=tmp_path / "missing.md")
        html = w.view_html()
        assert "Not found" in html

    def test_with_content(self, tmp_path):
        f = tmp_path / "AGENTS.md"
        f.write_text("Be careful.")
        w = AgentsMd(path=f)
        html = w.view_html()
        assert "Be careful." in html


# --- Context change detection ---


class TestContextChangedHtml:
    def _make_context(self):
        return Context(
            system_prompt=SystemPrompt(),
            children=[Goal(), TaskList()],
            conversation_log=ConversationLog(),
        )

    def test_all_html_returns_all_widgets(self):
        ctx = self._make_context()
        all_h = ctx.all_html()
        # system_prompt + goal + task_list + conversation_log = 4
        assert len(all_h) == 4

    def test_changed_html_empty_after_all_html(self):
        ctx = self._make_context()
        ctx.all_html()
        changed = ctx.changed_html()
        assert changed == []

    async def test_changed_html_detects_goal_change(self):
        ctx = self._make_context()
        ctx.all_html()
        # Mutate the goal widget
        await ctx.children[0].update("set_goal", {"goal": "New goal"})
        changed = ctx.changed_html()
        assert len(changed) == 1
        assert "New goal" in changed[0]

    async def test_changed_html_only_returns_changed(self):
        ctx = self._make_context()
        ctx.all_html()
        # Mutate task list, leave goal alone
        await ctx.children[1].update("create_task", {"description": "Do stuff"})
        changed = ctx.changed_html()
        ids = [c for c in changed if "widget-task-list" in c]
        assert len(ids) == 1
        # Goal should not be in changed
        goal_ids = [c for c in changed if "widget-goal" in c]
        assert len(goal_ids) == 0
