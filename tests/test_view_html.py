"""Tests for widget HTML rendering and Context change detection."""

import pytest
from pydantic_ai import models

from calipso.widgets.agents_md import create_agents_md
from calipso.widgets.context import Context
from calipso.widgets.conversation_log import (
    UserMessageReceived,
    create_conversation_log,
)
from calipso.widgets.goal import GoalModel, create_goal
from calipso.widgets.goal import view_html as goal_view_html
from calipso.widgets.system_prompt import SystemPromptModel, create_system_prompt
from calipso.widgets.system_prompt import view_html as sp_view_html
from calipso.widgets.task_list import create_task_list

models.ALLOW_MODEL_REQUESTS = False
pytestmark = pytest.mark.anyio


# --- widget_id ---


class TestWidgetId:
    def test_system_prompt_id(self):
        assert create_system_prompt().widget_id() == "widget-system-prompt"

    def test_goal_id(self):
        assert create_goal().widget_id() == "widget-goal"

    def test_task_list_id(self):
        assert create_task_list().widget_id() == "widget-task-list"

    def test_conversation_log_id(self):
        assert create_conversation_log().widget_id() == "widget-conversation-log"

    def test_agents_md_id(self, tmp_path):
        assert create_agents_md(directory=tmp_path).widget_id() == "widget-agents-md"


# --- view_html contains correct id ---


class TestViewHtmlIds:
    def test_system_prompt_has_id(self):
        html = create_system_prompt(text="Hello").view_html()
        assert 'id="widget-system-prompt"' in html

    def test_goal_has_id(self):
        html = create_goal().view_html()
        assert 'id="widget-goal"' in html

    def test_task_list_has_id(self):
        html = create_task_list().view_html()
        assert 'id="widget-task-list"' in html

    def test_conversation_log_has_id(self):
        html = create_conversation_log().view_html()
        assert 'id="widget-conversation-log"' in html


# --- view_html content ---


class TestSystemPromptHtml:
    def test_renders_text(self):
        html = sp_view_html(SystemPromptModel(text="Be helpful."))
        assert "Be helpful." in html

    def test_escapes_html(self):
        html = sp_view_html(SystemPromptModel(text="<script>alert(1)</script>"))
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_renders_markdown(self):
        html = sp_view_html(SystemPromptModel(text="**bold**"))
        assert "<strong>bold</strong>" in html


class TestGoalHtml:
    def test_no_goal(self):
        html = goal_view_html(GoalModel())
        assert "No goal set" in html

    def test_with_goal(self):
        html = goal_view_html(GoalModel(text="Build a dashboard"))
        assert "Build a dashboard" in html

    def test_escapes_goal(self):
        html = goal_view_html(GoalModel(text="<b>xss</b>"))
        assert "<b>" not in html


class TestTaskListHtml:
    def test_empty(self):
        html = create_task_list().view_html()
        assert "No tasks" in html

    async def test_with_tasks(self):
        tl = create_task_list()
        await tl.dispatch_llm("create_task", {"description": "Write tests"})
        html = tl.view_html()
        assert "Write tests" in html
        assert "<ul>" in html

    async def test_escapes_task_description(self):
        tl = create_task_list()
        await tl.dispatch_llm("create_task", {"description": "<img src=x>"})
        html = tl.view_html()
        assert "<img" not in html


class TestConversationLogHtml:
    def test_empty(self):
        html = create_conversation_log().view_html()
        assert "No messages yet" in html

    def test_with_user_message(self):
        cl = create_conversation_log()
        cl.send(UserMessageReceived(text="Hello"))
        html = cl.view_html()
        assert "Hello" in html
        assert "You:" in html


class TestAgentsMdHtml:
    def test_missing_file_shows_warning_and_reload(self, tmp_path):
        w = create_agents_md(directory=tmp_path)
        html = w.view_html()
        assert "Neither" in html
        assert "reload_agents_md" in html

    def test_with_content_shows_reload(self, tmp_path):
        (tmp_path / "AGENTS.md").write_text("Be careful.")
        w = create_agents_md(directory=tmp_path)
        html = w.view_html()
        assert "Be careful." in html
        assert "reload_agents_md" in html
        assert "AGENTS.md" in html


# --- Context change detection ---


class TestContextChangedHtml:
    def _make_context(self):
        return Context(
            system_prompt=create_system_prompt(),
            children=[create_goal(), create_task_list()],
            conversation_log=create_conversation_log(),
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
        # Mutate the goal widget via dispatch
        await ctx.children[0].dispatch_llm("set_goal", {"goal": "New goal"})
        changed = ctx.changed_html()
        assert len(changed) == 1
        assert "New goal" in changed[0]

    async def test_changed_html_only_returns_changed(self):
        ctx = self._make_context()
        ctx.all_html()
        # Mutate task list, leave goal alone
        await ctx.children[1].dispatch_llm("create_task", {"description": "Do stuff"})
        changed = ctx.changed_html()
        ids = [c for c in changed if "widget-task-list" in c]
        assert len(ids) == 1
        # Goal should not be in changed
        goal_ids = [c for c in changed if "widget-goal" in c]
        assert len(goal_ids) == 0
