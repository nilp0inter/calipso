"""Tests for the Context widget — composition and tool dispatch."""

import pytest
from pydantic_ai import models
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)

from calipso.widgets import (
    Context,
    create_conversation_log,
    create_goal,
    create_system_prompt,
    create_task_list,
)

models.ALLOW_MODEL_REQUESTS = False
pytestmark = pytest.mark.anyio


class TestContextComposition:
    def test_view_messages_composes_children(self):
        ctx = Context(
            system_prompt=create_system_prompt(text="Hello"),
            children=[create_goal(text="Win")],
            conversation_log=create_conversation_log(),
        )
        msgs = list(ctx.view_messages())
        sys_contents = [
            p.content
            for m in msgs
            if isinstance(m, ModelRequest)
            for p in m.parts
            if isinstance(p, SystemPromptPart)
        ]
        user_contents = [
            p.content
            for m in msgs
            if isinstance(m, ModelRequest)
            for p in m.parts
            if isinstance(p, UserPromptPart)
        ]
        assert any("Hello" in c for c in sys_contents)
        assert any("Win" in c for c in user_contents)

    def test_view_tools_composes_children(self):
        ctx = Context(
            system_prompt=create_system_prompt(),
            children=[create_goal(), create_task_list()],
            conversation_log=create_conversation_log(),
        )
        tools = list(ctx.view_tools())
        names = {t.name for t in tools}
        assert "set_goal" in names
        assert "create_task" in names

    def test_view_messages_includes_conversation(self):
        ctx = Context(
            system_prompt=create_system_prompt(),
            children=[],
            conversation_log=create_conversation_log(),
        )
        ctx.add_user_message("Hi there")
        msgs = list(ctx.view_messages())
        assert any(
            hasattr(p, "content") and "Hi there" in p.content
            for m in msgs
            for p in m.parts
        )

    def test_state_panels_appear_after_conversation(self):
        ctx = Context(
            system_prompt=create_system_prompt(text="Identity"),
            children=[create_goal(text="Win")],
            conversation_log=create_conversation_log(),
        )
        ctx.add_user_message("Hello")
        msgs = list(ctx.view_messages())
        # System prompts (identity, action rules)
        sys_contents = [
            p.content
            for m in msgs
            if isinstance(m, ModelRequest)
            for p in m.parts
            if isinstance(p, SystemPromptPart)
        ]
        assert "Identity" in sys_contents[0]
        # State panels use UserPromptPart
        user_contents = [
            p.content
            for m in msgs
            if isinstance(m, ModelRequest)
            for p in m.parts
            if isinstance(p, UserPromptPart)
        ]
        assert "CURRENT STATE" in user_contents[-3]
        assert "Win" in user_contents[-2]
        assert "END STATE" in user_contents[-1]


class TestContextDispatch:
    async def test_dispatch_tool_call_to_goal(self):
        goal = create_goal()
        ctx = Context(
            system_prompt=create_system_prompt(),
            children=[goal],
            conversation_log=create_conversation_log(),
        )
        ctx.add_user_message("set my goal")
        # Start an action first (protocol enforcement)
        start_response = ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="action_log_start",
                    args={"description": "Set goal"},
                    tool_call_id="call_0",
                )
            ]
        )
        await ctx.handle_response(start_response)

        response = ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="set_goal",
                    args={"goal": "Ship it"},
                    tool_call_id="call_1",
                )
            ]
        )
        results, _ = await ctx.handle_response(response)
        assert len(results) == 1
        assert "Ship it" in results[0][1]
        assert ctx.children[0].model.text == "Ship it"

    async def test_dispatch_unknown_tool(self):
        ctx = Context(
            system_prompt=create_system_prompt(),
            children=[],
            conversation_log=create_conversation_log(),
        )
        ctx.add_user_message("test")
        # Start an action first
        await ctx.handle_response(
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="action_log_start",
                        args={"description": "test"},
                        tool_call_id="call_0",
                    )
                ]
            )
        )
        response = ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="nonexistent",
                    args={},
                    tool_call_id="call_1",
                )
            ]
        )
        results, _ = await ctx.handle_response(response)
        assert "Unknown" in results[0][1]

    async def test_action_log_protocol_enforcement(self):
        task_list = create_task_list()
        ctx = Context(
            system_prompt=create_system_prompt(),
            children=[task_list],
            conversation_log=create_conversation_log(),
        )
        ctx.add_user_message("test")
        # Try calling create_task without action_log_start
        response = ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="create_task",
                    args={"description": "Foo"},
                    tool_call_id="call_1",
                )
            ]
        )
        results, _ = await ctx.handle_response(response)
        assert "action_log_start" in results[0][1]
        # Task should NOT have been created
        assert len(task_list.model.tasks) == 0

    async def test_widget_event_dispatches_to_frontend_tool(self):
        task_list = create_task_list()
        ctx = Context(
            system_prompt=create_system_prompt(),
            children=[task_list],
            conversation_log=create_conversation_log(),
        )
        # Create a task via LLM path first (need action wrapper)
        ctx.add_user_message("test")
        await ctx.handle_response(
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="action_log_start",
                        args={"description": "create task"},
                        tool_call_id="c0",
                    )
                ]
            )
        )
        await ctx.handle_response(
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="create_task",
                        args={"description": "Test task"},
                        tool_call_id="c1",
                    )
                ]
            )
        )
        # Now use frontend event to update it
        result = ctx.handle_widget_event(
            "update_task_status", {"task_id": 1, "status": "done"}
        )
        assert result is not None
        assert "done" in result
        assert task_list.model.tasks[0].status.value == "done"

    async def test_widget_event_rejects_non_frontend_tool(self):
        ctx = Context(
            system_prompt=create_system_prompt(),
            children=[],
            conversation_log=create_conversation_log(),
        )
        # action_log_start is NOT in frontend_tools
        result = ctx.handle_widget_event("action_log_start", {"description": "Hacked"})
        assert result is None

    async def test_widget_event_rejects_unknown_tool(self):
        ctx = Context(
            system_prompt=create_system_prompt(),
            children=[],
            conversation_log=create_conversation_log(),
        )
        result = ctx.handle_widget_event("nonexistent", {})
        assert result is None

    async def test_text_response_recorded_in_conversation(self):
        ctx = Context(
            system_prompt=create_system_prompt(),
            children=[],
            conversation_log=create_conversation_log(),
        )
        ctx.add_user_message("Hello")
        response = ModelResponse(parts=[TextPart(content="Hi back!")])
        results, _ = await ctx.handle_response(response)
        assert results == []  # no tool calls
        assert len(ctx.conversation_log.model.turns) == 1
        assert len(ctx.conversation_log.model.turns[0].segments[0].messages) == 1
