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
                    tool_name="begin_step",
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
                        tool_name="begin_step",
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

    async def test_step_protocol_enforcement(self):
        task_list = create_task_list()
        ctx = Context(
            system_prompt=create_system_prompt(),
            children=[task_list],
            conversation_log=create_conversation_log(),
        )
        ctx.add_user_message("test")
        # Try calling create_task without begin_step
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
        assert "begin_step" in results[0][1]
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
                        tool_name="begin_step",
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
        await ctx.handle_widget_event(
            "update_task_status", {"task_id": 1, "status": "done"}
        )
        assert task_list.model.tasks[0].status.value == "done"

    async def test_widget_event_rejects_non_frontend_tool(self):
        ctx = Context(
            system_prompt=create_system_prompt(),
            children=[],
            conversation_log=create_conversation_log(),
        )
        # begin_step is NOT in frontend_tools
        result = await ctx.handle_widget_event(
            "begin_step", {"description": "Hacked"}
        )
        assert result is None

    async def test_widget_event_rejects_unknown_tool(self):
        ctx = Context(
            system_prompt=create_system_prompt(),
            children=[],
            conversation_log=create_conversation_log(),
        )
        result = await ctx.handle_widget_event("nonexistent", {})
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

    async def test_end_step_rejected_when_not_first_tool_call(self):
        goal = create_goal()
        ctx = Context(
            system_prompt=create_system_prompt(),
            children=[goal],
            conversation_log=create_conversation_log(),
        )
        ctx.add_user_message("test")
        # Start action and call a tool
        await ctx.handle_response(
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="begin_step",
                        args={"description": "do stuff"},
                        tool_call_id="c0",
                    )
                ]
            )
        )
        await ctx.handle_response(
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="set_goal",
                        args={"goal": "Win"},
                        tool_call_id="c1",
                    )
                ]
            )
        )
        # Now send set_goal BEFORE end_step in same response
        results, _ = await ctx.handle_response(
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="set_goal",
                        args={"goal": "Lose"},
                        tool_call_id="c2",
                    ),
                    ToolCallPart(
                        tool_name="end_step",
                        args={"result": "hallucinated summary"},
                        tool_call_id="c3",
                    ),
                ]
            )
        )
        # end_step should be rejected
        end_result = next(r for r in results if r[0] == "c3")
        assert "first tool call" in end_result[1]
        # Action should still be active
        assert ctx.conversation_log.model.active_step is not None

    async def test_end_step_rejected_when_duplicated(self):
        goal = create_goal()
        ctx = Context(
            system_prompt=create_system_prompt(),
            children=[goal],
            conversation_log=create_conversation_log(),
        )
        ctx.add_user_message("test")
        # Start action and call a tool
        await ctx.handle_response(
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="begin_step",
                        args={"description": "do stuff"},
                        tool_call_id="c0",
                    )
                ]
            )
        )
        await ctx.handle_response(
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="set_goal",
                        args={"goal": "Win"},
                        tool_call_id="c1",
                    )
                ]
            )
        )
        # Two end_step calls in same response
        results, _ = await ctx.handle_response(
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="end_step",
                        args={"result": "first end"},
                        tool_call_id="c2",
                    ),
                    ToolCallPart(
                        tool_name="end_step",
                        args={"result": "second end"},
                        tool_call_id="c3",
                    ),
                ]
            )
        )
        # Both should be rejected
        assert all("first tool call" in r[1] for r in results)
        assert ctx.conversation_log.model.active_step is not None

    async def test_end_step_first_followed_by_new_action_succeeds(self):
        goal = create_goal()
        ctx = Context(
            system_prompt=create_system_prompt(),
            children=[goal],
            conversation_log=create_conversation_log(),
        )
        ctx.add_user_message("test")
        # Start action and call a tool
        await ctx.handle_response(
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="begin_step",
                        args={"description": "first action"},
                        tool_call_id="c0",
                    )
                ]
            )
        )
        await ctx.handle_response(
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="set_goal",
                        args={"goal": "Win"},
                        tool_call_id="c1",
                    )
                ]
            )
        )
        # end_step first, then begin_step — should both succeed
        results, _ = await ctx.handle_response(
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="end_step",
                        args={"result": "set the goal"},
                        tool_call_id="c2",
                    ),
                    ToolCallPart(
                        tool_name="begin_step",
                        args={"description": "second action"},
                        tool_call_id="c3",
                    ),
                ]
            )
        )
        end_result = next(r for r in results if r[0] == "c2")
        start_result = next(r for r in results if r[0] == "c3")
        assert "Step logged" in end_result[1]
        assert "Step started" in start_result[1]
        # New action should be active
        assert ctx.conversation_log.model.active_step == "second action"

    async def test_begin_step_rejected_when_action_already_active(self):
        goal = create_goal()
        ctx = Context(
            system_prompt=create_system_prompt(),
            children=[goal],
            conversation_log=create_conversation_log(),
        )
        ctx.add_user_message("test")
        # Start an action
        await ctx.handle_response(
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="begin_step",
                        args={"description": "first"},
                        tool_call_id="c0",
                    )
                ]
            )
        )
        # Try to start another without ending
        results, _ = await ctx.handle_response(
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="begin_step",
                        args={"description": "second"},
                        tool_call_id="c1",
                    )
                ]
            )
        )
        # begin_step is not exposed during an active step → Unknown tool
        assert "Unknown tool" in results[0][1]
        assert ctx.conversation_log.model.active_step == "first"

    async def test_end_step_alone_succeeds(self):
        goal = create_goal()
        ctx = Context(
            system_prompt=create_system_prompt(),
            children=[goal],
            conversation_log=create_conversation_log(),
        )
        ctx.add_user_message("test")
        # Start action and call a tool
        await ctx.handle_response(
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="begin_step",
                        args={"description": "do stuff"},
                        tool_call_id="c0",
                    )
                ]
            )
        )
        await ctx.handle_response(
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="set_goal",
                        args={"goal": "Win"},
                        tool_call_id="c1",
                    )
                ]
            )
        )
        # end_step alone
        results, _ = await ctx.handle_response(
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="end_step",
                        args={"result": "set the goal to Win"},
                        tool_call_id="c2",
                    ),
                ]
            )
        )
        assert "Step logged" in results[0][1]
        assert ctx.conversation_log.model.active_step is None
