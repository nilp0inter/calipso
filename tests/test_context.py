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
)
from calipso.widgets.conversation_log import TaskStatus

models.ALLOW_MODEL_REQUESTS = False
pytestmark = pytest.mark.anyio


def _tc(name: str, args: dict, call_id: str) -> ToolCallPart:
    return ToolCallPart(tool_name=name, args=args, tool_call_id=call_id)


async def _create_and_start(ctx: Context, desc: str = "work") -> int:
    """Helper: create a task and start it. Returns task id."""
    await ctx.handle_response(
        ModelResponse(parts=[_tc("create_task", {"description": desc}, "c0")])
    )
    # next_id was 1 for the first creation; just return 1 for callers that
    # create a single task.
    await ctx.handle_response(
        ModelResponse(parts=[_tc("start_task", {"task_id": 1}, "c1")])
    )
    return 1


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
            children=[create_goal()],
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
        sys_contents = [
            p.content
            for m in msgs
            if isinstance(m, ModelRequest)
            for p in m.parts
            if isinstance(p, SystemPromptPart)
        ]
        assert "Identity" in sys_contents[0]
        user_contents = [
            p.content
            for m in msgs
            if isinstance(m, ModelRequest)
            for p in m.parts
            if isinstance(p, UserPromptPart)
        ]
        # Identity/rules → user message → [CURRENT STATE → state panels → END STATE]
        assert "CURRENT STATE" in user_contents[-3]
        assert "Win" in user_contents[-2]
        assert "END STATE" in user_contents[-1]

    def test_protocol_free_collected_from_widgets(self):
        ctx = Context(
            system_prompt=create_system_prompt(),
            children=[create_goal()],
            conversation_log=create_conversation_log(),
        )
        assert {"set_goal", "clear_goal"} <= ctx._protocol_free_tool_names


class TestContextDispatch:
    async def test_task_management_tools_callable_without_task(self):
        """create_task, remove_task, start_task (when a pending task exists) are
        always callable — they are the protocol's own plumbing."""
        ctx = Context(
            system_prompt=create_system_prompt(),
            children=[],
            conversation_log=create_conversation_log(),
        )
        ctx.add_user_message("start")
        results, _ = await ctx.handle_response(
            ModelResponse(parts=[_tc("create_task", {"description": "Plan"}, "c0")])
        )
        assert "Created task 1" in results[0][1]

    async def test_goal_tool_callable_without_task(self):
        """Goal tools are declared protocol-free — callable outside a task."""
        goal = create_goal()
        ctx = Context(
            system_prompt=create_system_prompt(),
            children=[goal],
            conversation_log=create_conversation_log(),
        )
        ctx.add_user_message("test")
        results, _ = await ctx.handle_response(
            ModelResponse(parts=[_tc("set_goal", {"goal": "Ship it"}, "c1")])
        )
        assert "Ship it" in results[0][1]
        assert goal.model.text == "Ship it"

    async def test_other_tool_rejected_without_active_task(self):
        """Non-task, non-goal tools require an active task — rejected by
        the protocol check before owner lookup."""
        ctx = Context(
            system_prompt=create_system_prompt(),
            children=[],
            conversation_log=create_conversation_log(),
        )
        ctx.add_user_message("test")
        results, _ = await ctx.handle_response(
            ModelResponse(parts=[_tc("some_random_tool", {}, "c1")])
        )
        assert "outside a task" in results[0][1]

    async def test_dispatch_tool_to_goal_inside_task(self):
        goal = create_goal()
        ctx = Context(
            system_prompt=create_system_prompt(),
            children=[goal],
            conversation_log=create_conversation_log(),
        )
        ctx.add_user_message("set goal")
        await _create_and_start(ctx, "do it")
        response = ModelResponse(parts=[_tc("set_goal", {"goal": "Ship it"}, "c2")])
        results, _ = await ctx.handle_response(response)
        assert "Ship it" in results[0][1]
        assert goal.model.text == "Ship it"

    async def test_dispatch_unknown_tool(self):
        ctx = Context(
            system_prompt=create_system_prompt(),
            children=[],
            conversation_log=create_conversation_log(),
        )
        ctx.add_user_message("test")
        await _create_and_start(ctx)
        response = ModelResponse(parts=[_tc("nonexistent", {}, "c2")])
        results, _ = await ctx.handle_response(response)
        assert "Unknown" in results[0][1]

    async def test_start_task_rejected_when_another_active(self):
        """When a task is in_progress, start_task is no longer exposed by
        view_tools — Context rejects the call as an unknown tool."""
        ctx = Context(
            system_prompt=create_system_prompt(),
            children=[],
            conversation_log=create_conversation_log(),
        )
        ctx.add_user_message("test")
        await ctx.handle_response(
            ModelResponse(parts=[_tc("create_task", {"description": "A"}, "c0")])
        )
        await ctx.handle_response(
            ModelResponse(parts=[_tc("create_task", {"description": "B"}, "c0b")])
        )
        await ctx.handle_response(
            ModelResponse(parts=[_tc("start_task", {"task_id": 1}, "c1")])
        )
        assert ctx.conversation_log.model.active_task_id == 1
        # Attempt to start task 2 while 1 is active
        results, _ = await ctx.handle_response(
            ModelResponse(parts=[_tc("start_task", {"task_id": 2}, "c2")])
        )
        assert "Unknown tool" in results[0][1]
        assert ctx.conversation_log.model.active_task_id == 1

    async def test_close_current_task_rejected_without_memory(self):
        """close_current_task requires at least one memory."""
        ctx = Context(
            system_prompt=create_system_prompt(),
            children=[],
            conversation_log=create_conversation_log(),
        )
        ctx.add_user_message("test")
        await _create_and_start(ctx)
        # view_tools gates close_current_task behind memories ≥ 1, so the
        # LLM would normally not be offered it — but if it tries, the pre-scan
        # (not-in-exposed) path returns "Unknown tool".
        results, _ = await ctx.handle_response(
            ModelResponse(parts=[_tc("close_current_task", {}, "c2")])
        )
        assert "Unknown" in results[0][1]
        assert ctx.conversation_log.model.active_task_id == 1

    async def test_close_current_task_happy_path(self):
        """create → start → memory → close closes the task."""
        ctx = Context(
            system_prompt=create_system_prompt(),
            children=[],
            conversation_log=create_conversation_log(),
        )
        ctx.add_user_message("test")
        await _create_and_start(ctx)
        await ctx.handle_response(
            ModelResponse(parts=[_tc("task_memory", {"text": "note A"}, "c2")])
        )
        results, _ = await ctx.handle_response(
            ModelResponse(parts=[_tc("close_current_task", {}, "c3")])
        )
        assert "Closed task" in results[0][1]
        assert ctx.conversation_log.model.active_task_id is None
        assert ctx.conversation_log.model.tasks[1].status == TaskStatus.DONE

    async def test_close_current_task_must_be_first(self):
        """close_current_task alongside other tools in a response is rejected."""
        goal = create_goal()
        ctx = Context(
            system_prompt=create_system_prompt(),
            children=[goal],
            conversation_log=create_conversation_log(),
        )
        ctx.add_user_message("test")
        await _create_and_start(ctx)
        await ctx.handle_response(
            ModelResponse(parts=[_tc("task_memory", {"text": "note"}, "c2")])
        )
        # close_current_task with another tool call (not first).
        results, _ = await ctx.handle_response(
            ModelResponse(
                parts=[
                    _tc("set_goal", {"goal": "Win"}, "c3"),
                    _tc("close_current_task", {}, "c4"),
                ]
            )
        )
        close_result = next(r for r in results if r[0] == "c4")
        assert "first and only" in close_result[1]
        assert ctx.conversation_log.model.active_task_id == 1

    async def test_close_current_task_must_appear_once(self):
        ctx = Context(
            system_prompt=create_system_prompt(),
            children=[],
            conversation_log=create_conversation_log(),
        )
        ctx.add_user_message("test")
        await _create_and_start(ctx)
        await ctx.handle_response(
            ModelResponse(parts=[_tc("task_memory", {"text": "n"}, "c2")])
        )
        results, _ = await ctx.handle_response(
            ModelResponse(
                parts=[
                    _tc("close_current_task", {}, "c3"),
                    _tc("close_current_task", {}, "c4"),
                ]
            )
        )
        for r in results:
            assert "first and only" in r[1]
        assert ctx.conversation_log.model.active_task_id == 1

    async def test_widget_event_dispatches_to_frontend_tool(self):
        ctx = Context(
            system_prompt=create_system_prompt(),
            children=[],
            conversation_log=create_conversation_log(),
        )
        await ctx.handle_widget_event("create_task", {"description": "From UI"})
        assert len(ctx.conversation_log.model.tasks) == 1
        assert ctx.conversation_log.model.tasks[1].description == "From UI"

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
        log = ctx.conversation_log.model.log
        # user message + response recorded
        assert any(item.user_message == "Hello" for item in log)
        assert any(item.response is response for item in log)

    async def test_picks_consumed_at_handle_response_start(self):
        """Picks set in one response are consumed at the start of the next
        handle_response, so they apply to exactly one request/response cycle."""
        ctx = Context(
            system_prompt=create_system_prompt(),
            children=[],
            conversation_log=create_conversation_log(),
        )
        # Build up a done task to pick.
        ctx.add_user_message("test")
        await _create_and_start(ctx)
        await ctx.handle_response(
            ModelResponse(parts=[_tc("task_memory", {"text": "n"}, "c2")])
        )
        await ctx.handle_response(
            ModelResponse(parts=[_tc("close_current_task", {}, "c3")])
        )
        # LLM picks the done task in a response.
        await ctx.handle_response(
            ModelResponse(parts=[_tc("task_pick", {"task_id": 1}, "c4")])
        )
        assert ctx.conversation_log.model.picks_for_next_request == frozenset({1})
        # Next handle_response consumes the picks at the start.
        await ctx.handle_response(ModelResponse(parts=[TextPart(content="ok")]))
        assert ctx.conversation_log.model.picks_for_next_request == frozenset()
