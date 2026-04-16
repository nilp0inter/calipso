"""Tests for the runner — agentic loop with FunctionModel."""

import pytest
from pydantic_ai import models
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from calipso.runner import run_turn
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


async def test_simple_text_response():
    """Model returns text immediately — no tool calls."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content="Hello from Calipso!")])

    model = FunctionModel(respond)
    ctx = Context(
        system_prompt=create_system_prompt(),
        conversation_log=create_conversation_log(),
    )

    result = await run_turn(model, ctx, "Hi")
    assert result == "Hello from Calipso!"
    # The user message and the text response both got logged.
    log = ctx.conversation_log.model.log
    assert any(item.user_message == "Hi" for item in log)
    assert any(
        item.response is not None
        and any(isinstance(p, TextPart) for p in item.response.parts)
        for item in log
    )


async def test_goal_callable_without_task():
    """set_goal is protocol-free: the model can call it directly with no task."""
    call_count = 0

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(parts=[_tc("set_goal", {"goal": "Ship"}, "c1")])
        return ModelResponse(parts=[TextPart(content="done")])

    model = FunctionModel(respond)
    goal = create_goal()
    ctx = Context(
        system_prompt=create_system_prompt(),
        children=[goal],
        conversation_log=create_conversation_log(),
    )

    result = await run_turn(model, ctx, "Set a goal")
    assert result == "done"
    assert goal.model.text == "Ship"


async def test_full_task_lifecycle():
    """Model runs create → start → memory → memory → close and produces text."""
    call_count = 0

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(
                parts=[
                    _tc("create_task", {"description": "Explore repo"}, "c0"),
                    _tc("start_task", {"task_id": 1}, "c1"),
                    _tc("set_goal", {"goal": "Learn"}, "c2"),
                ]
            )
        if call_count == 2:
            return ModelResponse(
                parts=[
                    _tc(
                        "task_memory",
                        {"text": "Goal is Learn; no other state yet"},
                        "c3",
                    ),
                ]
            )
        if call_count == 3:
            return ModelResponse(parts=[_tc("close_current_task", {}, "c4")])
        return ModelResponse(parts=[TextPart(content="All done.")])

    model = FunctionModel(respond)
    goal = create_goal()
    ctx = Context(
        system_prompt=create_system_prompt(),
        children=[goal],
        conversation_log=create_conversation_log(),
    )

    result = await run_turn(model, ctx, "Please explore")
    assert result == "All done."
    assert goal.model.text == "Learn"
    tasks = ctx.conversation_log.model.tasks
    assert tasks[1].status == TaskStatus.DONE
    assert tasks[1].memories == ["Goal is Learn; no other state yet"]


async def test_tool_outside_task_is_rejected():
    """A non-protocol-free tool without an active task yields an error result,
    and the runner keeps looping until the model gives up with text."""
    call_count = 0

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Attempt to call a phony non-protocol-free tool.
            return ModelResponse(parts=[_tc("no_such_tool", {}, "c0")])
        return ModelResponse(parts=[TextPart(content="gave up")])

    model = FunctionModel(respond)
    ctx = Context(
        system_prompt=create_system_prompt(),
        conversation_log=create_conversation_log(),
    )
    result = await run_turn(model, ctx, "nope")
    assert result == "gave up"
