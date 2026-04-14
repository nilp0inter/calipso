"""Tests for the runner — agentic loop with TestModel/FunctionModel."""

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

models.ALLOW_MODEL_REQUESTS = False
pytestmark = pytest.mark.anyio


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
    assert len(ctx.conversation_log.model.turns) == 1


async def test_tool_call_then_text():
    """Model calls a tool (with action log protocol), then responds with text."""
    call_count = 0

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="action_log_start",
                        args={"description": "Set the goal"},
                        tool_call_id="call_0",
                    ),
                    ToolCallPart(
                        tool_name="set_goal",
                        args={"goal": "Test goal"},
                        tool_call_id="call_1",
                    ),
                    ToolCallPart(
                        tool_name="action_log_end",
                        args={"result": "Goal set to Test goal"},
                        tool_call_id="call_2",
                    ),
                ]
            )
        return ModelResponse(parts=[TextPart(content="Goal has been set!")])

    model = FunctionModel(respond)
    goal = create_goal()
    ctx = Context(
        system_prompt=create_system_prompt(),
        children=[goal],
        conversation_log=create_conversation_log(),
    )

    result = await run_turn(model, ctx, "Set a goal")
    assert result == "Goal has been set!"
    assert goal.model.text == "Test goal"


async def test_multiple_tool_calls_in_sequence():
    """Model makes multiple tool calls across two actions."""
    call_count = 0

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="action_log_start",
                        args={"description": "Set goal step 1"},
                        tool_call_id="call_0",
                    ),
                    ToolCallPart(
                        tool_name="set_goal",
                        args={"goal": "Step 1"},
                        tool_call_id="call_1",
                    ),
                    ToolCallPart(
                        tool_name="action_log_end",
                        args={"result": "Goal set to Step 1"},
                        tool_call_id="call_2",
                    ),
                ]
            )
        if call_count == 2:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="action_log_start",
                        args={"description": "Set goal step 2"},
                        tool_call_id="call_3",
                    ),
                    ToolCallPart(
                        tool_name="set_goal",
                        args={"goal": "Step 2"},
                        tool_call_id="call_4",
                    ),
                    ToolCallPart(
                        tool_name="action_log_end",
                        args={"result": "Goal set to Step 2"},
                        tool_call_id="call_5",
                    ),
                ]
            )
        return ModelResponse(parts=[TextPart(content="Done!")])

    model = FunctionModel(respond)
    goal = create_goal()
    ctx = Context(
        system_prompt=create_system_prompt(),
        children=[goal],
        conversation_log=create_conversation_log(),
    )

    result = await run_turn(model, ctx, "Do two things")
    assert result == "Done!"
    assert goal.model.text == "Step 2"
    assert call_count == 3
