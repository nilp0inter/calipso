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
from calipso.widgets import Context, Conversation, Goal, SystemPrompt

models.ALLOW_MODEL_REQUESTS = False
pytestmark = pytest.mark.anyio


async def test_simple_text_response():
    """Model returns text immediately — no tool calls."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content="Hello from Calipso!")])

    model = FunctionModel(respond)
    ctx = Context(children=[SystemPrompt()], conversation=Conversation())

    result = await run_turn(model, ctx, "Hi")
    assert result == "Hello from Calipso!"
    assert len(ctx.conversation.turns) == 1


async def test_tool_call_then_text():
    """Model calls a tool, then responds with text."""
    call_count = 0

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="set_goal",
                        args={"goal": "Test goal"},
                        tool_call_id="call_1",
                    )
                ]
            )
        return ModelResponse(parts=[TextPart(content="Goal has been set!")])

    model = FunctionModel(respond)
    goal = Goal()
    ctx = Context(children=[goal], conversation=Conversation())

    result = await run_turn(model, ctx, "Set a goal")
    assert result == "Goal has been set!"
    assert goal.text == "Test goal"


async def test_multiple_tool_calls_in_sequence():
    """Model makes multiple tool calls across turns."""
    call_count = 0

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="set_goal",
                        args={"goal": "Step 1"},
                        tool_call_id="call_1",
                    )
                ]
            )
        if call_count == 2:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="set_goal",
                        args={"goal": "Step 2"},
                        tool_call_id="call_2",
                    )
                ]
            )
        return ModelResponse(parts=[TextPart(content="Done!")])

    model = FunctionModel(respond)
    goal = Goal()
    ctx = Context(children=[goal], conversation=Conversation())

    result = await run_turn(model, ctx, "Do two things")
    assert result == "Done!"
    assert goal.text == "Step 2"
    assert call_count == 3
