"""Tests for the Context widget — composition and tool dispatch."""

from pydantic_ai import models
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
)

from calipso.widgets import (
    ActionLog,
    Context,
    Conversation,
    Goal,
    SystemPrompt,
    TaskList,
)

models.ALLOW_MODEL_REQUESTS = False


class TestContextComposition:
    def test_view_messages_composes_children(self):
        ctx = Context(
            children=[SystemPrompt(text="Hello"), Goal(text="Win")],
            conversation=Conversation(),
        )
        msgs = list(ctx.view_messages())
        contents = [
            p.content
            for m in msgs
            if isinstance(m, ModelRequest)
            for p in m.parts
            if isinstance(p, SystemPromptPart)
        ]
        assert any("Hello" in c for c in contents)
        assert any("Win" in c for c in contents)

    def test_view_tools_composes_children(self):
        ctx = Context(
            children=[Goal(), TaskList()],
            conversation=Conversation(),
        )
        tools = list(ctx.view_tools())
        names = {t.name for t in tools}
        assert "set_goal" in names
        assert "create_task" in names

    def test_view_messages_includes_conversation(self):
        ctx = Context(children=[], conversation=Conversation())
        ctx.add_user_message("Hi there")
        msgs = list(ctx.view_messages())
        assert any(
            hasattr(p, "content") and "Hi there" in p.content
            for m in msgs
            for p in m.parts
        )


class TestContextDispatch:
    def test_dispatch_tool_call_to_goal(self):
        ctx = Context(
            children=[Goal()],
            conversation=Conversation(),
        )
        response = ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="set_goal",
                    args={"goal": "Ship it"},
                    tool_call_id="call_1",
                )
            ]
        )
        results = ctx.handle_response(response)
        assert len(results) == 1
        assert "Ship it" in results[0][1]
        assert ctx.children[0].text == "Ship it"

    def test_dispatch_unknown_tool(self):
        ctx = Context(children=[], conversation=Conversation())
        response = ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="nonexistent",
                    args={},
                    tool_call_id="call_1",
                )
            ]
        )
        results = ctx.handle_response(response)
        assert "Unknown" in results[0][1]

    def test_action_log_protocol_enforcement(self):
        ctx = Context(
            children=[TaskList()],
            conversation=Conversation(),
            action_log=ActionLog(),
        )
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
        results = ctx.handle_response(response)
        assert "action_log_start" in results[0][1]
        # Task should NOT have been created
        task_list = ctx.children[0]
        assert len(task_list.tasks) == 0

    def test_text_response_recorded_in_conversation(self):
        ctx = Context(children=[], conversation=Conversation())
        ctx.add_user_message("Hello")
        response = ModelResponse(parts=[TextPart(content="Hi back!")])
        results = ctx.handle_response(response)
        assert results == []  # no tool calls
        assert len(ctx.conversation.turns) == 1
        assert len(ctx.conversation.turns[0].messages) == 1
