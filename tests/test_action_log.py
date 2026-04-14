from unittest.mock import MagicMock

import pytest
from pydantic_ai import Agent, ModelRequest, ModelResponse, models
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from calipso.capabilities.action_log import ActionLog, LogEntry

models.ALLOW_MODEL_REQUESTS = False
pytestmark = pytest.mark.anyio


class TestActionLogState:
    def test_initial_state_empty(self):
        al = ActionLog()
        assert al.entries == []
        assert al._active_action is None
        assert al._allowed_tool is None
        assert al._next_id == 1

    def test_initial_with_entries(self):
        al = ActionLog(
            entries=[
                LogEntry(id=1, action="Do thing", result="Done"),
                LogEntry(id=3, action="Do other", result="Also done"),
            ]
        )
        assert al._next_id == 4

    def test_instructions_contain_rules(self):
        al = ActionLog()
        instructions = al.get_instructions()
        assert "MUST call action_log_start" in instructions
        assert "action_log_end" in instructions


class TestActionLogTools:
    async def test_start_then_end(self):
        al = ActionLog()
        agent = Agent("test", defer_model_check=True, capabilities=[al])

        call_count = 0

        def model_fn(
            messages: list[ModelRequest | ModelResponse], info: AgentInfo
        ) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="action_log_start",
                            args={"description": "Do the thing"},
                        )
                    ]
                )
            elif call_count == 2:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="action_log_end",
                            args={"result": "Thing done"},
                        )
                    ]
                )
            else:
                return ModelResponse(parts=[TextPart(content="All done!")])

        with agent.override(model=FunctionModel(model_fn)):
            await agent.run("Do something")

        assert len(al.entries) == 1
        assert al.entries[0].id == 1
        assert al.entries[0].action == "Do the thing"
        assert al.entries[0].result == "Thing done"
        assert al._active_action is None


class TestProtocolEnforcement:
    async def test_tool_before_start_raises(self):
        al = ActionLog()
        call = MagicMock()
        call.tool_name = "create_task"
        with pytest.raises(ModelRetry, match="Call action_log_start first"):
            await al.before_tool_execute(
                None,
                call=call,
                tool_def=None,
                args={},  # type: ignore
            )

    async def test_action_log_tools_always_allowed(self):
        al = ActionLog()
        call = MagicMock()
        call.tool_name = "action_log_start"
        result = await al.before_tool_execute(
            None,
            call=call,
            tool_def=None,
            args={"description": "x"},  # type: ignore
        )
        assert result == {"description": "x"}

    async def test_first_tool_after_start_locks(self):
        al = ActionLog()
        al._active_action = "test"
        call = MagicMock()
        call.tool_name = "create_task"
        await al.before_tool_execute(
            None,
            call=call,
            tool_def=None,
            args={},  # type: ignore
        )
        assert al._allowed_tool == "create_task"

    async def test_different_tool_after_lock_raises(self):
        al = ActionLog()
        al._active_action = "test"
        al._allowed_tool = "create_task"
        call = MagicMock()
        call.tool_name = "set_goal"
        with pytest.raises(ModelRetry, match="Only 'create_task' is allowed"):
            await al.before_tool_execute(
                None,
                call=call,
                tool_def=None,
                args={},  # type: ignore
            )

    async def test_same_tool_after_lock_passes(self):
        al = ActionLog()
        al._active_action = "test"
        al._allowed_tool = "create_task"
        call = MagicMock()
        call.tool_name = "create_task"
        result = await al.before_tool_execute(
            None,
            call=call,
            tool_def=None,
            args={"x": 1},  # type: ignore
        )
        assert result == {"x": 1}


class TestCompaction:
    def _make_action_messages(self):
        """Build a realistic message sequence for one completed action."""
        return [
            # 0: user prompt
            ModelRequest(parts=[UserPromptPart(content="do something")]),
            # 1: model calls action_log_start
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="action_log_start",
                        args={"description": "Do the thing"},
                        tool_call_id="c1",
                    )
                ]
            ),
            # 2: tool return for start
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name="action_log_start",
                        content="Action started: Do the thing",
                        tool_call_id="c1",
                    )
                ]
            ),
            # 3: model calls the actual tool
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="set_goal",
                        args={"goal": "test"},
                        tool_call_id="c2",
                    )
                ]
            ),
            # 4: tool return
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name="set_goal",
                        content="Goal set: test",
                        tool_call_id="c2",
                    )
                ]
            ),
            # 5: model calls action_log_end
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="action_log_end",
                        args={"result": "Goal was set"},
                        tool_call_id="c3",
                    )
                ]
            ),
            # 6: tool return for end
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name="action_log_end",
                        content="Action #1 logged.",
                        tool_call_id="c3",
                    )
                ]
            ),
        ]

    def test_find_compaction_ranges(self):
        msgs = self._make_action_messages()
        ranges = ActionLog._find_compaction_ranges(msgs)
        assert len(ranges) == 1
        start, end, action, result = ranges[0]
        assert start == 1
        assert end == 7
        assert action == "Do the thing"
        assert result == "Goal was set"

    async def test_compaction_replaces_with_summary(self):
        al = ActionLog()
        al._needs_compaction = True

        from pydantic_ai.models import ModelRequestContext

        msgs = self._make_action_messages()
        # Add trailing ModelRequest (the current request pydantic-ai appends)
        msgs.append(ModelRequest(parts=[UserPromptPart(content="next turn")]))

        rc = ModelRequestContext(
            model=None,  # type: ignore
            messages=msgs,
            model_settings=None,
            model_request_parameters=None,  # type: ignore
        )

        result = await al.before_model_request(None, rc)  # type: ignore
        # user prompt + summary + trailing current request = 3 messages
        assert len(result.messages) == 3
        assert result.messages[0].parts[0].content == "do something"
        summary = result.messages[1]
        assert isinstance(summary, ModelResponse)
        assert summary.parts[0].content == (
            "- Task: Do the thing\n  Result: Goal was set"
        )
        # Last message is the current ModelRequest (preserved)
        assert isinstance(result.messages[2], ModelRequest)
        assert result.messages[2].parts[0].content == "next turn"

    def test_summary_message_format(self):
        entry = LogEntry(id=1, action="Refactor parse", result="3 files changed")
        msg = ActionLog._summary_message(entry)
        assert isinstance(msg, ModelResponse)
        assert msg.parts[0].content == (
            "- Task: Refactor parse\n  Result: 3 files changed"
        )
