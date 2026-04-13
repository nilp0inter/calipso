import pytest
from pydantic_ai import Agent, models
from pydantic_ai.models.test import TestModel

from calipso.capabilities.goal import Goal

models.ALLOW_MODEL_REQUESTS = False
pytestmark = pytest.mark.anyio


class TestGoalState:
    def test_initial_no_goal(self):
        g = Goal()
        assert g.text is None

    def test_initial_with_goal(self):
        g = Goal(text="Ship v1")
        assert g.text == "Ship v1"

    def test_render_no_goal(self):
        g = Goal()
        assert g._render() == "Goal: No goal set"

    def test_render_with_goal(self):
        g = Goal(text="Ship v1")
        assert g._render() == "Goal: Ship v1"


class TestGoalTools:
    async def test_set_goal(self):
        g = Goal()
        agent = Agent("test", defer_model_check=True, capabilities=[g])
        with agent.override(
            model=TestModel(call_tools=["set_goal"], custom_output_text="Goal set!")
        ):
            await agent.run("Set a goal")

        assert g.text is not None

    async def test_clear_goal(self):
        g = Goal(text="Old goal")
        agent = Agent("test", defer_model_check=True, capabilities=[g])
        with agent.override(
            model=TestModel(call_tools=["clear_goal"], custom_output_text="Cleared!")
        ):
            await agent.run("Clear the goal")

        assert g.text is None
