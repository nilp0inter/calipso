import pytest
from pydantic_ai import models
from pydantic_ai.models.test import TestModel

from calipso.agent import agent

models.ALLOW_MODEL_REQUESTS = False
pytestmark = pytest.mark.anyio


async def test_agent_responds():
    with agent.override(model=TestModel(custom_output_text="Hello! I'm Calipso.")):
        result = await agent.run("Hello!")
    assert "Calipso" in result.output
