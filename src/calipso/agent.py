import os
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any

import httpx

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from calipso.capabilities import Goal, TaskList


@dataclass
class SystemPrompt(AbstractCapability[Any]):
    text: str = (
        "You are Calipso, a friendly AI assistant."
        " Say hello and introduce yourself briefly."
    )

    def get_instructions(self):
        return self.text


def create_http_client(
    *,
    request_hook: Callable[[httpx.Request], Coroutine[Any, Any, None]] | None = None,
    response_hook: Callable[[httpx.Response], Coroutine[Any, Any, None]] | None = None,
) -> httpx.AsyncClient:
    """Create an httpx client with optional event hooks for capturing wire traffic."""
    hooks: dict[str, list[httpx.EventHook]] = {"request": [], "response": []}
    if request_hook is not None:
        hooks["request"].append(request_hook)
    if response_hook is not None:
        hooks["response"].append(response_hook)
    return httpx.AsyncClient(
        timeout=httpx.Timeout(timeout=30, connect=5),
        event_hooks=hooks,
    )


def create_agent(http_client: httpx.AsyncClient | None = None) -> Agent:
    """Create the Calipso agent, optionally with a custom httpx client."""
    provider = OpenAIProvider(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        http_client=http_client,
    )
    model = OpenAIChatModel("minimax/minimax-m2.7", provider=provider)
    return Agent(
        model,
        defer_model_check=True,
        capabilities=[
            SystemPrompt(),
            TaskList(),
            Goal(),
        ],
    )


agent = create_agent()
