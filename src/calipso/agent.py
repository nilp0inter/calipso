import os
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from calipso.capabilities import ActionLog, Goal, TaskList


@dataclass
class SystemPrompt(AbstractCapability[Any]):
    text: str = (
        "You are Calipso, an AI coding assistant.\n# Current Status of the World"
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


def create_agent(
    http_client: httpx.AsyncClient | None = None,
    extra_capabilities: list[AbstractCapability[Any]] | None = None,
) -> Agent:
    """Create the Calipso agent, optionally with a custom httpx client."""
    provider = OpenAIProvider(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        http_client=http_client,
    )
    model = OpenAIChatModel("x-ai/grok-4-fast", provider=provider)
    capabilities: list[AbstractCapability[Any]] = [
        SystemPrompt(),
        ActionLog(),
        TaskList(),
        Goal(),
    ]
    if extra_capabilities:
        capabilities.extend(extra_capabilities)
    return Agent(
        model,
        defer_model_check=True,
        capabilities=capabilities,
    )


agent = create_agent()
