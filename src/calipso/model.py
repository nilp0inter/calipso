"""Model/provider setup for Calipso."""

import os
from collections.abc import Callable, Coroutine
from typing import Any

import httpx
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


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


def create_model(http_client: httpx.AsyncClient | None = None) -> Model:
    """Create the model instance with provider configuration."""
    provider = OpenAIProvider(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        http_client=http_client,
    )
    return OpenAIChatModel("x-ai/grok-4-fast", provider=provider)
