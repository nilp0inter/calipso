"""CLI entry point for Calipso."""

import json
from pathlib import Path

import httpx

from calipso.model import create_http_client, create_model
from calipso.runner import run_turn_sync
from calipso.widgets import (
    AgentsMd,
    Context,
    ConversationLog,
    Goal,
    SystemPrompt,
    TaskList,
)

PROMPT_DIR = Path("prompts")


def _next_num() -> int:
    """Return the next available file number in PROMPT_DIR."""
    existing = sorted(PROMPT_DIR.glob("*.json"))
    return int(existing[-1].stem) + 1 if existing else 0


class WireCapture:
    """Captures raw HTTP request/response pairs sent to the model provider."""

    def __init__(self):
        self.exchanges: list[dict] = []
        self._pending_request: dict | None = None

    async def on_request(self, request: httpx.Request) -> None:
        try:
            body = json.loads(request.content)
        except (json.JSONDecodeError, ValueError):
            body = request.content.decode(errors="replace")
        self._pending_request = {
            "method": request.method,
            "url": str(request.url),
            "body": body,
        }

    async def on_response(self, response: httpx.Response) -> None:
        await response.aread()
        try:
            body = json.loads(response.content)
        except (json.JSONDecodeError, ValueError):
            body = response.content.decode(errors="replace")
        self.exchanges.append(
            {
                "request": self._pending_request,
                "response": {
                    "status_code": response.status_code,
                    "body": body,
                },
            }
        )
        self._pending_request = None


def main():
    PROMPT_DIR.mkdir(exist_ok=True)
    turn_num = _next_num()

    capture = WireCapture()
    http_client = create_http_client(
        request_hook=capture.on_request,
        response_hook=capture.on_response,
    )
    model = create_model(http_client=http_client)

    context = Context(
        system_prompt=SystemPrompt(),
        children=[
            AgentsMd(),
            Goal(),
            TaskList(),
        ],
        conversation_log=ConversationLog(),
    )

    while True:
        try:
            user_input = input("You: ")
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if not user_input.strip():
            continue

        capture.exchanges.clear()
        output = run_turn_sync(model, context, user_input)

        out = PROMPT_DIR / f"{turn_num:04d}.json"
        out.write_text(json.dumps(capture.exchanges, indent=2))
        turn_num += 1

        print(f"Calipso: {output}")
