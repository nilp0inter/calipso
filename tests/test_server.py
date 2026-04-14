"""Tests for DashboardServer — HTTP and WebSocket endpoints."""

import pytest
from aiohttp.test_utils import TestClient, TestServer
from pydantic_ai import models

from calipso.server import DashboardServer, _add_oob
from calipso.widgets import (
    Context,
    create_conversation_log,
    create_goal,
    create_system_prompt,
    create_task_list,
)

models.ALLOW_MODEL_REQUESTS = False

pytestmark = pytest.mark.anyio


def _make_context():
    return Context(
        system_prompt=create_system_prompt(),
        children=[create_goal(), create_task_list()],
        conversation_log=create_conversation_log(),
    )


async def test_index_returns_html():
    ctx = _make_context()
    server = DashboardServer(ctx)
    async with TestClient(TestServer(server._app)) as client:
        resp = await client.get("/")
        assert resp.status == 200
        text = await resp.text()
        assert "Calipso Dashboard" in text


async def test_ws_receives_initial_state():
    ctx = _make_context()
    server = DashboardServer(ctx)
    async with TestClient(TestServer(server._app)) as client:
        async with client.ws_connect("/ws") as ws:
            msg = await ws.receive_str()
            assert "hx-swap-oob" in msg
            assert "widget-system-prompt" in msg
            assert "widget-goal" in msg


async def test_ws_sends_user_input():
    ctx = _make_context()
    server = DashboardServer(ctx)
    async with TestClient(TestServer(server._app)) as client:
        async with client.ws_connect("/ws") as ws:
            _ = await ws.receive_str()  # consume initial state
            await ws.send_json({"user_input": "Hello"})
            text = await server.input_queue.get()
            assert text == "Hello"


async def test_push_updates_sends_changed():
    ctx = _make_context()
    server = DashboardServer(ctx)
    async with TestClient(TestServer(server._app)) as client:
        async with client.ws_connect("/ws") as ws:
            _ = await ws.receive_str()  # consume initial state

            # Mutate a widget and push
            await ctx.children[0].dispatch_llm("set_goal", {"goal": "Test goal"})
            await server.push_updates()

            msg = await ws.receive_str()
            assert "Test goal" in msg
            assert "hx-swap-oob" in msg


def test_add_oob_injects_attribute():
    fragment = '<div id="widget-goal" class="widget">content</div>'
    result = _add_oob(fragment)
    assert result.startswith('<div hx-swap-oob="true"')
    assert 'id="widget-goal"' in result
