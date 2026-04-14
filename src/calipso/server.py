"""DashboardServer — aiohttp HTTP + WebSocket server for live widget visualization."""

import asyncio
import json
import re
from pathlib import Path

from aiohttp import web

from calipso.widgets.context import Context

STATIC_DIR = Path(__file__).parent / "static"


def _add_oob(fragment: str) -> str:
    """Inject hx-swap-oob="true" into the root element of an HTML fragment."""
    return re.sub(r"^(<\w+)", r'\1 hx-swap-oob="true"', fragment)


class DashboardServer:
    def __init__(
        self, context: Context, host: str = "localhost", port: int = 8080
    ) -> None:
        self.context = context
        self.host = host
        self.port = port
        self.input_queue: asyncio.Queue[str] = asyncio.Queue()
        self._clients: set[web.WebSocketResponse] = set()
        self._app = web.Application()
        self._app.router.add_get("/", self._handle_index)
        self._app.router.add_get("/ws", self._handle_ws)

    async def _handle_index(self, request: web.Request) -> web.Response:
        html = (STATIC_DIR / "index.html").read_text()
        return web.Response(text=html, content_type="text/html")

    async def _handle_ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self._clients.add(ws)

        # Send initial state for all widgets
        all_html = self.context.all_html()
        initial = "\n".join(_add_oob(f) for f in all_html)
        await ws.send_str(initial)

        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    if "widget_event" in data:
                        event = data["widget_event"]
                        self.context.handle_widget_event(
                            event["tool_name"],
                            event.get("args", {}),
                        )
                        await self.push_updates()
                    else:
                        user_input = data.get("user_input", "")
                        user_input = user_input.strip()
                        if user_input:
                            await self.input_queue.put(user_input)
        finally:
            self._clients.discard(ws)
        return ws

    async def push_turn_start(self) -> None:
        """Signal that a turn has started: show thinking indicator, disable input."""
        payload = "\n".join(
            [
                '<div id="thinking" hx-swap-oob="true" class="active">'
                '<span class="dot"></span><span class="dot"></span>'
                '<span class="dot"></span><span>Thinking...</span></div>',
                '<fieldset id="input-fieldset" hx-swap-oob="true" disabled>'
                '<input type="text" name="user_input" '
                'placeholder="Thinking..." autocomplete="off" />'
                "<button>Send</button></fieldset>",
            ]
        )
        await self._broadcast(payload)

    async def push_turn_end(self) -> None:
        """Signal that a turn has ended: hide thinking indicator, re-enable input."""
        payload = "\n".join(
            [
                '<div id="thinking" hx-swap-oob="true">'
                '<span class="dot"></span><span class="dot"></span>'
                '<span class="dot"></span><span>Thinking...</span></div>',
                '<fieldset id="input-fieldset" hx-swap-oob="true">'
                '<input type="text" name="user_input" '
                'placeholder="Type a message..." autocomplete="off" />'
                "<button>Send</button></fieldset>",
            ]
        )
        await self._broadcast(payload)

    async def _broadcast(self, payload: str) -> None:
        """Send a payload to all connected WebSocket clients."""
        dead: set[web.WebSocketResponse] = set()
        for ws in self._clients:
            try:
                await ws.send_str(payload)
            except Exception:
                dead.add(ws)
        self._clients -= dead

    async def push_updates(self) -> None:
        """Compute changed widgets and push HTML to all connected clients."""
        changed = self.context.changed_html()
        if not changed:
            return
        payload = "\n".join(_add_oob(f) for f in changed)
        await self._broadcast(payload)

    async def start(self) -> None:
        """Start the HTTP+WS server (non-blocking)."""
        runner = web.AppRunner(self._app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
