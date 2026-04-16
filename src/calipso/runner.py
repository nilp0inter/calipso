"""Agentic loop that drives the Context widget via Model.request()."""

from collections.abc import Awaitable, Callable

from pydantic_ai.models import Model, ModelRequestParameters

from calipso.widgets.context import Context


async def run_turn(
    model: Model,
    context: Context,
    user_input: str,
    on_update: Callable[[], Awaitable[None]] | None = None,
) -> str:
    """Execute a single conversational turn (may involve multiple model calls).

    Adds the user message, then loops: compose context → call model →
    dispatch tool calls → repeat until the model produces a text response.

    If *on_update* is provided it is awaited after every state mutation
    (user message added, tool calls dispatched) so that observers (e.g. the
    browser dashboard) can push fresh widget HTML.
    """
    context.add_user_message(user_input)
    if on_update:
        await on_update()

    while True:
        messages = list(context.view_messages())
        tools = list(context.view_tools())
        params = ModelRequestParameters(function_tools=tools)

        response = await model.request(messages, None, params)

        tool_results = await context.handle_response(response, on_update=on_update)

        if on_update:
            await on_update()

        if not tool_results:
            # No tool calls — the model produced a text response
            return response.text or ""
