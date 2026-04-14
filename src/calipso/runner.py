"""Agentic loop that drives the Context widget via Model.request()."""

from collections.abc import Awaitable, Callable

from pydantic_ai.messages import ModelRequest, ToolCallPart, ToolReturnPart
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

        tool_results, segment = context.handle_response(response)

        if on_update:
            await on_update()

        if not tool_results:
            # No tool calls — the model produced a text response
            return response.text or ""

        # Build tool return message and record it in the same segment
        tool_return_parts = [
            ToolReturnPart(
                tool_name=_find_tool_name(response, call_id),
                content=result,
                tool_call_id=call_id,
            )
            for call_id, result in tool_results
        ]
        tool_request = ModelRequest(parts=tool_return_parts)
        context.conversation_log.add_tool_results(tool_request, segment)


def _find_tool_name(response, call_id: str) -> str:
    """Find the tool name for a given call_id in the response."""
    for part in response.parts:
        if isinstance(part, ToolCallPart) and part.tool_call_id == call_id:
            return part.tool_name
    return "unknown"
