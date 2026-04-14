"""Elm-style Cmd type for describing side effects.

In Elm, ``update`` returns ``(Model, Cmd Msg)`` — a description of an
effect, not its execution.  The runtime executes the Cmd and feeds the
result back as a new Msg.

Three variants:

* ``CmdNone`` — no effect, no response.
* ``CmdToolResult(text)`` — respond to an LLM tool call with text.
* ``CmdEffect(perform, to_msg)`` — an async thunk the runtime awaits,
  then converts the raw return value into a Msg via ``to_msg``.

The ``Initiator`` type (``LLM`` / ``UI``) is carried on Msgs that can
originate from either path, so ``update`` knows whether to respond.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# Initiator — who triggered this Msg
# ---------------------------------------------------------------------------


class Initiator(Enum):
    LLM = "llm"
    UI = "ui"


# ---------------------------------------------------------------------------
# Cmd variants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CmdNone:
    """No effect, no response."""


@dataclass(frozen=True)
class CmdToolResult:
    """No effect; respond to the LLM tool call with the given text."""

    text: str


@dataclass(frozen=True)
class CmdEffect:
    """A side effect to execute.

    ``perform`` is an async thunk (zero-arg) that does I/O.
    ``to_msg`` converts the raw return value into a typed Msg
    which is fed back into ``update``.
    """

    perform: Callable[[], Awaitable[Any]]
    to_msg: Callable[[Any], Any]


Cmd = CmdNone | CmdToolResult | CmdEffect

none = CmdNone()
"""Singleton — no effect, no response."""


def tool_result(text: str) -> CmdToolResult:
    """Respond to an LLM tool call with the given text."""
    return CmdToolResult(text=text)


def effect(
    perform: Callable[[], Awaitable[Any]],
    to_msg: Callable[[Any], Any],
) -> CmdEffect:
    """Describe a side effect to execute."""
    return CmdEffect(perform=perform, to_msg=to_msg)


def for_initiator(initiator: Initiator, text: str) -> Cmd:
    """Return ``tool_result(text)`` for LLM, ``none`` for UI."""
    match initiator:
        case Initiator.LLM:
            return tool_result(text)
        case Initiator.UI:
            return none
