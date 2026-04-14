"""AgentsMd widget — loads behavioral instructions from AGENTS.md or CLAUDE.md."""

import html as html_mod
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path

from pydantic_ai.messages import ModelMessage, ModelRequest, SystemPromptPart
from pydantic_ai.tools import ToolDefinition

from calipso.cmd import Cmd, Initiator, effect, for_initiator
from calipso.widget import WidgetHandle, create_widget, render_md

_CANDIDATES = ("AGENTS.md", "CLAUDE.md")


# --- Model ---


@dataclass(frozen=True)
class AgentsMdModel:
    loaded_path: str | None = None
    content: str | None = None
    error: str | None = None


# --- Messages ---


@dataclass(frozen=True)
class AgentsReloaded:
    loaded_path: str | None
    content: str | None
    error: str | None
    initiator: Initiator


@dataclass(frozen=True)
class ReloadRequested:
    initiator: Initiator


AgentsMdMsg = AgentsReloaded | ReloadRequested


# --- Update (pure) ---


def _create_update(directory: Path):
    def update(model: AgentsMdModel, msg: AgentsMdMsg) -> tuple[AgentsMdModel, Cmd]:
        match msg:
            case ReloadRequested(initiator=init):

                async def perform():
                    return _load_from_disk(directory, init)

                return model, effect(perform=perform, to_msg=lambda msg: msg)
            case AgentsReloaded(loaded_path=lp, content=c, error=e, initiator=init):
                new_model = replace(model, loaded_path=lp, content=c, error=e)
                if e:
                    return new_model, for_initiator(init, e)
                return new_model, for_initiator(init, f"Loaded: {lp}")

    return update


# --- Views ---

_TOOL_DEFS = [
    ToolDefinition(
        name="reload_agents_md",
        description="Reload behavioral instructions from AGENTS.md or CLAUDE.md.",
        parameters_json_schema={"type": "object", "properties": {}},
    ),
]


def view_messages(model: AgentsMdModel) -> Iterator[ModelMessage]:
    if model.content:
        yield ModelRequest(parts=[SystemPromptPart(content=model.content)])


def view_tools(model: AgentsMdModel) -> Iterator[ToolDefinition]:
    yield from _TOOL_DEFS


def view_html(model: AgentsMdModel) -> str:
    if model.error:
        warning = (
            f'<p class="agents-md-warning"><em>{html_mod.escape(model.error)}</em></p>'
        )
        body = warning
    else:
        filename = Path(model.loaded_path).name if model.loaded_path else ""
        body = f"<p><strong>{html_mod.escape(filename)}</strong></p>" + render_md(
            model.content or ""
        )

    reload_btn = (
        "<button onclick=\"sendWidgetEvent('reload_agents_md', {})\""
        ' class="btn-remove" title="Reload">'
        "Reload</button>"
    )
    return (
        '<div id="widget-agents-md" class="widget">'
        f"<h3>AGENTS.md {reload_btn}</h3>{body}</div>"
    )


# --- I/O helper ---


def _load_from_disk(directory: Path, initiator: Initiator) -> AgentsReloaded:
    """Read AGENTS.md or CLAUDE.md from disk and return a Msg."""
    for name in _CANDIDATES:
        p = directory / name
        try:
            text = p.read_text()
        except (FileNotFoundError, OSError):
            continue
        if text.strip():
            return AgentsReloaded(
                loaded_path=str(p), content=text, error=None, initiator=initiator
            )
    return AgentsReloaded(
        loaded_path=None,
        content=None,
        error=f"Neither AGENTS.md nor CLAUDE.md found in {directory}",
        initiator=initiator,
    )


# --- Anticorruption layers ---


def from_llm(model: AgentsMdModel, tool_name: str, args: dict) -> AgentsMdMsg:
    match tool_name:
        case "reload_agents_md":
            return ReloadRequested(initiator=Initiator.LLM)
    raise ValueError(f"AgentsMd: unknown tool '{tool_name}'")


def from_ui(model: AgentsMdModel, event_name: str, args: dict) -> AgentsMdMsg | None:
    match event_name:
        case "reload_agents_md":
            return ReloadRequested(initiator=Initiator.UI)
    return None


# --- Factory ---


def create_agents_md(directory: Path | None = None) -> WidgetHandle:
    if directory is None:
        directory = Path.cwd()

    initial_msg = _load_from_disk(directory, Initiator.LLM)
    initial_model = AgentsMdModel(
        loaded_path=initial_msg.loaded_path,
        content=initial_msg.content,
        error=initial_msg.error,
    )

    return create_widget(
        id="widget-agents-md",
        model=initial_model,
        update=_create_update(directory),
        view_messages=view_messages,
        view_tools=view_tools,
        view_html=view_html,
        from_llm=from_llm,
        from_ui=from_ui,
        frontend_tools=frozenset({"reload_agents_md"}),
    )
