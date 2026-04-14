"""AgentsMd widget — loads behavioral instructions from AGENTS.md or CLAUDE.md."""

import html as html_mod
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path

from pydantic_ai.messages import ModelMessage, ModelRequest, SystemPromptPart
from pydantic_ai.tools import ToolDefinition

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


AgentsMdMsg = AgentsReloaded


# --- Update (pure) ---


def update(model: AgentsMdModel, msg: AgentsMdMsg) -> tuple[AgentsMdModel, str]:
    match msg:
        case AgentsReloaded(loaded_path=lp, content=c, error=e):
            new_model = replace(model, loaded_path=lp, content=c, error=e)
            if e:
                return new_model, e
            return new_model, f"Loaded: {lp}"


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


def _load_from_disk(directory: Path) -> AgentsReloaded:
    """Read AGENTS.md or CLAUDE.md from disk and return a Msg."""
    for name in _CANDIDATES:
        p = directory / name
        try:
            text = p.read_text()
        except (FileNotFoundError, OSError):
            continue
        if text.strip():
            return AgentsReloaded(loaded_path=str(p), content=text, error=None)
    return AgentsReloaded(
        loaded_path=None,
        content=None,
        error=f"Neither AGENTS.md nor CLAUDE.md found in {directory}",
    )


# --- Anticorruption layers ---


def _create_from_llm(directory: Path):
    async def from_llm(model: AgentsMdModel, tool_name: str, args: dict) -> AgentsMdMsg:
        match tool_name:
            case "reload_agents_md":
                return _load_from_disk(directory)
        raise ValueError(f"AgentsMd: unknown tool '{tool_name}'")

    return from_llm


def _create_from_ui(directory: Path):
    def from_ui(
        model: AgentsMdModel, event_name: str, args: dict
    ) -> AgentsMdMsg | None:
        match event_name:
            case "reload_agents_md":
                return _load_from_disk(directory)
        return None

    return from_ui


# --- Factory ---


def create_agents_md(directory: Path | None = None) -> WidgetHandle:
    if directory is None:
        directory = Path.cwd()

    initial_msg = _load_from_disk(directory)
    initial_model = AgentsMdModel(
        loaded_path=initial_msg.loaded_path,
        content=initial_msg.content,
        error=initial_msg.error,
    )

    return create_widget(
        id="widget-agents-md",
        model=initial_model,
        update=update,
        view_messages=view_messages,
        view_tools=view_tools,
        view_html=view_html,
        from_llm=_create_from_llm(directory),
        from_ui=_create_from_ui(directory),
        frontend_tools=frozenset({"reload_agents_md"}),
    )
