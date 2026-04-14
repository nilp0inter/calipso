"""FileExplorer widget — filesystem navigation and non-Python file reading."""

import html as html_mod
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path

from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.tools import ToolDefinition

from calipso.widget import WidgetHandle, create_widget

# --- Model ---


@dataclass(frozen=True)
class FileExplorerModel:
    listing_path: str | None = None
    listing_entries: tuple[tuple[str, bool], ...] | None = None
    listing_text: str | None = None
    open_file_path: str | None = None
    open_file_content: str | None = None


# --- Messages ---


@dataclass(frozen=True)
class DirectoryListed:
    path: str
    entries: tuple[tuple[str, bool], ...]
    listing_text: str


@dataclass(frozen=True)
class DirectoryListError:
    error: str


@dataclass(frozen=True)
class FileRead:
    path: str
    content: str


@dataclass(frozen=True)
class FileReadError:
    error: str


@dataclass(frozen=True)
class CloseReadFile:
    pass


FileExplorerMsg = (
    DirectoryListed | DirectoryListError | FileRead | FileReadError | CloseReadFile
)


# --- Update (pure) ---


def update(
    model: FileExplorerModel, msg: FileExplorerMsg
) -> tuple[FileExplorerModel, str]:
    match msg:
        case DirectoryListed(path=path, entries=entries, listing_text=text):
            return (
                replace(
                    model,
                    listing_path=path,
                    listing_entries=entries,
                    listing_text=text,
                ),
                text,
            )
        case DirectoryListError(error=error):
            return model, error
        case FileRead(path=path, content=content):
            return (
                replace(model, open_file_path=path, open_file_content=content),
                content,
            )
        case FileReadError(error=error):
            return model, error
        case CloseReadFile():
            if model.open_file_path is None:
                return model, "No file is open."
            path = model.open_file_path
            return (
                replace(model, open_file_path=None, open_file_content=None),
                f"Closed: {path}",
            )


# --- Views ---

_TOOL_DEFS = [
    ToolDefinition(
        name="list_directory",
        description=(
            "List files and subdirectories at a given path. "
            "Directories are marked with a trailing /."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to list. Defaults to '.'.",
                    "default": ".",
                },
            },
        },
    ),
    ToolDefinition(
        name="read_file",
        description=(
            "Read a non-Python file's contents. "
            "For Python files, use open_file from the Code Explorer instead."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read.",
                },
            },
            "required": ["path"],
        },
    ),
    ToolDefinition(
        name="close_read_file",
        description="Close the currently open file in the File Explorer.",
        parameters_json_schema={"type": "object", "properties": {}},
    ),
]


def view_messages(model: FileExplorerModel) -> Iterator[ModelMessage]:
    lines = ["## File Explorer"]
    if model.listing_text is not None:
        lines.append("")
        lines.append(model.listing_text)
    if model.open_file_path is not None:
        lines.append("")
        lines.append(f"**Open file:** `{model.open_file_path}`")
        lines.append("```")
        lines.append(model.open_file_content or "")
        lines.append("```")
    if model.listing_text is None and model.open_file_path is None:
        lines.append("No file open.")
    yield ModelRequest(parts=[UserPromptPart(content="\n".join(lines))])


def view_tools(model: FileExplorerModel) -> Iterator[ToolDefinition]:
    yield from _TOOL_DEFS


def view_html(model: FileExplorerModel) -> str:
    parts = []

    if model.listing_entries is not None:
        items = []
        for name, is_dir in model.listing_entries:
            escaped = html_mod.escape(name)
            cls = "entry-dir" if is_dir else "entry-file"
            suffix = "/" if is_dir else ""
            items.append(f'<li class="{cls}">{escaped}{suffix}</li>')
        header = html_mod.escape(model.listing_path or ".")
        parts.append(
            f"<p><strong>{header}</strong></p>"
            f'<ul class="dir-listing">{"".join(items)}</ul>'
        )

    if model.open_file_path is not None:
        filename = Path(model.open_file_path).name
        escaped_path = html_mod.escape(model.open_file_path)
        close_btn = (
            "<button onclick=\"sendWidgetEvent('close_read_file', {})\""
            ' class="btn-remove" title="Close file">'
            "&times;</button>"
        )
        escaped_content = html_mod.escape(model.open_file_content or "")
        parts.append(
            f'<div class="file-entry" title="{escaped_path}">'
            f"<code>{html_mod.escape(filename)}</code>"
            f"{close_btn}</div>"
            f"<pre><code>{escaped_content}</code></pre>"
        )

    if not parts:
        content = "<p><em>No file open</em></p>"
    else:
        content = "".join(parts)

    return (
        '<div id="widget-file-explorer" class="widget">'
        f"<h3>File Explorer</h3>{content}</div>"
    )


# --- Anticorruption layers ---


async def from_llm(
    model: FileExplorerModel, tool_name: str, args: dict
) -> FileExplorerMsg:
    match tool_name:
        case "list_directory":
            return _do_list_directory(args.get("path", "."))
        case "read_file":
            return _do_read_file(args["path"])
        case "close_read_file":
            return CloseReadFile()
    raise ValueError(f"FileExplorer: unknown tool '{tool_name}'")


def from_ui(
    model: FileExplorerModel, event_name: str, args: dict
) -> FileExplorerMsg | None:
    match event_name:
        case "close_read_file":
            return CloseReadFile()
    return None


# --- I/O helpers ---


def _do_list_directory(path: str) -> FileExplorerMsg:
    p = Path(path)
    if not p.is_dir():
        return DirectoryListError(error=f"Not a directory: {path}")
    entries_raw = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name))
    entries = tuple((e.name, e.is_dir()) for e in entries_raw)
    lines = [f"Contents of `{path}`:", ""]
    for entry in entries_raw:
        suffix = "/" if entry.is_dir() else ""
        lines.append(f"- {entry.name}{suffix}")
    if not entries_raw:
        lines.append("(empty directory)")
    listing_text = "\n".join(lines)
    return DirectoryListed(path=path, entries=entries, listing_text=listing_text)


def _do_read_file(path: str) -> FileExplorerMsg:
    if path.endswith(".py"):
        return FileReadError(
            error=(
                "Python files should be read with the Code Explorer's "
                "open_file tool, not read_file."
            )
        )
    p = Path(path)
    if not p.is_file():
        return FileReadError(error=f"File not found: {path}")
    try:
        content = p.read_text()
    except UnicodeDecodeError:
        return FileReadError(error=f"Cannot read binary file: {path}")
    return FileRead(path=path, content=content)


# --- Factory ---


def create_file_explorer() -> WidgetHandle:
    return create_widget(
        id="widget-file-explorer",
        model=FileExplorerModel(),
        update=update,
        view_messages=view_messages,
        view_tools=view_tools,
        view_html=view_html,
        from_llm=from_llm,
        from_ui=from_ui,
        frontend_tools=frozenset({"close_read_file"}),
    )
