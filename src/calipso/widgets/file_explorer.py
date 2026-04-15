"""FileExplorer widget — filesystem navigation and non-Python file reading."""

import html as html_mod
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path

from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.tools import ToolDefinition

from calipso.cmd import Cmd, Initiator, effect, for_initiator, tool_result
from calipso.widget import WidgetHandle, create_widget

# --- Model ---


@dataclass(frozen=True)
class FileExplorerModel:
    listing_path: str | None = None
    listing_entries: tuple[tuple[str, bool], ...] | None = None
    listing_text: str | None = None
    open_files: tuple[tuple[str, str], ...] = ()


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
    path: str
    initiator: Initiator


@dataclass(frozen=True)
class ListDirectoryRequested:
    path: str


@dataclass(frozen=True)
class ReadFileRequested:
    path: str


FileExplorerMsg = (
    DirectoryListed
    | DirectoryListError
    | FileRead
    | FileReadError
    | CloseReadFile
    | ListDirectoryRequested
    | ReadFileRequested
)


# --- Update (pure) ---


def update(
    model: FileExplorerModel, msg: FileExplorerMsg
) -> tuple[FileExplorerModel, Cmd]:
    match msg:
        case ListDirectoryRequested(path=path):

            async def perform():
                return _do_list_directory(path)

            return model, effect(perform=perform, to_msg=lambda msg: msg)
        case ReadFileRequested(path=path):
            if path.endswith(".py"):
                return model, tool_result(
                    "Python files should be read with the Code Explorer's "
                    "open_file tool, not read_file."
                )

            async def perform():
                return _do_read_file(path)

            return model, effect(perform=perform, to_msg=lambda msg: msg)
        case DirectoryListed(path=path, entries=entries, listing_text=text):
            return (
                replace(
                    model,
                    listing_path=path,
                    listing_entries=entries,
                    listing_text=text,
                ),
                tool_result(f"Listed: {path}"),
            )
        case DirectoryListError(error=error):
            return model, tool_result(error)
        case FileRead(path=path, content=content):
            existing = tuple(
                (p, c) for p, c in model.open_files if p != path
            )
            return (
                replace(
                    model,
                    open_files=existing + ((path, content),),
                ),
                tool_result(f"Opened: {path}"),
            )
        case FileReadError(error=error):
            return model, tool_result(error)
        case CloseReadFile(path=path, initiator=init):
            if not any(p == path for p, _ in model.open_files):
                return model, for_initiator(init, "No file is open.")
            return (
                replace(
                    model,
                    open_files=tuple(
                        (p, c)
                        for p, c in model.open_files
                        if p != path
                    ),
                ),
                for_initiator(init, f"Closed: {path}"),
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
        description="Close an open file in the File Explorer.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path of the file to close.",
                },
            },
            "required": ["path"],
        },
    ),
]


def view_messages(model: FileExplorerModel) -> Iterator[ModelMessage]:
    lines = ["## File Explorer"]
    if model.listing_text is not None:
        lines.append("")
        lines.append(model.listing_text)
    for path, content in model.open_files:
        lines.append("")
        lines.append(f"**Open file:** `{path}`")
        lines.append("```")
        lines.append(content)
        lines.append("```")
    if model.listing_text is None and not model.open_files:
        lines.append("No file open.")
    yield ModelRequest(parts=[UserPromptPart(content="\n".join(lines))])


def view_tools(model: FileExplorerModel) -> Iterator[ToolDefinition]:
    yield from _TOOL_DEFS


def view_html(model: FileExplorerModel) -> str:
    parts = []

    root_btn = (
        "<button onclick=\"sendWidgetEvent('list_directory', {path: '.'})\""
        ' class="btn-root">Root</button>'
    )
    parts.append(root_btn)

    if model.listing_entries is not None:
        listing_path = model.listing_path or "."
        items = []
        for name, is_dir in model.listing_entries:
            escaped_name = html_mod.escape(name)
            child_path = f"{listing_path}/{name}"
            escaped_path = html_mod.escape(child_path, quote=True)
            if is_dir:
                items.append(
                    f'<li class="entry-dir" ondblclick="'
                    f"sendWidgetEvent('list_directory', "
                    f"{{path: '{escaped_path}'}})"
                    f'" style="cursor:pointer">'
                    f"{escaped_name}/</li>"
                )
            else:
                items.append(
                    f'<li class="entry-file" ondblclick="'
                    f"sendWidgetEvent('read_file', "
                    f"{{path: '{escaped_path}'}})"
                    f'" style="cursor:pointer">'
                    f"{escaped_name}</li>"
                )
        header = html_mod.escape(listing_path)
        parts.append(
            f"<p><strong>{header}</strong></p>"
            f'<ul class="dir-listing">{"".join(items)}</ul>'
        )

    for file_path, file_content in model.open_files:
        filename = Path(file_path).name
        escaped_path = html_mod.escape(file_path, quote=True)
        close_btn = (
            "<button onclick=\"sendWidgetEvent('close_read_file',"
            f" {{path: '{escaped_path}'}})\""
            ' class="btn-remove" title="Close file">'
            "&times;</button>"
        )
        escaped_content = html_mod.escape(file_content)
        parts.append(
            f'<div class="file-entry" title="{escaped_path}">'
            f"<code>{html_mod.escape(filename)}</code>"
            f"{close_btn}</div>"
            f"<pre><code>{escaped_content}</code></pre>"
        )

    content = "".join(parts)

    return (
        '<div id="widget-file-explorer" class="widget">'
        f"<h3>File Explorer</h3>{content}</div>"
    )


# --- Anticorruption layers ---


def from_llm(model: FileExplorerModel, tool_name: str, args: dict) -> FileExplorerMsg:
    match tool_name:
        case "list_directory":
            return ListDirectoryRequested(path=args.get("path", "."))
        case "read_file":
            return ReadFileRequested(path=args["path"])
        case "close_read_file":
            return CloseReadFile(
                path=args["path"], initiator=Initiator.LLM
            )
    raise ValueError(f"FileExplorer: unknown tool '{tool_name}'")


def from_ui(
    model: FileExplorerModel, event_name: str, args: dict
) -> FileExplorerMsg | None:
    match event_name:
        case "list_directory":
            return ListDirectoryRequested(path=args.get("path", "."))
        case "read_file":
            return ReadFileRequested(path=args["path"])
        case "close_read_file":
            return CloseReadFile(
                path=args["path"], initiator=Initiator.UI
            )
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
        frontend_tools=frozenset({"list_directory", "read_file", "close_read_file"}),
    )
