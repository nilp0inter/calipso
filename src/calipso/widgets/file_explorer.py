"""FileExplorer widget — filesystem navigation and file reading."""

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
class OpenDirectory:
    path: str
    entries: tuple[tuple[str, bool], ...]
    listing_text: str


@dataclass(frozen=True)
class FileExplorerModel:
    open_directories: tuple[OpenDirectory, ...] = ()
    open_files: tuple[tuple[str, str], ...] = ()


# --- Messages ---


@dataclass(frozen=True)
class DirectoryOpened:
    path: str
    entries: tuple[tuple[str, bool], ...]
    listing_text: str


@dataclass(frozen=True)
class DirectoryOpenError:
    error: str


@dataclass(frozen=True)
class FileRead:
    path: str
    content: str


@dataclass(frozen=True)
class FileReadError:
    error: str


@dataclass(frozen=True)
class CloseDirectory:
    path: str
    initiator: Initiator


@dataclass(frozen=True)
class CloseReadFile:
    path: str
    initiator: Initiator


@dataclass(frozen=True)
class OpenDirectoryRequested:
    path: str


@dataclass(frozen=True)
class ReadFileRequested:
    path: str


FileExplorerMsg = (
    DirectoryOpened
    | DirectoryOpenError
    | FileRead
    | FileReadError
    | CloseDirectory
    | CloseReadFile
    | OpenDirectoryRequested
    | ReadFileRequested
)


# --- Update (pure) ---


def update(
    model: FileExplorerModel, msg: FileExplorerMsg
) -> tuple[FileExplorerModel, Cmd]:
    match msg:
        case OpenDirectoryRequested(path=path):

            async def perform():
                return _do_open_directory(path)

            return model, effect(perform=perform, to_msg=lambda msg: msg)
        case ReadFileRequested(path=path):

            async def perform():
                return _do_read_file(path)

            return model, effect(perform=perform, to_msg=lambda msg: msg)
        case DirectoryOpened(path=path, entries=entries, listing_text=text):
            existing = tuple(d for d in model.open_directories if d.path != path)
            return (
                replace(
                    model,
                    open_directories=existing
                    + (OpenDirectory(path=path, entries=entries, listing_text=text),),
                ),
                tool_result(f"Opened directory: {path}"),
            )
        case DirectoryOpenError(error=error):
            return model, tool_result(error)
        case FileRead(path=path, content=content):
            existing = tuple((p, c) for p, c in model.open_files if p != path)
            return (
                replace(
                    model,
                    open_files=existing + ((path, content),),
                ),
                tool_result(f"Opened: {path}"),
            )
        case FileReadError(error=error):
            return model, tool_result(error)
        case CloseDirectory(path=path, initiator=init):
            if not any(d.path == path for d in model.open_directories):
                return model, for_initiator(init, "No directory is open.")
            return (
                replace(
                    model,
                    open_directories=tuple(
                        d for d in model.open_directories if d.path != path
                    ),
                ),
                for_initiator(init, f"Closed directory: {path}"),
            )
        case CloseReadFile(path=path, initiator=init):
            if not any(p == path for p, _ in model.open_files):
                return model, for_initiator(init, "No file is open.")
            return (
                replace(
                    model,
                    open_files=tuple((p, c) for p, c in model.open_files if p != path),
                ),
                for_initiator(init, f"Closed: {path}"),
            )


# --- Views ---

_TOOL_DEFS = [
    ToolDefinition(
        name="open_directory",
        description=(
            "Open a directory in the File Explorer, listing its files and "
            "subdirectories. Directories are marked with a trailing /. "
            "The listing persists until closed with close_directory. "
            "Re-opening the same path refreshes the listing."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to open. Defaults to '.'.",
                    "default": ".",
                },
            },
        },
    ),
    ToolDefinition(
        name="close_directory",
        description="Close an open directory in the File Explorer.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path of the directory to close.",
                },
            },
            "required": ["path"],
        },
    ),
    ToolDefinition(
        name="read_file",
        description="Read a file's contents.",
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
    for directory in model.open_directories:
        lines.append("")
        lines.append(directory.listing_text)
    for path, content in model.open_files:
        lines.append("")
        lines.append(f"**Open file:** `{path}`")
        lines.append("```")
        lines.append(content)
        lines.append("```")
    if not model.open_directories and not model.open_files:
        lines.append("No directory or file open.")
    yield ModelRequest(parts=[UserPromptPart(content="\n".join(lines))])


def view_tools(model: FileExplorerModel) -> Iterator[ToolDefinition]:
    yield from _TOOL_DEFS


def view_html(model: FileExplorerModel) -> str:
    parts = []

    root_btn = (
        "<button onclick=\"sendWidgetEvent('open_directory', {path: '.'})\""
        ' class="btn-root">Root</button>'
    )
    parts.append(root_btn)

    for directory in model.open_directories:
        escaped_dir_path = html_mod.escape(directory.path, quote=True)
        close_btn = (
            "<button onclick=\"sendWidgetEvent('close_directory',"
            f" {{path: '{escaped_dir_path}'}})\""
            ' class="btn-remove" title="Close directory">'
            "&times;</button>"
        )
        header = html_mod.escape(directory.path)
        parts.append(
            f'<div class="file-entry" title="{escaped_dir_path}">'
            f"<code>{header}</code>"
            f"{close_btn}</div>"
        )
        items = []
        for name, is_dir in directory.entries:
            escaped_name = html_mod.escape(name)
            child_path = f"{directory.path}/{name}"
            escaped_path = html_mod.escape(child_path, quote=True)
            if is_dir:
                items.append(
                    f'<li class="entry-dir" ondblclick="'
                    f"sendWidgetEvent('open_directory', "
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
        parts.append(f'<ul class="dir-listing">{"".join(items)}</ul>')

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
        case "open_directory":
            return OpenDirectoryRequested(path=args.get("path", "."))
        case "close_directory":
            return CloseDirectory(path=args["path"], initiator=Initiator.LLM)
        case "read_file":
            return ReadFileRequested(path=args["path"])
        case "close_read_file":
            return CloseReadFile(path=args["path"], initiator=Initiator.LLM)
    raise ValueError(f"FileExplorer: unknown tool '{tool_name}'")


def from_ui(
    model: FileExplorerModel, event_name: str, args: dict
) -> FileExplorerMsg | None:
    match event_name:
        case "open_directory":
            return OpenDirectoryRequested(path=args.get("path", "."))
        case "close_directory":
            return CloseDirectory(path=args["path"], initiator=Initiator.UI)
        case "read_file":
            return ReadFileRequested(path=args["path"])
        case "close_read_file":
            return CloseReadFile(path=args["path"], initiator=Initiator.UI)
    return None


# --- I/O helpers ---


def _do_open_directory(path: str) -> FileExplorerMsg:
    p = Path(path)
    if not p.is_dir():
        return DirectoryOpenError(error=f"Not a directory: {path}")
    entries_raw = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name))
    entries = tuple((e.name, e.is_dir()) for e in entries_raw)
    lines = [f"Contents of `{path}`:", ""]
    for entry in entries_raw:
        suffix = "/" if entry.is_dir() else ""
        lines.append(f"- {entry.name}{suffix}")
    if not entries_raw:
        lines.append("(empty directory)")
    listing_text = "\n".join(lines)
    return DirectoryOpened(path=path, entries=entries, listing_text=listing_text)


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
        frontend_tools=frozenset(
            {"open_directory", "close_directory", "read_file", "close_read_file"}
        ),
    )
