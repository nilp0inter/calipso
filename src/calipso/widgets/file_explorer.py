"""FileExplorer widget — filesystem navigation and non-Python file reading."""

import html as html_mod
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.tools import ToolDefinition

from calipso.widget import Widget


@dataclass
class FileExplorer(Widget):
    """Widget for listing directories and reading non-Python files."""

    current_listing: str | None = None
    open_file_path: str | None = None
    open_file_content: str | None = None
    _listing_path: str | None = field(init=False, repr=False, default=None)
    _listing_entries: list[tuple[str, bool]] | None = field(
        init=False, repr=False, default=None
    )
    _tool_defs: list[ToolDefinition] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._tool_defs = [
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

    def view_messages(self) -> Iterator[ModelMessage]:
        lines = ["## File Explorer"]
        if self.current_listing is not None:
            lines.append("")
            lines.append(self.current_listing)
        if self.open_file_path is not None:
            lines.append("")
            lines.append(f"**Open file:** `{self.open_file_path}`")
            lines.append("```")
            lines.append(self.open_file_content or "")
            lines.append("```")
        if self.current_listing is None and self.open_file_path is None:
            lines.append("No file open.")
        yield ModelRequest(parts=[UserPromptPart(content="\n".join(lines))])

    def view_tools(self) -> Iterator[ToolDefinition]:
        yield from self._tool_defs

    def view_html(self) -> str:
        parts = []

        # Directory listing
        if self._listing_entries is not None:
            items = []
            for name, is_dir in self._listing_entries:
                escaped = html_mod.escape(name)
                cls = "entry-dir" if is_dir else "entry-file"
                suffix = "/" if is_dir else ""
                items.append(f'<li class="{cls}">{escaped}{suffix}</li>')
            header = html_mod.escape(self._listing_path or ".")
            parts.append(
                f"<p><strong>{header}</strong></p>"
                f'<ul class="dir-listing">{"".join(items)}</ul>'
            )

        # Open file
        if self.open_file_path is not None:
            filename = Path(self.open_file_path).name
            escaped_path = html_mod.escape(self.open_file_path)
            close_btn = (
                "<button onclick=\"sendWidgetEvent('close_read_file', {})\""
                ' class="btn-remove" title="Close file">'
                "&times;</button>"
            )
            escaped_content = html_mod.escape(self.open_file_content or "")
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
            f'<div id="{self.widget_id()}" class="widget">'
            f"<h3>File Explorer</h3>{content}</div>"
        )

    async def update(self, tool_name: str, args: dict) -> str:
        if tool_name == "list_directory":
            return self._list_directory(args.get("path", "."))
        if tool_name == "read_file":
            return self._read_file(args["path"])
        if tool_name == "close_read_file":
            return self._close_read_file()
        raise NotImplementedError(f"FileExplorer does not handle tool '{tool_name}'")

    def frontend_tools(self) -> set[str]:
        return {"close_read_file"}

    def _list_directory(self, path: str) -> str:
        p = Path(path)
        if not p.is_dir():
            return f"Not a directory: {path}"
        entries = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name))
        self._listing_path = path
        self._listing_entries = [(e.name, e.is_dir()) for e in entries]
        lines = [f"Contents of `{path}`:", ""]
        for entry in entries:
            suffix = "/" if entry.is_dir() else ""
            lines.append(f"- {entry.name}{suffix}")
        if not entries:
            lines.append("(empty directory)")
        listing = "\n".join(lines)
        self.current_listing = listing
        return listing

    def _read_file(self, path: str) -> str:
        if path.endswith(".py"):
            return (
                "Python files should be read with the Code Explorer's "
                "open_file tool, not read_file."
            )
        p = Path(path)
        if not p.is_file():
            return f"File not found: {path}"
        try:
            content = p.read_text()
        except UnicodeDecodeError:
            return f"Cannot read binary file: {path}"
        self.open_file_path = path
        self.open_file_content = content
        return content

    def _close_read_file(self) -> str:
        if self.open_file_path is None:
            return "No file is open."
        path = self.open_file_path
        self.open_file_path = None
        self.open_file_content = None
        return f"Closed: {path}"
