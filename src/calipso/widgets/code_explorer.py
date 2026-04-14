"""CodeExplorer widget — tree-sitter-based code reading with LLM summarization."""

import html as html_mod
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import tree_sitter
import tree_sitter_python as tspython
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.tools import ToolDefinition

from calipso.summarizer import create_summarizer_agent
from calipso.widget import Widget, render_md

PY_LANGUAGE = tree_sitter.Language(tspython.language())

_DEFAULT_QUERY = "(function_definition) @fn (class_definition) @cls"

_COMMENT_QUERY = tree_sitter.Query(
    PY_LANGUAGE, "(comment) @c (expression_statement (string) @d)"
)


@dataclass
class OpenFile:
    """A file opened in the explorer with its cached parse tree."""

    path: str
    source: bytes
    tree: tree_sitter.Tree


@dataclass
class CodeExplorer(Widget):
    """Widget for reading code via tree-sitter queries with LLM summarization."""

    open_files: dict[str, OpenFile] = field(default_factory=dict)
    query_results: dict[str, str] = field(default_factory=dict)
    _parser: tree_sitter.Parser = field(init=False, repr=False)
    _summarizer: Agent = field(init=False, repr=False)
    _tool_defs: list[ToolDefinition] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._parser = tree_sitter.Parser(PY_LANGUAGE)
        self._summarizer = create_summarizer_agent()
        self._tool_defs = [
            ToolDefinition(
                name="open_file",
                description=(
                    "Open a Python file for exploration. "
                    "Parses the file and shows top-level structure "
                    "(function/class signatures with body summaries)."
                ),
                parameters_json_schema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the Python file.",
                        },
                    },
                    "required": ["path"],
                },
            ),
            ToolDefinition(
                name="close_file",
                description="Close a file and remove it from the explorer.",
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
            ToolDefinition(
                name="query",
                description=(
                    "Run a tree-sitter S-expression query against an open file. "
                    "Returns matched code with signatures preserved and bodies "
                    "replaced by summaries."
                ),
                parameters_json_schema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path of the open file to query.",
                        },
                        "query": {
                            "type": "string",
                            "description": "Tree-sitter S-expression query string.",
                        },
                    },
                    "required": ["path", "query"],
                },
            ),
            ToolDefinition(
                name="query_all",
                description=(
                    "Run a tree-sitter S-expression query against all open files."
                ),
                parameters_json_schema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Tree-sitter S-expression query string.",
                        },
                    },
                    "required": ["query"],
                },
            ),
        ]

    def view_messages(self) -> Iterator[ModelMessage]:
        if not self.open_files:
            text = "## Code Explorer\nNo files open."
        else:
            lines = [
                "## Code Explorer",
                "",
                "**All code bodies have been redacted.** "
                "Signatures are shown verbatim; bodies are replaced with "
                '`[...REDACTED...]` followed by a description. '
                "Comments and docstrings have been removed.",
                "",
                f"{len(self.open_files)} file(s) open:",
                "",
            ]
            for path in self.open_files:
                lines.append(f"- `{path}`")
                if path in self.query_results:
                    lines.append("```python")
                    lines.append(self.query_results[path])
                    lines.append("```")
            text = "\n".join(lines)
        yield ModelRequest(parts=[UserPromptPart(content=text)])

    def view_tools(self) -> Iterator[ToolDefinition]:
        yield from self._tool_defs

    def view_html(self) -> str:
        if not self.open_files:
            content = "<p><em>No files open</em></p>"
        else:
            parts = []
            for path in self.open_files:
                escaped_path = html_mod.escape(path)
                close_btn = (
                    f" <button onclick=\"sendWidgetEvent('close_file', "
                    f"{{path: '{html_mod.escape(path, quote=True)}'}})\""
                    f' class="btn-remove" title="Close file">x</button>'
                )
                parts.append(f"<div><code>{escaped_path}</code>{close_btn}</div>")
                if path in self.query_results:
                    rendered = render_md(f"```python\n{self.query_results[path]}\n```")
                    parts.append(rendered)
            content = "".join(parts)
        return (
            f'<div id="{self.widget_id()}" class="widget">'
            f"<h3>Code Explorer</h3>{content}</div>"
        )

    async def update(self, tool_name: str, args: dict) -> str:
        if tool_name == "open_file":
            return await self._open_file(args["path"])
        if tool_name == "close_file":
            return self._close_file(args["path"])
        if tool_name == "query":
            return await self._query(args["path"], args["query"])
        if tool_name == "query_all":
            return await self._query_all(args["query"])
        raise NotImplementedError(f"CodeExplorer does not handle tool '{tool_name}'")

    def frontend_tools(self) -> set[str]:
        return {"close_file"}

    async def _open_file(self, path: str) -> str:
        p = Path(path)
        if not p.is_file():
            return f"File not found: {path}"
        source = p.read_bytes()
        tree = self._parser.parse(source)
        self.open_files[path] = OpenFile(path=path, source=source, tree=tree)
        return await self._query(path, _DEFAULT_QUERY)

    def _close_file(self, path: str) -> str:
        if path not in self.open_files:
            return f"File not open: {path}"
        del self.open_files[path]
        self.query_results.pop(path, None)
        return f"Closed: {path}"

    async def _query(self, path: str, query_str: str) -> str:
        if path not in self.open_files:
            return f"File not open: {path}"
        of = self.open_files[path]
        try:
            q = tree_sitter.Query(PY_LANGUAGE, query_str)
        except tree_sitter.QueryError as e:
            return f"Invalid query: {e}"
        cursor = tree_sitter.QueryCursor(q)
        captures = cursor.captures(of.tree.root_node)
        if not captures:
            self.query_results[path] = "(no matches)"
            return f"{path}: no matches"
        nodes = []
        for node_list in captures.values():
            nodes.extend(node_list)
        nodes.sort(key=lambda n: n.start_byte)
        # Deduplicate overlapping nodes (keep the outermost)
        deduped = []
        for node in nodes:
            if deduped and node.start_byte < deduped[-1].end_byte:
                continue
            deduped.append(node)
        code_parts = []
        for node in deduped:
            text = _strip_comments(node, of.source)
            code_parts.append(text)
        raw_code = "\n\n".join(code_parts)
        result = await self._summarizer.run(raw_code)
        self.query_results[path] = result.output
        return f"{path}:\n{result.output}"

    async def _query_all(self, query_str: str) -> str:
        if not self.open_files:
            return "No files open."
        results = []
        for path in list(self.open_files):
            result = await self._query(path, query_str)
            results.append(result)
        return "\n\n".join(results)


def _strip_comments(node: tree_sitter.Node, source: bytes) -> str:
    """Extract node text with all comments and docstrings removed."""
    start = node.start_byte
    end = node.end_byte
    node_source = source[start:end]
    # Find comments/docstrings within this node's range
    cursor = tree_sitter.QueryCursor(_COMMENT_QUERY)
    captures = cursor.captures(node)
    removal_ranges = []
    for node_list in captures.values():
        for comment_node in node_list:
            r_start = comment_node.start_byte - start
            r_end = comment_node.end_byte - start
            if r_start >= 0 and r_end <= len(node_source):
                removal_ranges.append((r_start, r_end))
    # Sort descending so removals don't invalidate offsets
    removal_ranges.sort(reverse=True)
    result = bytearray(node_source)
    for r_start, r_end in removal_ranges:
        del result[r_start:r_end]
    # Clean up blank lines left behind
    lines = result.decode(errors="replace").splitlines()
    cleaned = [line for line in lines if line.strip()]
    return "\n".join(cleaned)
