"""CodeExplorer widget — tree-sitter-based code reading with LLM summarization."""

import html as html_mod
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path

import tree_sitter
import tree_sitter_python as tspython
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.tools import ToolDefinition

from calipso.cmd import Cmd, Initiator, effect, for_initiator, tool_result
from calipso.summarizer import create_summarizer_agent
from calipso.widget import WidgetHandle, create_widget

PY_LANGUAGE = tree_sitter.Language(tspython.language())

_DEFAULT_QUERY = "(function_definition) @fn (class_definition) @cls"

_COMMENT_QUERY = tree_sitter.Query(
    PY_LANGUAGE, "(comment) @c (expression_statement (string) @d)"
)


# --- Model ---


@dataclass(frozen=True)
class OpenFile:
    """A file opened in the explorer with its cached parse tree."""

    path: str
    source: bytes
    tree: tree_sitter.Tree


@dataclass(frozen=True)
class CodeExplorerModel:
    open_files: dict[str, OpenFile] = field(default_factory=dict)
    query_results: dict[str, str] = field(default_factory=dict)


# --- Messages ---


@dataclass(frozen=True)
class FileOpened:
    path: str
    open_file: OpenFile
    query_result: str


@dataclass(frozen=True)
class FileOpenError:
    path: str
    error: str


@dataclass(frozen=True)
class FileClosed:
    path: str
    initiator: Initiator


@dataclass(frozen=True)
class QueryCompleted:
    results: dict[str, str]


@dataclass(frozen=True)
class QueryError:
    error: str


@dataclass(frozen=True)
class OpenFileRequested:
    path: str


@dataclass(frozen=True)
class QueryRequested:
    path: str
    query: str


@dataclass(frozen=True)
class QueryAllRequested:
    query: str


CodeExplorerMsg = (
    FileOpened
    | FileOpenError
    | FileClosed
    | QueryCompleted
    | QueryError
    | OpenFileRequested
    | QueryRequested
    | QueryAllRequested
)


# --- Update ---


def _create_update(parser: tree_sitter.Parser, summarizer: Agent):
    """Create the update closure with access to I/O resources.

    The update function itself remains pure — it constructs CmdEffect
    descriptions but does not execute them. The parser and summarizer
    are captured for use in effect thunks.
    """

    def update(
        model: CodeExplorerModel, msg: CodeExplorerMsg
    ) -> tuple[CodeExplorerModel, Cmd]:
        match msg:
            case OpenFileRequested(path=path):

                async def perform_open():
                    p = Path(path)
                    if not p.is_file():
                        return FileOpenError(path=path, error=f"File not found: {path}")
                    source = p.read_bytes()
                    tree = parser.parse(source)
                    of = OpenFile(path=path, source=source, tree=tree)
                    qr = await _run_query(of, _DEFAULT_QUERY, summarizer)
                    return FileOpened(path=path, open_file=of, query_result=qr)

                return model, effect(perform=perform_open, to_msg=lambda msg: msg)

            case QueryRequested(path=path, query=query_str):
                if path not in model.open_files:
                    return model, tool_result(f"File not open: {path}")
                of = model.open_files[path]

                async def perform_query(of=of, query_str=query_str):
                    result = await _run_query(of, query_str, summarizer)
                    return QueryCompleted(results={of.path: result})

                return model, effect(perform=perform_query, to_msg=lambda msg: msg)

            case QueryAllRequested(query=query_str):
                if not model.open_files:
                    return model, tool_result("No files open.")
                open_files = dict(model.open_files)

                async def perform_query_all():
                    results = {}
                    for path, of in open_files.items():
                        results[path] = await _run_query(of, query_str, summarizer)
                    return QueryCompleted(results=results)

                return model, effect(perform=perform_query_all, to_msg=lambda msg: msg)

            case FileOpened(path=path, open_file=of, query_result=qr):
                new_files = {**model.open_files, path: of}
                new_results = {**model.query_results, path: qr}
                return (
                    replace(model, open_files=new_files, query_results=new_results),
                    tool_result(f"{path}:\n{qr}"),
                )
            case FileOpenError(path=path, error=error):
                return model, tool_result(error)
            case FileClosed(path=path, initiator=init):
                if path not in model.open_files:
                    return model, for_initiator(init, f"File not open: {path}")
                new_files = {k: v for k, v in model.open_files.items() if k != path}
                new_results = {
                    k: v for k, v in model.query_results.items() if k != path
                }
                return (
                    replace(model, open_files=new_files, query_results=new_results),
                    for_initiator(init, f"Closed: {path}"),
                )
            case QueryCompleted(results=results):
                new_results = {**model.query_results, **results}
                text = "\n\n".join(f"{p}:\n{r}" for p, r in results.items())
                return replace(model, query_results=new_results), tool_result(text)
            case QueryError(error=error):
                return model, tool_result(error)

    return update


# --- Views ---

_TOOL_DEFS = [
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
        description=("Run a tree-sitter S-expression query against all open files."),
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


def view_messages(model: CodeExplorerModel) -> Iterator[ModelMessage]:
    if not model.open_files:
        text = "## Code Explorer\nNo files open."
    else:
        lines = [
            "## Code Explorer",
            "",
            "**All code bodies have been redacted.** "
            "Signatures are shown verbatim; bodies are replaced with "
            "`# [ CODE REDACTED ]` and a docstring describing what "
            "the code does. Original comments and docstrings "
            "have been removed.",
            "",
            f"{len(model.open_files)} file(s) open:",
            "",
        ]
        for path in model.open_files:
            lines.append(f"- `{path}`")
            if path in model.query_results:
                lines.append("```python")
                lines.append(model.query_results[path])
                lines.append("```")
        text = "\n".join(lines)
    yield ModelRequest(parts=[UserPromptPart(content=text)])


def view_tools(model: CodeExplorerModel) -> Iterator[ToolDefinition]:
    yield from _TOOL_DEFS


def view_html(model: CodeExplorerModel) -> str:
    if not model.open_files:
        content = "<p><em>No files open</em></p>"
    else:
        parts = []
        for path in model.open_files:
            escaped_path = html_mod.escape(path)
            filename = Path(path).name
            close_btn = (
                f"<button onclick=\"sendWidgetEvent('close_file', "
                f"{{path: '{html_mod.escape(path, quote=True)}'}})\""
                f' class="btn-remove" title="Close file">'
                f"&times;</button>"
            )
            parts.append(
                f'<div class="file-entry" title="{escaped_path}">'
                f"<code>{html_mod.escape(filename)}</code>"
                f"{close_btn}</div>"
            )
            if path in model.query_results:
                escaped_code = html_mod.escape(model.query_results[path])
                parts.append(f"<pre><code>{escaped_code}</code></pre>")
        content = "".join(parts)
    return (
        '<div id="widget-code-explorer" class="widget">'
        f"<h3>Code Explorer</h3>{content}</div>"
    )


# --- Anticorruption layers ---


def from_llm(model: CodeExplorerModel, tool_name: str, args: dict) -> CodeExplorerMsg:
    match tool_name:
        case "open_file":
            return OpenFileRequested(path=args["path"])
        case "close_file":
            return FileClosed(path=args["path"], initiator=Initiator.LLM)
        case "query":
            return QueryRequested(path=args["path"], query=args["query"])
        case "query_all":
            return QueryAllRequested(query=args["query"])
    raise ValueError(f"CodeExplorer: unknown tool '{tool_name}'")


def from_ui(
    model: CodeExplorerModel, event_name: str, args: dict
) -> CodeExplorerMsg | None:
    match event_name:
        case "close_file":
            return FileClosed(path=args["path"], initiator=Initiator.UI)
    return None


# --- I/O helpers ---


async def _run_query(of: OpenFile, query_str: str, summarizer: Agent) -> str:
    """Execute a tree-sitter query and summarize the results."""
    try:
        q = tree_sitter.Query(PY_LANGUAGE, query_str)
    except tree_sitter.QueryError as e:
        return f"Invalid query: {e}"
    cursor = tree_sitter.QueryCursor(q)
    captures = cursor.captures(of.tree.root_node)
    if not captures:
        return "(no matches)"
    nodes = []
    for node_list in captures.values():
        nodes.extend(node_list)
    nodes.sort(key=lambda n: n.start_byte)
    deduped = []
    for node in nodes:
        if deduped and node.start_byte < deduped[-1].end_byte:
            continue
        deduped.append(node)
    code_parts = [_strip_comments(node, of.source) for node in deduped]
    raw_code = "\n\n".join(code_parts)
    result = await summarizer.run(raw_code)
    return result.output


def _strip_comments(node: tree_sitter.Node, source: bytes) -> str:
    """Extract node text with all comments and docstrings removed."""
    start = node.start_byte
    end = node.end_byte
    node_source = source[start:end]
    cursor = tree_sitter.QueryCursor(_COMMENT_QUERY)
    captures = cursor.captures(node)
    removal_ranges = []
    for node_list in captures.values():
        for comment_node in node_list:
            r_start = comment_node.start_byte - start
            r_end = comment_node.end_byte - start
            if r_start >= 0 and r_end <= len(node_source):
                removal_ranges.append((r_start, r_end))
    removal_ranges.sort(reverse=True)
    result = bytearray(node_source)
    for r_start, r_end in removal_ranges:
        del result[r_start:r_end]
    lines = result.decode(errors="replace").splitlines()
    cleaned = [line for line in lines if line.strip()]
    return "\n".join(cleaned)


# --- Factory ---


def create_code_explorer() -> WidgetHandle:
    parser = tree_sitter.Parser(PY_LANGUAGE)
    summarizer = create_summarizer_agent()

    return create_widget(
        id="widget-code-explorer",
        model=CodeExplorerModel(),
        update=_create_update(parser, summarizer),
        view_messages=view_messages,
        view_tools=view_tools,
        view_html=view_html,
        from_llm=from_llm,
        from_ui=from_ui,
        frontend_tools=frozenset({"close_file"}),
    )
