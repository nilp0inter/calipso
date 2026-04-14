"""Tests for the CodeExplorer widget."""

from unittest.mock import AsyncMock

import pytest
from pydantic_ai import models

from calipso.widgets.code_explorer import (
    CodeExplorer,
    _extract_signatures,
    _strip_comments,
)

models.ALLOW_MODEL_REQUESTS = False
pytestmark = pytest.mark.anyio

_SAMPLE_PYTHON = b"""\
# This is a comment
import os

def greet(name: str) -> str:
    \"\"\"Say hello.\"\"\"
    # build greeting
    return f"Hello, {name}!"

class Calculator:
    \"\"\"A simple calculator.\"\"\"

    def add(self, a: int, b: int) -> int:
        \"\"\"Add two numbers.\"\"\"
        return a + b

    def multiply(self, a: int, b: int) -> int:
        # multiply them
        return a * b
"""


_SUMMARIZER_OUTPUT = """\
def greet(name: str) -> str:
    [...REDACTED...] # builds and returns a greeting string

class Calculator:
    [...REDACTED...] # a simple calculator

    def add(self, a: int, b: int) -> int:
        [...REDACTED...] # returns the sum of a and b

    def multiply(self, a: int, b: int) -> int:
        [...REDACTED...] # returns the product of a and b
"""

_QUERY_OUTPUT = """\
def greet(name: str) -> str:
    [...REDACTED...] # builds and returns a greeting string
"""


def _mock_summarizer_run(output: str = _SUMMARIZER_OUTPUT):
    """Create a mock that replaces _summarizer.run() with a canned result."""
    mock_result = AsyncMock()
    mock_result.output = output
    return AsyncMock(return_value=mock_result)


@pytest.fixture
def explorer():
    """Create a CodeExplorer with a mocked summarizer."""
    w = CodeExplorer()
    w._summarizer.run = _mock_summarizer_run()
    return w


@pytest.fixture
def sample_file(tmp_path):
    """Write a sample Python file and return its path string."""
    p = tmp_path / "sample.py"
    p.write_bytes(_SAMPLE_PYTHON)
    return str(p)


# --- open_file ---


class TestOpenFile:
    async def test_open_file(self, explorer, sample_file):
        result = await explorer.update("open_file", {"path": sample_file})
        assert sample_file in explorer.open_files
        assert sample_file in explorer.query_results
        assert "def greet" in result
        assert "[...REDACTED...]" in result

    async def test_open_nonexistent_file(self, explorer):
        result = await explorer.update("open_file", {"path": "/nonexistent.py"})
        assert "not found" in result.lower()
        assert len(explorer.open_files) == 0


# --- close_file ---


class TestCloseFile:
    async def test_close_file(self, explorer, sample_file):
        await explorer.update("open_file", {"path": sample_file})
        result = await explorer.update("close_file", {"path": sample_file})
        assert "Closed" in result
        assert sample_file not in explorer.open_files
        assert sample_file not in explorer.query_results

    async def test_close_unknown_file(self, explorer):
        result = await explorer.update("close_file", {"path": "/nope.py"})
        assert "not open" in result.lower()


# --- query ---


class TestQuery:
    async def test_query_valid(self, explorer, sample_file):
        await explorer.update("open_file", {"path": sample_file})
        explorer._summarizer.run = _mock_summarizer_run(_QUERY_OUTPUT)
        result = await explorer.update(
            "query", {"path": sample_file, "query": "(function_definition) @fn"}
        )
        assert "def greet" in result
        assert "[...REDACTED...]" in result
        assert "[...REDACTED...]" in explorer.query_results[sample_file]

    async def test_query_invalid_syntax(self, explorer, sample_file):
        await explorer.update("open_file", {"path": sample_file})
        result = await explorer.update(
            "query", {"path": sample_file, "query": "(invalid_node_type @x"}
        )
        assert "Invalid query" in result

    async def test_query_unknown_file(self, explorer):
        result = await explorer.update(
            "query", {"path": "/nope.py", "query": "(function_definition) @fn"}
        )
        assert "not open" in result.lower()

    async def test_query_no_matches(self, explorer, tmp_path):
        p = tmp_path / "empty.py"
        p.write_bytes(b"x = 1\n")
        path = str(p)
        await explorer.update("open_file", {"path": path})
        explorer._summarizer.run = _mock_summarizer_run("should not be called")
        result = await explorer.update(
            "query", {"path": path, "query": "(class_definition) @cls"}
        )
        assert "no matches" in result.lower()


# --- query_all ---


class TestQueryAll:
    async def test_query_all(self, explorer, tmp_path):
        f1 = tmp_path / "a.py"
        f1.write_bytes(b"def foo(): pass\n")
        f2 = tmp_path / "b.py"
        f2.write_bytes(b"def bar(): pass\n")
        await explorer.update("open_file", {"path": str(f1)})
        await explorer.update("open_file", {"path": str(f2)})
        explorer._summarizer.run = _mock_summarizer_run(_QUERY_OUTPUT)
        result = await explorer.update(
            "query_all", {"query": "(function_definition) @fn"}
        )
        assert str(f1) in result
        assert str(f2) in result

    async def test_query_all_no_files(self, explorer):
        result = await explorer.update(
            "query_all", {"query": "(function_definition) @fn"}
        )
        assert "No files open" in result


# --- comment stripping ---


class TestCommentStripping:
    async def test_comments_stripped(self, explorer, sample_file):
        """Verify # comments are removed before the summarizer sees the code."""
        await explorer.update("open_file", {"path": sample_file})
        call_args = explorer._summarizer.run.call_args
        code_sent = call_args[0][0]
        assert "# This is a comment" not in code_sent
        assert "# build greeting" not in code_sent
        assert "# multiply them" not in code_sent

    async def test_docstrings_stripped(self, explorer, sample_file):
        """Verify docstrings are removed before the summarizer sees the code."""
        await explorer.update("open_file", {"path": sample_file})
        call_args = explorer._summarizer.run.call_args
        code_sent = call_args[0][0]
        assert '"""Say hello."""' not in code_sent
        assert '"""A simple calculator."""' not in code_sent
        assert '"""Add two numbers."""' not in code_sent

    def test_strip_comments_function(self):
        """Test the _strip_comments helper directly."""
        import tree_sitter
        import tree_sitter_python as tspython

        lang = tree_sitter.Language(tspython.language())
        parser = tree_sitter.Parser(lang)
        code = b'def foo():\n    # comment\n    """docstring"""\n    return 1\n'
        tree = parser.parse(code)
        node = tree.root_node.children[0]  # function_definition
        result = _strip_comments(node, code)
        assert "# comment" not in result
        assert '"""docstring"""' not in result
        assert "def foo():" in result
        assert "return 1" in result


# --- signature extraction ---


class TestExtractSignatures:
    def test_strips_leaked_code(self):
        """Drops everything except signatures and comments."""
        import tree_sitter
        import tree_sitter_python as tspython

        lang = tree_sitter.Language(tspython.language())
        parser = tree_sitter.Parser(lang)
        # Simulate a summarizer that leaked code
        llm_output = (
            b"import os\n\n"
            b"def foo(x: int) -> str:\n"
            b"    [...REDACTED...] # does something\n"
            b"    leaked = True\n\n"
            b"class Bar:\n"
            b"    [...REDACTED...] # a class\n"
            b"    x = 42\n\n"
            b"    def method(self) -> None:\n"
            b"        [...REDACTED...] # a method\n"
            b"        return leaked\n\n"
            b"y = 99\n"
        )
        result = _extract_signatures(llm_output, parser)
        # Signatures and comments kept
        assert "def foo(x: int) -> str:" in result
        assert "[...REDACTED...] # does something" in result
        assert "class Bar:" in result
        assert "[...REDACTED...] # a class" in result
        assert "def method(self) -> None:" in result
        assert "[...REDACTED...] # a method" in result
        # Leaked code stripped
        assert "import os" not in result
        assert "leaked = True" not in result
        assert "x = 42" not in result
        assert "return leaked" not in result
        assert "y = 99" not in result

    def test_handles_decorated_functions(self):
        import tree_sitter
        import tree_sitter_python as tspython

        lang = tree_sitter.Language(tspython.language())
        parser = tree_sitter.Parser(lang)
        llm_output = (
            b"@staticmethod\ndef helper() -> None:\n    [...REDACTED...] # helps\n"
        )
        result = _extract_signatures(llm_output, parser)
        assert "@staticmethod" in result
        assert "def helper() -> None:" in result
        assert "[...REDACTED...] # helps" in result

    def test_no_comment_still_redacted(self):
        """When the summarizer omits the comment, [...REDACTED...] is still added."""
        import tree_sitter
        import tree_sitter_python as tspython

        lang = tree_sitter.Language(tspython.language())
        parser = tree_sitter.Parser(lang)
        llm_output = b"def bare() -> None:\n    pass\n"
        result = _extract_signatures(llm_output, parser)
        assert "def bare() -> None:" in result
        assert "[...REDACTED...]" in result


# --- view_messages ---


class TestViewMessages:
    def test_view_messages_no_files(self):
        w = CodeExplorer()
        msgs = list(w.view_messages())
        assert len(msgs) == 1
        assert "No files open" in msgs[0].parts[0].content

    async def test_view_messages_with_files(self, explorer, sample_file):
        await explorer.update("open_file", {"path": sample_file})
        msgs = list(explorer.view_messages())
        content = msgs[0].parts[0].content
        assert sample_file in content
        assert "def greet" in content
        assert "REDACTED" in content


# --- view_html ---


class TestViewHtml:
    def test_view_html_empty(self):
        w = CodeExplorer()
        html = w.view_html()
        assert 'id="widget-code-explorer"' in html
        assert "No files open" in html

    async def test_view_html_with_files(self, explorer, sample_file):
        await explorer.update("open_file", {"path": sample_file})
        html = explorer.view_html()
        assert sample_file in html
        assert "close_file" in html


# --- frontend_tools ---


class TestFrontendTools:
    def test_frontend_tools(self):
        w = CodeExplorer()
        assert w.frontend_tools() == {"close_file"}


# --- widget_id ---


class TestWidgetId:
    def test_widget_id(self):
        w = CodeExplorer()
        assert w.widget_id() == "widget-code-explorer"
