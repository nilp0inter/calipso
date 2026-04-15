"""TestSuite widget — configure, run, cancel, and view test results."""

import asyncio
import html as html_mod
import os
from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import Any

from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.tools import ToolDefinition

from calipso.cmd import Cmd, Initiator, effect, for_initiator, none, tool_result
from calipso.widget import WidgetHandle, create_widget

# --- Model ---


@dataclass(frozen=True)
class TestSuiteModel:
    command: str | None = None
    env_vars: tuple[tuple[str, str], ...] = ()
    timeout: int = 30
    status: str = (
        "idle"  # idle | running | passed | failed | cancelled | timeout | error
    )
    stdout: str = ""
    stderr: str = ""
    # TODO: set automatically via future elm subscriptions
    stale: bool = False


# --- Messages ---


@dataclass(frozen=True)
class ConfigureTestRunner:
    command: str
    env_vars: tuple[tuple[str, str], ...]
    timeout: int
    initiator: Initiator


@dataclass(frozen=True)
class RunTestsRequested:
    initiator: Initiator


@dataclass(frozen=True)
class CancelTestsRequested:
    initiator: Initiator


@dataclass(frozen=True)
class TestsCompleted:
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class TestsCancelled:
    stdout: str
    stderr: str


@dataclass(frozen=True)
class TestsTimedOut:
    command: str
    timeout: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class TestsError:
    error: str
    stdout: str
    stderr: str


TestSuiteMsg = (
    ConfigureTestRunner
    | RunTestsRequested
    | CancelTestsRequested
    | TestsCompleted
    | TestsCancelled
    | TestsTimedOut
    | TestsError
)


# --- Update (closure pattern for subprocess tracking) ---


def _create_update():
    _proc_ref: dict[str, Any] = {"proc": None, "cancelled": False}

    def update(model: TestSuiteModel, msg: TestSuiteMsg) -> tuple[TestSuiteModel, Cmd]:
        match msg:
            case ConfigureTestRunner(
                command=command,
                env_vars=env_vars,
                timeout=timeout,
                initiator=initiator,
            ):
                return (
                    replace(
                        model,
                        command=command,
                        env_vars=env_vars,
                        timeout=timeout,
                    ),
                    for_initiator(initiator, f"Test runner configured: {command}"),
                )

            case RunTestsRequested(initiator=initiator):
                if model.status == "running":
                    return model, for_initiator(initiator, "Tests are already running.")
                if model.command is None:
                    return model, for_initiator(
                        initiator,
                        "No test command configured. Use configure_test_runner first.",
                    )

                command = model.command
                env_dict = dict(model.env_vars)
                timeout = model.timeout
                _proc_ref["cancelled"] = False

                async def perform_run():
                    env = {**os.environ, **env_dict}
                    try:
                        proc = await asyncio.create_subprocess_shell(
                            command,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                            env=env,
                        )
                        _proc_ref["proc"] = proc
                        try:
                            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                                proc.communicate(), timeout=timeout
                            )
                        except asyncio.TimeoutError:
                            proc.kill()
                            stdout_bytes, stderr_bytes = await proc.communicate()
                            return TestsTimedOut(
                                command=command,
                                timeout=timeout,
                                stdout=stdout_bytes.decode(errors="replace"),
                                stderr=stderr_bytes.decode(errors="replace"),
                            )
                        finally:
                            _proc_ref["proc"] = None

                        stdout = stdout_bytes.decode(errors="replace")
                        stderr = stderr_bytes.decode(errors="replace")

                        if _proc_ref["cancelled"]:
                            return TestsCancelled(stdout=stdout, stderr=stderr)

                        return TestsCompleted(
                            returncode=proc.returncode,
                            stdout=stdout,
                            stderr=stderr,
                        )
                    except Exception as e:
                        _proc_ref["proc"] = None
                        return TestsError(error=str(e), stdout="", stderr="")

                return (
                    replace(model, status="running", stdout="", stderr=""),
                    effect(perform=perform_run, to_msg=lambda m: m),
                )

            case CancelTestsRequested(initiator=initiator):
                if model.status != "running":
                    return model, for_initiator(
                        initiator, "No tests are currently running."
                    )

                async def perform_cancel():
                    proc = _proc_ref.get("proc")
                    if proc is not None:
                        _proc_ref["cancelled"] = True
                        proc.terminate()
                    return None

                return model, effect(
                    perform=perform_cancel,
                    to_msg=lambda _: None,
                )

            case TestsCompleted(returncode=rc, stdout=stdout, stderr=stderr):
                status = "passed" if rc == 0 else "failed"
                return (
                    replace(
                        model,
                        status=status,
                        stdout=stdout,
                        stderr=stderr,
                        stale=False,
                    ),
                    tool_result(f"Tests {status}. Return code: {rc}"),
                )

            case TestsCancelled(stdout=stdout, stderr=stderr):
                return (
                    replace(
                        model,
                        status="cancelled",
                        stdout=stdout,
                        stderr=stderr,
                    ),
                    tool_result("Tests cancelled."),
                )

            case TestsTimedOut(command=cmd, timeout=tout, stdout=stdout, stderr=stderr):
                parts = [f"Command `{cmd}` timed out after {tout}s."]
                if stdout:
                    parts.append(f"\nstdout:\n{stdout}")
                if stderr:
                    parts.append(f"\nstderr:\n{stderr}")
                return (
                    replace(
                        model,
                        status="timeout",
                        stdout=stdout,
                        stderr=stderr,
                    ),
                    tool_result("\n".join(parts)),
                )

            case TestsError(error=error, stdout=stdout, stderr=stderr):
                parts = [f"Test error: {error}"]
                if stdout:
                    parts.append(f"\nstdout:\n{stdout}")
                if stderr:
                    parts.append(f"\nstderr:\n{stderr}")
                return (
                    replace(
                        model,
                        status="error",
                        stdout=stdout,
                        stderr=stderr,
                    ),
                    tool_result("\n".join(parts)),
                )

        return model, none

    return update


# --- Views ---

_TOOL_DEFS = [
    ToolDefinition(
        name="configure_test_runner",
        description=(
            "Set or update the test runner configuration. "
            "Can be called mid-conversation to change settings."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": (
                        "Shell command to run the test suite "
                        "(e.g., 'task tests:unit', 'uv run pytest -v')."
                    ),
                },
                "env_vars": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                    "description": "Environment variables to set.",
                    "default": {},
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds.",
                },
            },
            "required": ["command", "timeout"],
        },
    ),
    ToolDefinition(
        name="run_tests",
        description=(
            "Run the configured test suite. "
            "Rejects if tests are already running or no command is configured."
        ),
        parameters_json_schema={"type": "object", "properties": {}},
    ),
    ToolDefinition(
        name="cancel_tests",
        description=(
            "Cancel the currently running test suite. Only works if tests are running."
        ),
        parameters_json_schema={"type": "object", "properties": {}},
    ),
]


def view_messages(model: TestSuiteModel) -> Iterator[ModelMessage]:
    lines = ["## Test Suite"]

    if model.command is None:
        lines.append("No test runner configured.")
    else:
        lines.append(f"**Command:** `{model.command}`")
        if model.env_vars:
            env_str = ", ".join(f"{k}={v}" for k, v in model.env_vars)
            lines.append(f"**Env:** {env_str}")
        lines.append(f"**Timeout:** {model.timeout}s")

    stale_marker = " (STALE)" if model.stale else ""
    lines.append(f"**Status:** {model.status}{stale_marker}")

    if model.stdout:
        lines.append("\n**stdout:**")
        lines.append(f"```\n{model.stdout}\n```")
    if model.stderr:
        lines.append("\n**stderr:**")
        lines.append(f"```\n{model.stderr}\n```")

    yield ModelRequest(parts=[UserPromptPart(content="\n".join(lines))])


def view_tools(model: TestSuiteModel) -> Iterator[ToolDefinition]:
    yield from _TOOL_DEFS


def view_html(model: TestSuiteModel) -> str:
    badge_colors = {
        "idle": "#6b7280",
        "running": "#2563eb",
        "passed": "#16a34a",
        "failed": "#dc2626",
        "cancelled": "#d97706",
        "timeout": "#d97706",
        "error": "#dc2626",
    }
    color = badge_colors.get(model.status, "#6b7280")
    stale_html = ' <span style="color:#d97706">(stale)</span>' if model.stale else ""
    status_html = (
        f'<span style="color:{color};font-weight:600">'
        f"{html_mod.escape(model.status)}</span>{stale_html}"
    )

    cmd_val = html_mod.escape(model.command or "", quote=True)
    env_val = html_mod.escape(
        ", ".join(f"{k}={v}" for k, v in model.env_vars), quote=True
    )
    timeout_val = model.timeout

    config_form = (
        '<div style="margin-bottom:0.5rem">'
        '<input type="text" id="ts-cmd" value="'
        + cmd_val
        + '" placeholder="Test command..." '
        'style="width:100%;margin-bottom:0.3rem;padding:0.3rem;'
        'border:1px solid #ccc;border-radius:4px;font-size:0.85rem">'
        '<input type="text" id="ts-env" value="'
        + env_val
        + '" placeholder="KEY=val, ..." '
        'style="width:100%;margin-bottom:0.3rem;padding:0.3rem;'
        'border:1px solid #ccc;border-radius:4px;font-size:0.85rem">'
        '<input type="number" id="ts-timeout" value="'
        + str(timeout_val)
        + '" placeholder="Timeout (s)" '
        'style="width:80px;padding:0.3rem;'
        'border:1px solid #ccc;border-radius:4px;font-size:0.85rem">'
        ' <button onclick="'
        "var env={};"
        " document.getElementById('ts-env').value.split(',')"
        ".forEach(function(p){"
        "var kv=p.trim().split('=');"
        "if(kv[0]&&kv[0].trim())env[kv[0].trim()]=kv[1]||''});"
        " sendWidgetEvent('configure_test_runner', {"
        "command: document.getElementById('ts-cmd').value,"
        " env_vars: env,"
        " timeout: parseInt(document.getElementById('ts-timeout').value)||30"
        '})">'
        "Save</button>"
        "</div>"
    )

    if model.status == "running":
        buttons = (
            "<button onclick=\"sendWidgetEvent('cancel_tests', {})\">Cancel</button>"
        )
    else:
        buttons = (
            "<button onclick=\"sendWidgetEvent('run_tests', {})\">Run Tests</button>"
        )

    stdout_html = (
        '<pre style="max-height:300px;overflow-y:auto;background:#f8f8f8;'
        'padding:0.5rem;border-radius:4px;font-size:0.8rem"><code>'
        + html_mod.escape(model.stdout)
        + "</code></pre>"
        if model.stdout
        else ""
    )
    stderr_html = (
        '<pre style="max-height:200px;overflow-y:auto;background:#fff5f5;'
        "border-left:3px solid #dc2626;"
        'padding:0.5rem;border-radius:4px;font-size:0.8rem"><code>'
        + html_mod.escape(model.stderr)
        + "</code></pre>"
        if model.stderr
        else ""
    )

    return (
        '<div id="widget-test-suite" class="widget">'
        "<h3>Test Suite</h3>"
        f"<p>Status: {status_html}</p>"
        f"{config_form}"
        f"{buttons}"
        f"{stdout_html}"
        f"{stderr_html}"
        "</div>"
    )


# --- Anticorruption layers ---


def _env_dict_to_tuple(env: dict) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(env.items()))


def from_llm(model: TestSuiteModel, tool_name: str, args: dict) -> TestSuiteMsg:
    match tool_name:
        case "configure_test_runner":
            return ConfigureTestRunner(
                command=args["command"],
                env_vars=_env_dict_to_tuple(args.get("env_vars", {})),
                timeout=args["timeout"],
                initiator=Initiator.LLM,
            )
        case "run_tests":
            return RunTestsRequested(initiator=Initiator.LLM)
        case "cancel_tests":
            return CancelTestsRequested(initiator=Initiator.LLM)
    raise ValueError(f"TestSuite: unknown tool '{tool_name}'")


def from_ui(model: TestSuiteModel, event_name: str, args: dict) -> TestSuiteMsg | None:
    match event_name:
        case "configure_test_runner":
            return ConfigureTestRunner(
                command=args["command"],
                env_vars=_env_dict_to_tuple(args.get("env_vars", {})),
                timeout=int(args.get("timeout", 30)),
                initiator=Initiator.UI,
            )
        case "run_tests":
            return RunTestsRequested(initiator=Initiator.UI)
        case "cancel_tests":
            return CancelTestsRequested(initiator=Initiator.UI)
    return None


# --- Factory ---


def create_test_suite() -> WidgetHandle:
    return create_widget(
        id="widget-test-suite",
        model=TestSuiteModel(),
        update=_create_update(),
        view_messages=view_messages,
        view_tools=view_tools,
        view_html=view_html,
        from_llm=from_llm,
        from_ui=from_ui,
        frontend_tools=frozenset(
            {"configure_test_runner", "run_tests", "cancel_tests"}
        ),
    )
