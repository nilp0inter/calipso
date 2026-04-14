"""Code summarizer — cheap LLM that extracts signatures and describes bodies."""

import os

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

_INSTRUCTIONS = """\
You summarize Python code. You receive code with comments already stripped.

For every function or class, keep the SIGNATURE line EXACTLY as written, \
then replace its body with a docstring that fully describes what the \
body does, followed by the comment "# [ CODE REDACTED ]".

NEVER change, shorten, or paraphrase a signature. Copy it character for \
character. NEVER invent new classes or functions that are not in the input. \
NEVER add "raise" or "pass" or "NotImplemented" or any code. \
Only the signature and a docstring.

Drop all imports, assignments, and top-level statements.

===== EXAMPLE 1 =====
INPUT:
def greet(name: str) -> str:
    return f"Hello, {name}!"

OUTPUT:
def greet(name: str) -> str:
    \"\"\"Return a greeting string with the given name.\"\"\"
    # [ CODE REDACTED ]

===== EXAMPLE 2 =====
INPUT:
def process_batch(items: list[Item], config: Config) -> Result:
    validated = validate(config)
    chunks = chunk_list(items, validated.size)
    results = pool.map(run, chunks)
    return aggregate(results)

OUTPUT:
def process_batch(items: list[Item], config: Config) -> Result:
    \"\"\"Validate the config, split items into chunks based on the
    validated chunk size, process each chunk in parallel using a
    pool, and aggregate all results into a single Result.\"\"\"
    # [ CODE REDACTED ]

===== EXAMPLE 3 =====
INPUT:
class Calculator:
    def add(self, a: int, b: int) -> int:
        return a + b

    def multiply(self, a: int, b: int) -> int:
        return a * b

OUTPUT:
class Calculator:
    \"\"\"A calculator with add and multiply operations.\"\"\"
    # [ CODE REDACTED ]

    def add(self, a: int, b: int) -> int:
        \"\"\"Return the sum of a and b.\"\"\"
        # [ CODE REDACTED ]

    def multiply(self, a: int, b: int) -> int:
        \"\"\"Return the product of a and b.\"\"\"
        # [ CODE REDACTED ]

===== EXAMPLE 4 =====
INPUT:
@dataclass
class Widget:
    def widget_id(self) -> str:
        name = re.sub(r"(?<!^)(?=[A-Z])", "-", type(self).__name__).lower()
        return f"widget-{name}"

    def view_messages(self) -> Iterator[ModelMessage]:
        return iter(())

    async def update(self, tool_name: str, args: dict) -> str:
        raise NotImplementedError(
            f"{type(self).__name__} does not handle tool '{tool_name}'"
        )

OUTPUT:
@dataclass
class Widget:
    \"\"\"Base widget class with ID generation, message views, and
    an update handler that subclasses must override.\"\"\"
    # [ CODE REDACTED ]

    def widget_id(self) -> str:
        \"\"\"Convert the class name to kebab-case and return it
        prefixed with 'widget-' as a stable HTML element ID.\"\"\"
        # [ CODE REDACTED ]

    def view_messages(self) -> Iterator[ModelMessage]:
        \"\"\"Return an empty iterator (no messages by default).\"\"\"
        # [ CODE REDACTED ]

    async def update(self, tool_name: str, args: dict) -> str:
        \"\"\"Raise NotImplementedError indicating that this widget
        does not handle the given tool name.\"\"\"
        # [ CODE REDACTED ]

===== END EXAMPLES =====

Output ONLY the summarized code. No markdown fences. No explanations.
"""


def create_summarizer_agent() -> Agent:
    """Create a Pydantic AI agent for code summarization."""
    provider = OpenAIProvider(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )
    model = OpenAIChatModel(
        "google/gemini-3.1-flash-lite-preview",
        provider=provider,
    )
    return Agent(
        model,
        instructions=_INSTRUCTIONS,
        output_type=str,
    )
