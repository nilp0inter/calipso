"""Code summarizer — cheap LLM that extracts signatures and describes bodies."""

import os

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

_INSTRUCTIONS = """\
You summarize Python code. Comments have already been stripped.

RULES:
- Keep every def/class SIGNATURE line exactly as written.
- Replace the BODY of every function/method/class with EXACTLY ONE line:
  [...REDACTED...] # short description of what the body does
- The "# short description" part is MANDATORY. Never omit it.
- Drop imports, assignments, and any other top-level code.
- Output ONLY the result. No markdown fences, no explanations.

EXAMPLE INPUT:
def process_batch(items: list[Item], config: Config) -> Result:
    validated = validate(config)
    chunks = chunk_list(items, validated.size)
    results = pool.map(run, chunks)
    return aggregate(results)

class Processor:
    def run(self) -> None:
        self.setup()
        self.execute()

EXAMPLE OUTPUT:
def process_batch(items: list[Item], config: Config) -> Result:
    [...REDACTED...] # validates config, chunks items, maps over pool, aggregates

class Processor:
    [...REDACTED...] # processor with setup and execute lifecycle

    def run(self) -> None:
        [...REDACTED...] # calls setup then execute
"""


def create_summarizer_agent() -> Agent:
    """Create a Pydantic AI agent for code summarization."""
    provider = OpenAIProvider(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )
    model = OpenAIChatModel(
        "liquid/lfm-2.5-1.2b-thinking:free",
        provider=provider,
    )
    return Agent(
        model,
        instructions=_INSTRUCTIONS,
        output_type=str,
    )
