"""Code summarizer — cheap LLM that extracts signatures and describes bodies."""

import os

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

_INSTRUCTIONS = """\
You are a code summarizer. You receive Python source code with comments \
already stripped.

Your job:
1. Keep every function and class SIGNATURE exactly as-is (the `def` or \
`class` line, including decorators).
2. Replace every function/method BODY with "[...REDACTED...]" followed by \
a single-line comment describing what it does.
3. For top-level statements (imports, assignments, etc.), keep them as-is.
4. Output ONLY the transformed code. No explanations, no markdown fences.

Example input:
def process_batch(items: list[Item], config: Config) -> Result:
    validated = validate(config)
    chunks = chunk_list(items, validated.size)
    results = pool.map(run, chunks)
    return aggregate(results)

Example output:
def process_batch(items: list[Item], config: Config) -> Result:
    [...REDACTED...] # validates config, chunks items, maps over pool, aggregates
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
