import asyncio
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def call_llm(
    messages: list[dict],
    model: str = "gpt-4o",
    temperature: float = 0,
    response_format: dict | None = None,
) -> str:
    """Send a chat completion request to the OpenAI API.

    Args:
        messages: Conversation turns in OpenAI message format.
        model: The model identifier to use.
        temperature: Sampling temperature; 0 produces deterministic output.
        response_format: Optional output constraint, e.g. ``{"type": "json_object"}``.

    Returns:
        The content string of the first choice.

    Raises:
        openai.APIError: On network or upstream failures.
    """
    kwargs: dict = {"model": model, "messages": messages, "temperature": temperature}
    if response_format:
        kwargs["response_format"] = response_format
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


async def async_call_llm(
    messages: list[dict],
    model: str = "gpt-4o",
    temperature: float = 0,
    response_format: dict | None = None,
) -> str:
    """Async wrapper around ``call_llm`` using a thread pool for I/O parallelism.

    Runs the synchronous OpenAI call in ``asyncio.to_thread`` so that multiple
    agent calls can proceed concurrently via ``asyncio.gather``.

    Args:
        messages: Conversation turns in OpenAI message format.
        model: The model identifier to use.
        temperature: Sampling temperature.
        response_format: Optional output constraint.

    Returns:
        The content string of the first choice.
    """
    return await asyncio.to_thread(call_llm, messages, model, temperature, response_format)
