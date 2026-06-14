from __future__ import annotations

from typing import AsyncGenerator

import structlog
from openai import AsyncOpenAI

from src.config import settings

logger = structlog.get_logger(__name__)


class LLMClient:
    """
    Async LLM client pointing at Ollama's OpenAI-compatible endpoint.
    Default model: settings.online_model (gemma4 or qwen3.5).
    """

    def __init__(self) -> None:
        self._client = AsyncOpenAI(
            base_url=settings.online_base_url,
            api_key=settings.llm_api_key,
        )

    async def complete(
        self,
        system: str,
        user: str,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str:
        model = model or settings.online_model
        response = await self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = response.choices[0].message.content or ""
        logger.debug(
            "llm_complete",
            model=model,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )
        return text

    async def complete_with_history(
        self,
        messages: list[dict],
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str:
        model = model or settings.online_model
        response = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    async def stream_complete(
        self,
        system: str,
        user: str,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> AsyncGenerator[str, None]:
        model = model or settings.online_model
        stream = await self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
