"""Configurable LLM client for generating clinical summaries."""

from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass
from typing import Any, Optional

import requests


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for the :class:`LLMClient`."""

    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    timeout: float = 30.0
    base_url: Optional[str] = None
    api_key: Optional[str] = None

    @classmethod
    def from_env(cls, **overrides: Any) -> "LLMConfig":
        """Build configuration using environment overrides if available."""

        api_key = overrides.get("api_key") or os.getenv("OPENAI_API_KEY")
        model = overrides.get("model") or os.getenv("OPENAI_MODEL")
        base_url = overrides.get("base_url") or os.getenv("OPENAI_BASE_URL")
        provider = overrides.get("provider") or overrides.get(
            "provider", "openai"
        )
        return cls(
            provider=provider,
            model=model or cls.model,
            temperature=overrides.get("temperature", cls.temperature),
            max_tokens=overrides.get("max_tokens", cls.max_tokens),
            timeout=overrides.get("timeout", cls.timeout),
            base_url=base_url,
            api_key=api_key,
        )


class LLMClient:
    """Deterministic language model client with validation safeguards."""

    _RETRY_DELAYS = (0.5, 1.0, 2.0)

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        self._config = config or LLMConfig.from_env()
        self._provider = self._config.provider.lower()
        if self._provider == "openai" and not self._config.api_key:
            raise ValueError(
                "OPENAI_API_KEY must be set for provider='openai'"
            )
        if self._provider == "custom" and not self._config.base_url:
            raise ValueError(
                "base_url must be provided for provider='custom'"
            )

    def generate(self, prompt: str, max_words: int) -> str:
        """Generate a summary respecting deterministic and safety limits."""

        max_tokens = self._compute_max_tokens(max_words)
        cleaned_prompt = prompt.strip()
        if not cleaned_prompt:
            raise ValueError("Prompt must be non-empty")

        for attempt, delay in enumerate((*self._RETRY_DELAYS, None), start=1):
            try:
                raw_text = self._dispatch(cleaned_prompt, max_tokens, max_words)
                return self._post_process(raw_text, max_words)
            except Exception as exc:  # noqa: BLE001
                if attempt >= len(self._RETRY_DELAYS) + 1:
                    logger.error("LLM generation failed: %s", exc)
                    raise
                logger.warning(
                    "LLM generation error (attempt %s): %s", attempt, exc
                )
                time.sleep(delay or 0.0)

        raise RuntimeError("Exhausted LLM retries")

    def _dispatch(
        self, prompt: str, max_tokens: int, max_words: int
    ) -> str:
        if self._provider == "openai":
            return self._generate_openai(prompt, max_tokens, max_words)
        if self._provider == "custom":
            return self._generate_custom(prompt, max_words)
        raise ValueError(f"Unsupported LLM provider '{self._provider}'")

    def _generate_openai(
        self, prompt: str, max_tokens: int, max_words: int
    ) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "openai package is required for provider='openai'"
            ) from exc

        system_message = (
            "You are a clinical reporting assistant. Produce concise, clinically"
            " valid summaries.\nConstraints:\n- Mention only the provided"
            " pathway names.\n- Do not introduce any new diseases, genes, or"
            " pathways.\n- Keep the summary ≤ {max_words} words (4–5 sentences)."
            "\n- Use clear clinical language suitable for diagnostic reporting."
            "\n- If unsure, say 'insufficient information' rather than inventing"
            " content."
        ).format(max_words=max_words)

        client = OpenAI(
            api_key=self._config.api_key,
            base_url=self._config.base_url or None,
        )

        response = client.chat.completions.create(
            model=self._config.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            temperature=self._config.temperature,
            max_tokens=max_tokens,
            timeout=self._config.timeout,
        )
        choice = response.choices[0]
        message = choice.message
        return message.content or ""

    def _generate_custom(self, prompt: str, max_words: int) -> str:
        response = requests.post(
            self._config.base_url,
            json={"prompt": prompt, "max_words": max_words},
            timeout=self._config.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        text = payload.get("summary") or payload.get("text")
        if not isinstance(text, str):
            raise ValueError("Custom LLM response missing 'summary' field")
        return text

    @staticmethod
    def _post_process(text: str, max_words: int) -> str:
        collapsed = " ".join(text.strip().split())
        if not collapsed:
            raise ValueError("LLM returned empty content")
        words = collapsed.split()
        if len(words) > max_words:
            collapsed = " ".join(words[:max_words])
        return collapsed

    def _compute_max_tokens(self, max_words: int) -> int:
        if self._config.max_tokens is not None:
            return max(1, self._config.max_tokens)
        estimated = int(max_words * 1.5)
        return max(1, estimated)


def get_default_llm_client() -> LLMClient:
    """Instantiate an :class:`LLMClient` using environment configuration."""

    return LLMClient()


