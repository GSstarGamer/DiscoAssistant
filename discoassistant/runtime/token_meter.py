from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any


LOGGER = logging.getLogger("discoassistant")


class TokenMeter:
    """Tracks cumulative token usage across the OpenRouter session."""

    __slots__ = (
        "_total",
        "_prompt",
        "_output",
        "_cache_hit",
        "_cache_miss",
        "_on_change",
    )

    def __init__(self, *, on_change: Callable[[], None] | None = None) -> None:
        self._total = 0
        self._prompt = 0
        self._output = 0
        self._cache_hit = 0
        self._cache_miss = 0
        self._on_change = on_change

    @property
    def total(self) -> int:
        return self._total

    @property
    def prompt(self) -> int:
        return self._prompt

    @property
    def output(self) -> int:
        return self._output

    @property
    def cache_hit(self) -> int:
        return self._cache_hit

    @property
    def cache_miss(self) -> int:
        return self._cache_miss

    def record(self, response_payload: dict[str, Any]) -> None:
        usage = response_payload.get("usage")
        if not isinstance(usage, dict):
            return

        prompt_tokens = usage.get("prompt_tokens") or 0
        output_tokens = usage.get("completion_tokens") or 0
        cache_hit_tokens = usage.get("prompt_cache_hit_tokens") or 0
        cache_miss_tokens = usage.get("prompt_cache_miss_tokens") or 0
        total_tokens = usage.get("total_tokens") or (prompt_tokens + output_tokens)

        if isinstance(prompt_tokens, int) and prompt_tokens > 0:
            self._prompt += prompt_tokens
        if isinstance(output_tokens, int) and output_tokens > 0:
            self._output += output_tokens
        if isinstance(cache_hit_tokens, int) and cache_hit_tokens > 0:
            self._cache_hit += cache_hit_tokens
        if isinstance(cache_miss_tokens, int) and cache_miss_tokens > 0:
            self._cache_miss += cache_miss_tokens

        if isinstance(total_tokens, int) and total_tokens > 0:
            self._total += total_tokens
            if self._on_change is not None:
                self._on_change()

        cache_hit_ratio = 0.0
        cacheable = (cache_hit_tokens or 0) + (cache_miss_tokens or 0)
        if cacheable > 0:
            cache_hit_ratio = (cache_hit_tokens or 0) / cacheable
        LOGGER.info(
            "tokens model=%s prompt=%s (cache_hit=%s cache_miss=%s ratio=%.2f) output=%s session_total=%s",
            response_payload.get("model", "?"),
            prompt_tokens,
            cache_hit_tokens,
            cache_miss_tokens,
            cache_hit_ratio,
            output_tokens,
            self._total,
        )
