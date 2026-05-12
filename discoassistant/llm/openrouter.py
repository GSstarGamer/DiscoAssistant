from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from time import monotonic
from typing import Any

import aiohttp


LOGGER = logging.getLogger("discoassistant")


TokenUsageCallback = Callable[[dict[str, Any]], None] | None


class OpenRouterClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        app_name: str,
        site_url: str | None,
        default_model: str,
        on_token_usage: TokenUsageCallback = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._app_name = app_name
        self._site_url = site_url
        self._default_model = default_model
        self._on_token_usage = on_token_usage
        self._session: aiohttp.ClientSession | None = None

    async def setup(self) -> None:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "X-Title": self._app_name,
        }
        if self._site_url:
            headers["HTTP-Referer"] = self._site_url
        self._session = aiohttp.ClientSession(headers=headers)
        LOGGER.info(
            "OpenRouterClient session ready default_model=%s",
            self._default_model,
        )

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()

    @property
    def session(self) -> aiohttp.ClientSession | None:
        return self._session

    @property
    def default_model(self) -> str:
        return self._default_model

    async def chat(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]],
        extra_payload: dict[str, Any] | None = None,
        max_attempts: int = 4,
    ) -> dict[str, Any]:
        if self._session is None:
            raise RuntimeError("OpenRouterClient.setup() has not been called.")

        payload: dict[str, Any] = {
            "model": model or self._default_model,
            "messages": messages,
        }
        if extra_payload:
            payload.update(extra_payload)

        max_attempts = max(1, int(max_attempts))
        last_error: Exception | None = None
        selected_model = model or self._default_model
        for attempt in range(1, max_attempts + 1):
            request_started_at = monotonic()
            try:
                LOGGER.info(
                    "OpenRouter request start model=%s attempt=%s messages=%s tools=%s",
                    selected_model,
                    attempt,
                    len(messages),
                    "tools" in payload,
                )
                async with self._session.post(
                    f"{self._base_url}/chat/completions",
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    response_payload = await response.json()
                    if self._on_token_usage is not None:
                        try:
                            self._on_token_usage(response_payload)
                        except Exception:
                            LOGGER.exception("on_token_usage callback raised")
                    LOGGER.info(
                        "OpenRouter request success model=%s attempt=%s duration=%.2fs",
                        selected_model,
                        attempt,
                        monotonic() - request_started_at,
                    )
                    return response_payload
            except aiohttp.ClientResponseError as exc:
                last_error = exc
                should_retry = exc.status in {429, 500, 502, 503, 504} and attempt < max_attempts
                if not should_retry:
                    LOGGER.error(
                        "OpenRouter request failed model=%s attempt=%s duration=%.2fs status=%s error=%s",
                        selected_model,
                        attempt,
                        monotonic() - request_started_at,
                        exc.status,
                        exc,
                    )
                    raise

                retry_after = _retry_after_seconds(exc.headers)
                delay = retry_after if retry_after is not None else min(8.0, 1.5 * (2 ** (attempt - 1)))
                LOGGER.warning(
                    "OpenRouter request failed model=%s status=%s attempt=%s/%s duration=%.2fs retry_in=%.2fs",
                    selected_model,
                    exc.status,
                    attempt,
                    max_attempts,
                    monotonic() - request_started_at,
                    delay,
                )
                await asyncio.sleep(delay)
            except aiohttp.ClientError as exc:
                last_error = exc
                if attempt >= max_attempts:
                    LOGGER.error(
                        "OpenRouter network error model=%s attempt=%s duration=%.2fs error=%s",
                        selected_model,
                        attempt,
                        monotonic() - request_started_at,
                        exc,
                    )
                    raise
                delay = min(8.0, 1.0 * (2 ** (attempt - 1)))
                LOGGER.warning(
                    "OpenRouter network error model=%s attempt=%s/%s duration=%.2fs retry_in=%.2fs error=%s",
                    selected_model,
                    attempt,
                    max_attempts,
                    monotonic() - request_started_at,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)

        if last_error is not None:
            raise last_error
        raise RuntimeError("OpenRouter request failed without a captured exception.")


def _retry_after_seconds(headers: Any) -> float | None:
    if not headers:
        return None
    value = headers.get("Retry-After")
    if value is None:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, seconds)
