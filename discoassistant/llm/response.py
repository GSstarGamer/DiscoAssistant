from __future__ import annotations

import logging
from typing import Any

from discoassistant.errors.formatter import fallback_response_text


LOGGER = logging.getLogger("discoassistant")


def extract_response_text(
    response: dict[str, Any],
    *,
    messages: list[dict[str, Any]] | None = None,
) -> str:
    choices = response.get("choices", [])
    if not choices:
        LOGGER.warning("OpenRouter response had no choices. Falling back to default reply.")
        return fallback_response_text(
            messages,
            default="I couldn't produce a reply just now.",
        )

    choice = choices[0]
    finish_reason = choice.get("finish_reason")
    message = choices[0].get("message", {})
    content = message.get("content", "")

    if isinstance(content, str):
        text = content.strip()
        if text:
            return text

    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_value = item.get("text", "").strip()
                if text_value:
                    text_parts.append(text_value)
        if text_parts:
            return "\n".join(text_parts)

    LOGGER.warning(
        "OpenRouter response had no usable text content. finish_reason=%r content_type=%s tool_calls=%s message_keys=%s",
        finish_reason,
        type(content).__name__,
        bool(message.get("tool_calls")),
        sorted(message.keys()),
    )
    return fallback_response_text(messages)
