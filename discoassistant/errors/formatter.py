from __future__ import annotations

import json
from typing import Any

import aiohttp


def safe_runtime_error_reply(exc: Exception) -> str:
    if isinstance(exc, aiohttp.ClientResponseError) and exc.status == 429:
        if "openrouter.ai" in str(exc.request_info.real_url):
            return "Model rate-limited me. Try again in a few seconds."
        return "Rate-limited by upstream service. Try again in a few seconds."
    message = str(exc).strip()
    if message:
        return f"I hit an internal error: {type(exc).__name__}: {message}"
    return f"I hit an internal error: {type(exc).__name__}."


def synthesize_deterministic_reply(messages: list[dict[str, Any]] | None) -> str | None:
    if not messages:
        return None
    for item in reversed(messages):
        if item.get("role") != "tool":
            continue
        try:
            payload = json.loads(item.get("content", "{}"))
        except json.JSONDecodeError:
            continue
        if payload.get("ok") is False:
            continue
        name = item.get("name")
        if name == "read_channel_messages":
            return "I caught up on recent messages — what would you like me to do with them?"
        if name == "lookup_user":
            display = payload.get("display_name") or payload.get("username") or "the user"
            return f"That's {display} (id `{payload.get('target_user_id')}`)."
        if name == "send_dm" and payload.get("ok") is True:
            return f"Message sent to user `{payload.get('target_user_id')}`."
        if name == "remember" and payload.get("ok") is True:
            return "Saved."
        if name == "replace_in_memory" and payload.get("ok") is True:
            return "Updated."
        if name == "server_log" and payload.get("ok") is True:
            status = payload.get("status") or payload.get("action") or "ok"
            return f"Server log {status}."
    return None


def fallback_response_text(
    messages: list[dict[str, Any]] | None,
    *,
    default: str = "I couldn't produce a normal reply, but I can try again.",
) -> str:
    if messages:
        for item in reversed(messages):
            if item.get("role") != "tool":
                continue
            try:
                payload = json.loads(item.get("content", "{}"))
            except json.JSONDecodeError:
                continue

            if payload.get("ok") is False:
                error_message = str(payload.get("error_message", "")).strip()
                if error_message:
                    return f"Couldn't do that: {error_message}"
                return "Couldn't do that."

            if item.get("name") == "send_dm" and payload.get("ok") is True:
                target_user_id = payload.get("target_user_id")
                return f"Message sent to user `{target_user_id}`."

    return default
