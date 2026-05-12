from __future__ import annotations

import logging
from typing import Any

from discoassistant.llm.tool_registry import ToolContext, ToolSpec


LOGGER = logging.getLogger("discoassistant")


SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "send_dm",
        "description": (
            "Send a direct message to a specific Discord user. "
            "target_user_id must be explicit (from a mention, lookup_user, or context) — never guess. "
            "Non-owner may only target the owner; owner may target anyone. "
            "Never claim the message was sent unless this tool returned ok=true."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "target_user_id": {
                    "type": "integer",
                    "description": "Discord user id to DM. Required.",
                },
                "content": {
                    "type": "string",
                    "description": "Direct message content to send.",
                },
            },
            "required": ["target_user_id", "content"],
            "additionalProperties": False,
        },
    },
}


async def handler(arguments: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    services = ctx.services
    message = ctx.message
    author_id = message.author.id
    owner_user_id = services.owner_user_id
    is_owner = author_id == owner_user_id

    # Top-level dispatch normalizes content -> message; accept both forms here too.
    send_args = {
        "target_user_id": arguments.get("target_user_id"),
        "message": arguments.get("content", arguments.get("message", "")),
    }

    requested_target_user_id = send_args.get("target_user_id")
    if requested_target_user_id is None:
        raise ValueError("send_message.target_user_id is required. Do not guess recipient.")
    target_user_id = services.resolve_target_user_id(message, requested_target_user_id)

    content = str(send_args.get("message", "")).strip()
    if not content:
        raise ValueError("send_message.message is required.")

    if not is_owner and target_user_id != owner_user_id:
        raise ValueError("Only owner can send DMs to arbitrary users.")
    if services.user_id is not None and target_user_id == services.user_id:
        raise ValueError("Do not DM logged-in assistant account.")

    user = services.get_user(target_user_id)
    if user is None:
        user = await services.fetch_user(target_user_id)

    dm_channel = user.dm_channel
    if dm_channel is None:
        dm_channel = await user.create_dm()

    sent_message = await dm_channel.send(content)
    services.store_outgoing_dm_message(sent_message)
    return {
        "ok": True,
        "from_user_id": author_id,
        "target_user_id": target_user_id,
        "dm_channel_id": dm_channel.id,
        "message_id": sent_message.id,
        "sent_message": content,
    }


SPEC = ToolSpec(name="send_dm", schema=SCHEMA, handler=handler)
