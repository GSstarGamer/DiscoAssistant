from __future__ import annotations

import logging
from typing import Any

import discord

from discoassistant.llm.tool_registry import ToolContext, ToolSpec


LOGGER = logging.getLogger("discoassistant")


SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "read_channel_messages",
        "description": (
            "Read recent messages from the current Discord channel (or another channel by id). "
            "Use whenever the user's message references prior context, asks 'what were we talking about', "
            "mentions you with no other text, or otherwise can't be answered without scrollback. "
            "Default to the current channel and a small window; paginate older with before_message_id only if needed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "How many past messages to fetch. Default size is fine unless more context is needed.",
                    "minimum": 1,
                    "maximum": 100,
                },
                "before_message_id": {
                    "type": "integer",
                    "description": "Fetch messages older than this message id. Use returned older ids to paginate farther back.",
                },
                "target_channel_id": {
                    "type": "integer",
                    "description": "Optional explicit Discord channel id. Use this when the user mentions a specific #channel.",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    },
}


async def handler(arguments: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    services = ctx.services
    message = ctx.message
    limit = int(arguments.get("limit", services.max_history_messages()))
    limit = max(1, min(limit, 100))
    requested_channel_id = arguments.get("target_channel_id")
    target_channel = message.channel
    before_message_id = arguments.get("before_message_id", message.id)
    if requested_channel_id is not None:
        target_channel_id = int(requested_channel_id)
        target_channel = services.channel_from_message_context(message, target_channel_id)
        if target_channel is None:
            fetched_channel = await services.fetch_channel(target_channel_id)
            target_channel = fetched_channel
        before_message_id = arguments.get("before_message_id")
    LOGGER.info(
        "Fetching channel history requester=%s target_channel_id=%s limit=%s before_message_id=%s",
        message.author.id,
        getattr(target_channel, "id", None),
        limit,
        before_message_id,
    )

    if not hasattr(target_channel, "history"):
        raise ValueError("get_channel_history target channel does not support message history.")

    before_message = discord.Object(id=int(before_message_id)) if before_message_id is not None else None
    history: list[dict[str, Any]] = []
    async for item in target_channel.history(limit=limit, before=before_message, oldest_first=False):
        history.append(
            {
                "message_id": item.id,
                "author_user_id": item.author.id,
                "author_username": str(item.author),
                "author_display_name": services.display_name_for_message_author(item),
                "content": services.message_text_for_context(item),
                "created_at": item.created_at.isoformat(),
            }
        )
        services.store_historical_dm_message(item)

    return {
        "channel_id": target_channel.id,
        "fetched_count": len(history),
        "messages": history,
    }


SPEC = ToolSpec(name="read_channel_messages", schema=SCHEMA, handler=handler)
