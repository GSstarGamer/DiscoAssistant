from __future__ import annotations

import logging
from typing import Any

from discoassistant.llm.tool_registry import ToolContext, ToolSpec


LOGGER = logging.getLogger("discoassistant")


SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "server_log",
        "description": (
            "Manage the passive server-memory queue for the current guild. "
            "action='preview' shows queued message count and starts a confirmation window — call this first when the owner asks to flush. "
            "action='flush' summarizes the queue into server memory; call only after preview AND explicit owner confirmation in the same turn-pair. "
            "action='cancel' discards a pending flush confirmation when the owner declines. "
            "Owner only."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["preview", "flush", "cancel"],
                    "description": "Which queue operation to run.",
                },
                "guild_id": {
                    "type": "integer",
                    "description": "Optional explicit guild id. Leave unset in a server channel to use the current guild.",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    },
}


def _require_owner(ctx: ToolContext, tool_name: str) -> None:
    if ctx.message.author.id != ctx.services.owner_user_id:
        raise ValueError(f"{tool_name} is restricted to the owner.")


async def _preview(arguments: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    del arguments
    services = ctx.services
    message = ctx.message
    _require_owner(ctx, "preview_passive_server_memory_flush")
    if message.guild is None:
        raise ValueError("preview_passive_server_memory_flush requires a guild/server channel.")

    guild_id = message.guild.id
    pending_count = services.passive_summarizer.pending_count(guild_id)
    services.register_passive_flush_confirmation(
        guild_id=guild_id,
        requester_user_id=message.author.id,
        pending_message_count=pending_count,
    )
    return {
        "ok": True,
        "guild_id": guild_id,
        "guild_name": message.guild.name,
        "pending_message_count": pending_count,
        "confirmation_required": True,
        "confirmation_window_seconds": 300,
        "status": "preview_ready",
    }


async def _flush(arguments: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    services = ctx.services
    message = ctx.message
    _require_owner(ctx, "flush_passive_server_memory")

    requested_guild_id = arguments.get("guild_id")
    if requested_guild_id is None:
        if message.guild is None:
            raise ValueError("flush_passive_server_memory.guild_id is required outside a guild channel.")
        guild_id = message.guild.id
    else:
        guild_id = int(requested_guild_id)
    confirmation = services.consume_passive_flush_confirmation(
        guild_id=guild_id,
        requester_user_id=message.author.id,
    )
    if confirmation is None:
        raise ValueError(
            "Preview required before flush. Call preview_passive_server_memory_flush first, tell the user how many messages are queued, and only flush after they explicitly confirm."
        )
    LOGGER.info("Forcing passive server memory flush requester=%s guild_id=%s", message.author.id, guild_id)

    if not services.passive_summarizer.is_enabled_for_guild(guild_id):
        raise ValueError(f"Passive guild memory is not enabled for guild_id={guild_id}.")

    result = await services.passive_summarizer.flush_now(guild_id)
    result["preview_pending_message_count"] = confirmation.pending_message_count
    return result


async def _cancel(arguments: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    del arguments
    services = ctx.services
    message = ctx.message
    _require_owner(ctx, "cancel_passive_server_memory_flush")
    if message.guild is None:
        raise ValueError("cancel_passive_server_memory_flush requires a guild/server channel.")

    cancelled = services.discard_passive_flush_confirmation(
        guild_id=message.guild.id,
        requester_user_id=message.author.id,
    )
    return {
        "ok": cancelled,
        "guild_id": message.guild.id,
        "guild_name": message.guild.name,
        "status": "cancelled" if cancelled else "no_pending_confirmation",
    }


async def handler(arguments: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    action = str(arguments.get("action", "")).strip().lower()
    if action == "preview":
        return await _preview(arguments, ctx)
    if action == "flush":
        return await _flush(arguments, ctx)
    if action == "cancel":
        return await _cancel(arguments, ctx)
    raise ValueError("server_log.action must be 'preview', 'flush', or 'cancel'.")


SPEC = ToolSpec(name="server_log", schema=SCHEMA, handler=handler, owner_only=True)
