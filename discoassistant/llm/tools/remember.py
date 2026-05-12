from __future__ import annotations

import logging
from typing import Any

from discoassistant.llm.tool_registry import ToolContext, ToolSpec


LOGGER = logging.getLogger("discoassistant")


SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "remember",
        "description": (
            "Save a durable note. scope='user' writes to the user's own memory file (identity, projects, long-term preferences, biographical facts). "
            "scope='server' writes to the current guild's shared memory (server-wide rules, channel norms, cues that should affect replies to everyone in this guild). "
            "Only the owner may write to scope='server'. Never save throwaway details — only things that should still matter next week."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "scope": {
                    "type": "string",
                    "enum": ["user", "server"],
                    "description": "Which memory file to append to.",
                },
                "note": {
                    "type": "string",
                    "description": "Concise markdown-safe note worth remembering.",
                },
            },
            "required": ["scope", "note"],
            "additionalProperties": False,
        },
    },
}


def _require_owner(ctx: ToolContext, tool_name: str) -> None:
    if ctx.message.author.id != ctx.services.owner_user_id:
        raise ValueError(f"{tool_name} is restricted to the owner.")


async def _append_user_memory(arguments: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    services = ctx.services
    message = ctx.message
    _require_owner(ctx, "append_user_memory")
    note = str(arguments.get("note", "")).strip()
    if not note:
        raise ValueError("append_user_memory.note is required.")

    path = services.user_memory_store.append_for_user(
        user_id=message.author.id,
        note=note,
        author_display_name=services.display_name_for_message_author(message),
        source_channel_id=message.channel.id,
    )
    return {
        "ok": True,
        "user_id": message.author.id,
        "path": str(path),
        "appended_note": note,
    }


async def _append_server_memory(arguments: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    services = ctx.services
    message = ctx.message
    _require_owner(ctx, "append_server_memory")
    if message.guild is None:
        raise ValueError("append_server_memory requires a guild/server channel.")

    note = str(arguments.get("note", "")).strip()
    if not note:
        raise ValueError("append_server_memory.note is required.")
    LOGGER.info(
        "Appending server memory requester=%s guild_id=%s channel_id=%s note_chars=%s",
        message.author.id,
        message.guild.id,
        message.channel.id,
        len(note),
    )

    path = services.guild_memory_store.append_for_guild(
        guild_id=message.guild.id,
        guild_name=message.guild.name,
        note=note,
        author_display_name=services.display_name_for_message_author(message),
        source_channel_id=message.channel.id,
        owner_priority=True,
    )
    await services.owner_notifier.notify_owner_of_guild_memory_update(
        guild_id=message.guild.id,
        guild_name=message.guild.name,
        summary=(
            "Owner-approved guild memory append:\n"
            f"- channel_id: {message.channel.id}\n"
            f"- author: {services.display_name_for_message_author(message)}\n"
            f"- note: {note}"
        ),
    )
    return {
        "ok": True,
        "guild_id": message.guild.id,
        "guild_name": message.guild.name,
        "path": str(path),
        "appended_note": note,
    }


async def handler(arguments: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    scope = str(arguments.get("scope", "")).strip().lower()
    note_args = {"note": arguments.get("note", "")}
    if scope == "user":
        return await _append_user_memory(note_args, ctx)
    if scope == "server":
        return await _append_server_memory(note_args, ctx)
    raise ValueError("remember.scope must be 'user' or 'server'.")


SPEC = ToolSpec(name="remember", schema=SCHEMA, handler=handler)
