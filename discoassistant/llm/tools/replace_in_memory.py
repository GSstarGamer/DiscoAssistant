from __future__ import annotations

import logging
from typing import Any

from discoassistant.llm.tool_registry import ToolContext, ToolSpec


LOGGER = logging.getLogger("discoassistant")


SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "replace_in_memory",
        "description": (
            "Edit an existing memory by substring replacement. "
            "scope chooses user vs server file. old_text must match a unique substring; new_text replaces it. "
            "Owner only."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "scope": {
                    "type": "string",
                    "enum": ["user", "server"],
                    "description": "Which memory file to edit.",
                },
                "old_text": {
                    "type": "string",
                    "description": "Existing substring to replace. Must match exactly.",
                },
                "new_text": {
                    "type": "string",
                    "description": "Replacement text.",
                },
            },
            "required": ["scope", "old_text", "new_text"],
            "additionalProperties": False,
        },
    },
}


def _require_owner(ctx: ToolContext, tool_name: str) -> None:
    if ctx.message.author.id != ctx.services.owner_user_id:
        raise ValueError(f"{tool_name} is restricted to the owner.")


async def _edit_user_memory(arguments: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    services = ctx.services
    message = ctx.message
    _require_owner(ctx, "edit_user_memory")
    old_text = str(arguments.get("old_text", "")).strip()
    new_text = str(arguments.get("new_text", "")).strip()
    if not old_text or not new_text:
        raise ValueError("edit_user_memory.old_text and new_text are required.")

    path, updated = services.user_memory_store.replace_for_user(
        user_id=message.author.id,
        old_text=old_text,
        new_text=new_text,
    )
    return {
        "ok": updated,
        "user_id": message.author.id,
        "path": str(path),
        "old_text": old_text,
        "new_text": new_text,
        "status": "updated" if updated else "not_found",
    }


async def _edit_server_memory(arguments: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    services = ctx.services
    message = ctx.message
    _require_owner(ctx, "edit_server_memory")
    if message.guild is None:
        raise ValueError("edit_server_memory requires a guild/server channel.")

    old_text = str(arguments.get("old_text", "")).strip()
    new_text = str(arguments.get("new_text", "")).strip()
    if not old_text or not new_text:
        raise ValueError("edit_server_memory.old_text and new_text are required.")
    LOGGER.info(
        "Editing server memory requester=%s guild_id=%s old_chars=%s new_chars=%s",
        message.author.id,
        message.guild.id,
        len(old_text),
        len(new_text),
    )

    path, updated = services.guild_memory_store.replace_for_guild(
        guild_id=message.guild.id,
        old_text=old_text,
        new_text=new_text,
    )
    if updated:
        await services.owner_notifier.notify_owner_of_guild_memory_update(
            guild_id=message.guild.id,
            guild_name=message.guild.name,
            summary=(
                "Owner-approved guild memory edit:\n"
                f"- old: {old_text}\n"
                f"- new: {new_text}"
            ),
        )
    return {
        "ok": updated,
        "guild_id": message.guild.id,
        "guild_name": message.guild.name,
        "path": str(path),
        "old_text": old_text,
        "new_text": new_text,
        "status": "updated" if updated else "not_found",
    }


async def handler(arguments: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    scope = str(arguments.get("scope", "")).strip().lower()
    edit_args = {
        "old_text": arguments.get("old_text", ""),
        "new_text": arguments.get("new_text", ""),
    }
    if scope == "user":
        return await _edit_user_memory(edit_args, ctx)
    if scope == "server":
        return await _edit_server_memory(edit_args, ctx)
    raise ValueError("replace_in_memory.scope must be 'user' or 'server'.")


SPEC = ToolSpec(name="replace_in_memory", schema=SCHEMA, handler=handler, owner_only=True)
