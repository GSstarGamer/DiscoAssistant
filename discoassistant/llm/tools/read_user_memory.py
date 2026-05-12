from __future__ import annotations

import logging
from typing import Any

from discoassistant.llm.tool_registry import ToolContext, ToolSpec


LOGGER = logging.getLogger("discoassistant")


SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "read_user_memory",
        "description": "Owner-only. Read the memory file for a specific Discord user id. Returns the markdown content.",
        "parameters": {
            "type": "object",
            "properties": {
                "target_user_id": {
                    "type": "integer",
                    "description": "Discord user id whose memory file to read.",
                }
            },
            "required": ["target_user_id"],
            "additionalProperties": False,
        },
    },
}


def _require_owner(ctx: ToolContext, tool_name: str) -> None:
    if ctx.message.author.id != ctx.services.owner_user_id:
        raise ValueError(f"{tool_name} is restricted to the owner.")


async def handler(arguments: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    services = ctx.services
    message = ctx.message
    _require_owner(ctx, "get_user_memory")

    requested_target_user_id = arguments.get("target_user_id")
    if requested_target_user_id is None:
        raise ValueError("get_user_memory.target_user_id is required.")
    target_user_id = services.resolve_target_user_id(message, requested_target_user_id)
    LOGGER.info("Reading user memory requester=%s target_user_id=%s", message.author.id, target_user_id)

    path = services.user_memory_store.path_for_user(target_user_id)
    content = services.user_memory_store.read_for_user(target_user_id)
    return {
        "ok": True,
        "target_user_id": target_user_id,
        "path": str(path),
        "exists": bool(content),
        "memory": content or "",
    }


SPEC = ToolSpec(name="read_user_memory", schema=SCHEMA, handler=handler, owner_only=True)
