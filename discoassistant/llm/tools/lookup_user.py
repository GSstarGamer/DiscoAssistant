from __future__ import annotations

import logging
from typing import Any

import discord

from discoassistant.llm.tool_registry import ToolContext, ToolSpec


LOGGER = logging.getLogger("discoassistant")


SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "lookup_user",
        "description": (
            "Get profile details for a Discord user (defaults to the current chatter). "
            "Use when the user asks who someone is, asks for an id, asks for bio details, "
            "or you need to verify a target before send_dm."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "target_user_id": {
                    "type": "integer",
                    "description": "Optional Discord user id. Omit to inspect the current message author.",
                }
            },
            "required": [],
            "additionalProperties": False,
        },
    },
}


async def handler(arguments: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    services = ctx.services
    message = ctx.message
    requested_target_user_id = arguments.get("target_user_id")
    target_user_id = (
        services.resolve_target_user_id(message, requested_target_user_id)
        if requested_target_user_id is not None
        else message.author.id
    )

    member = services.member_from_message_context(message, target_user_id)
    user = member or services.user_from_message_context(message, target_user_id)
    if user is None:
        user = await services.fetch_user(target_user_id)

    profile: Any | None = None
    profile_error: str | None = None
    try:
        if member is not None:
            profile = await member.profile(
                with_mutual_guilds=True,
                with_mutual_friends_count=False,
                with_mutual_friends=False,
            )
        elif hasattr(user, "profile"):
            profile = await user.profile(
                with_mutual_guilds=True,
                with_mutual_friends_count=False,
                with_mutual_friends=False,
            )
    except discord.HTTPException as exc:
        profile_error = f"{type(exc).__name__}: {exc}"

    effective_member = member if member is not None else (user if isinstance(user, discord.Member) else None)
    display_name = (
        services.display_name_for_message_author(message)
        if target_user_id == message.author.id
        else getattr(effective_member or user, "display_name", str(user))
    )
    bio = getattr(profile, "display_bio", None)
    if bio is None:
        bio = getattr(profile, "bio", None)
    if bio is None:
        bio = getattr(user, "bio", None)

    metadata = getattr(profile, "metadata", None)
    mutual_guilds = getattr(profile, "mutual_guilds", None)
    avatar = getattr(user, "display_avatar", None)
    banner = getattr(effective_member, "display_banner", None) or getattr(user, "banner", None)
    accent_color = None
    if metadata is not None and getattr(metadata, "accent_color", None) is not None:
        accent_color = str(metadata.accent_color)
    elif getattr(user, "accent_color", None) is not None:
        accent_color = str(user.accent_color)

    result: dict[str, Any] = {
        "ok": True,
        "target_user_id": user.id,
        "username": str(user),
        "name": getattr(user, "name", None),
        "display_name": display_name,
        "global_name": getattr(user, "global_name", None),
        "mention": getattr(user, "mention", f"<@{user.id}>"),
        "bot": getattr(user, "bot", False),
        "system": getattr(user, "system", False),
        "created_at": user.created_at.isoformat() if getattr(user, "created_at", None) else None,
        "avatar_url": str(avatar.url) if avatar is not None else None,
        "banner_url": str(banner.url) if banner is not None else None,
        "accent_color": accent_color,
        "bio": bio,
        "legacy_username": getattr(profile, "legacy_username", None),
        "profile_fetch_ok": profile_error is None,
        "profile_fetch_error": profile_error,
        "guild_member": {
            "in_current_guild": effective_member is not None,
            "nick": getattr(effective_member, "nick", None) if effective_member is not None else None,
            "joined_at": (
                effective_member.joined_at.isoformat()
                if effective_member is not None and getattr(effective_member, "joined_at", None)
                else None
            ),
            "guild_bio": getattr(profile, "guild_bio", None),
        },
        "mutual_guilds": (
            [
                {
                    "id": guild.id,
                    "name": getattr(getattr(guild, "guild", None), "name", None),
                    "nick": getattr(guild, "nick", None),
                }
                for guild in mutual_guilds[:10]
            ]
            if mutual_guilds
            else []
        ),
    }
    if metadata is not None:
        result["profile_metadata"] = {
            "pronouns": getattr(metadata, "pronouns", None),
            "banner_url": str(metadata.banner.url) if getattr(metadata, "banner", None) is not None else None,
        }
    return result


SPEC = ToolSpec(name="lookup_user", schema=SCHEMA, handler=handler)
