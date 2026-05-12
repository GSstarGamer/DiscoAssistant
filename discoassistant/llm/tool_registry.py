from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import discord

from discoassistant.memory import GuildMemoryStore, UserMemoryStore


LOGGER = logging.getLogger("discoassistant")


class BotServices(Protocol):
    """Protocol describing the slice of bot state that tool handlers depend on.

    The concrete bot implements every attribute and method declared here. Tool
    modules import this Protocol (not the bot class) so they can run without
    creating a circular dependency on `discoassistant.bot`.
    """

    user_memory_store: UserMemoryStore
    guild_memory_store: GuildMemoryStore
    passive_summarizer: Any
    owner_notifier: Any
    owner_user_id: int
    user_id: int | None

    def display_name_for_message_author(self, message: discord.Message) -> str: ...

    def message_text_for_context(self, message: discord.Message) -> str: ...

    def channel_from_message_context(
        self,
        message: discord.Message,
        channel_id: int,
    ) -> Any: ...

    def member_from_message_context(
        self,
        message: discord.Message,
        user_id: int,
    ) -> Any: ...

    def user_from_message_context(
        self,
        message: discord.Message,
        user_id: int,
    ) -> Any: ...

    def resolve_target_user_id(
        self,
        message: discord.Message,
        raw_value: Any,
    ) -> int: ...

    async def fetch_channel(self, channel_id: int) -> Any: ...

    async def fetch_user(self, user_id: int) -> Any: ...

    def get_user(self, user_id: int) -> Any: ...

    def store_historical_dm_message(self, message: discord.Message) -> None: ...

    def store_outgoing_dm_message(self, message: discord.Message) -> None: ...

    def is_direct_message(self, message: discord.Message) -> bool: ...

    def max_history_messages(self) -> int: ...

    def register_passive_flush_confirmation(
        self,
        *,
        guild_id: int,
        requester_user_id: int,
        pending_message_count: int,
    ) -> None: ...

    def consume_passive_flush_confirmation(
        self,
        *,
        guild_id: int,
        requester_user_id: int,
    ) -> Any: ...

    def discard_passive_flush_confirmation(
        self,
        *,
        guild_id: int,
        requester_user_id: int,
    ) -> bool: ...


@dataclass
class ToolContext:
    message: discord.Message
    services: BotServices


@dataclass
class ToolSpec:
    name: str
    schema: dict[str, Any]
    handler: Callable[[dict[str, Any], ToolContext], Awaitable[dict[str, Any]]]
    owner_only: bool = False


class ToolRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}
        self._order: list[str] = []

    def register(self, spec: ToolSpec) -> None:
        if spec.name not in self._specs:
            self._order.append(spec.name)
        self._specs[spec.name] = spec

    def __contains__(self, name: str) -> bool:
        return name in self._specs

    def names(self) -> list[str]:
        return list(self._order)

    def schemas_for(
        self,
        allowed_names: list[str],
        *,
        allow_model_tool_calls: bool,
        enabled_names: set[str],
    ) -> list[dict[str, Any]]:
        if not allow_model_tool_calls:
            return []

        allowed_set = enabled_names.intersection(allowed_names)
        schemas: list[dict[str, Any]] = []
        for name in self._order:
            if name not in allowed_set:
                continue
            spec = self._specs.get(name)
            if spec is None:
                continue
            schemas.append(spec.schema)
        return schemas

    async def dispatch(
        self,
        name: str,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> dict[str, Any]:
        spec = self._specs.get(name)
        if spec is None:
            raise RuntimeError(f"Unsupported tool call: {name}")
        return await spec.handler(arguments, context)
