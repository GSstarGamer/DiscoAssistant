from __future__ import annotations

from dataclasses import dataclass
from time import monotonic

import discord


@dataclass(slots=True)
class PendingPassiveFlushConfirmation:
    guild_id: int
    requester_user_id: int
    pending_message_count: int
    expires_at: float


class PassiveFlushConfirmationStore:
    """Tracks outstanding owner confirmations for passive guild memory flushes."""

    __slots__ = ("_pending",)

    def __init__(self) -> None:
        self._pending: dict[tuple[int, int], PendingPassiveFlushConfirmation] = {}

    def register(
        self,
        *,
        guild_id: int,
        requester_user_id: int,
        pending_message_count: int,
        ttl_seconds: float = 300,
    ) -> None:
        self._pending[(guild_id, requester_user_id)] = PendingPassiveFlushConfirmation(
            guild_id=guild_id,
            requester_user_id=requester_user_id,
            pending_message_count=pending_message_count,
            expires_at=monotonic() + ttl_seconds,
        )

    def consume(
        self,
        *,
        guild_id: int,
        requester_user_id: int,
    ) -> PendingPassiveFlushConfirmation | None:
        key = (guild_id, requester_user_id)
        confirmation = self._pending.get(key)
        if confirmation is None:
            return None
        if monotonic() > confirmation.expires_at:
            self._pending.pop(key, None)
            return None
        self._pending.pop(key, None)
        return confirmation

    def peek(
        self,
        *,
        guild_id: int,
        requester_user_id: int,
    ) -> PendingPassiveFlushConfirmation | None:
        key = (guild_id, requester_user_id)
        confirmation = self._pending.get(key)
        if confirmation is None:
            return None
        if monotonic() > confirmation.expires_at:
            self._pending.pop(key, None)
            return None
        return confirmation

    def discard(
        self,
        *,
        guild_id: int,
        requester_user_id: int,
    ) -> bool:
        key = (guild_id, requester_user_id)
        return self._pending.pop(key, None) is not None

    def has_pending_for_message(self, message: discord.Message) -> bool:
        if message.guild is None:
            return False
        return (
            self.peek(
                guild_id=message.guild.id,
                requester_user_id=message.author.id,
            )
            is not None
        )

    def context_for_message(self, message: discord.Message) -> str:
        if message.guild is None:
            return ""
        confirmation = self.peek(
            guild_id=message.guild.id,
            requester_user_id=message.author.id,
        )
        if confirmation is None:
            return ""
        return (
            "Pending passive server-memory flush confirmation is active.\n"
            f"- guild_id: {confirmation.guild_id}\n"
            f"- queued_message_count: {confirmation.pending_message_count}\n"
            "- If the owner confirms, call server_log with action='flush'.\n"
            "- If the owner declines, call server_log with action='cancel'.\n"
            "- Do not call server_log with action='preview' again while this confirmation is active.\n"
        )
