from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import discord

from discoassistant.runtime.keys import ConversationKey, conversation_key


LOGGER = logging.getLogger("discoassistant")


@dataclass(slots=True)
class PendingReply:
    message: discord.Message
    updated_at: float
    first_seen_at: float
    started_from_mention: bool
    started_from_active_conversation: bool
    messages: list[discord.Message] = field(default_factory=list)


class PendingReplyManager:
    """Tracks debounced incoming messages awaiting a reply."""

    __slots__ = ("_pending", "_signal")

    def __init__(
        self,
        *,
        signal_callback: Callable[[ConversationKey], None] | None = None,
    ) -> None:
        self._pending: dict[ConversationKey, PendingReply] = {}
        self._signal = signal_callback

    def upsert(
        self,
        message: discord.Message,
        *,
        is_direct_mention: bool,
        has_active_conversation: bool,
        now: float,
    ) -> tuple[ConversationKey, PendingReply]:
        key = conversation_key(message)
        pending = self._pending.get(key)
        if pending is None:
            first_seen_at = now
            started_from_mention = is_direct_mention
            started_from_active_conversation = has_active_conversation
            pending_messages = [message]
        else:
            first_seen_at = pending.first_seen_at
            started_from_mention = pending.started_from_mention or is_direct_mention
            started_from_active_conversation = (
                pending.started_from_active_conversation or has_active_conversation
            )
            pending_messages = [*pending.messages, message][-8:]

        new_pending = PendingReply(
            message=message,
            updated_at=now,
            first_seen_at=first_seen_at,
            started_from_mention=started_from_mention,
            started_from_active_conversation=started_from_active_conversation,
            messages=pending_messages,
        )
        self._pending[key] = new_pending

        if self._signal is not None:
            self._signal(key)

        LOGGER.info(
            "pending reply updated key=%s message_id=%s updated_at=%.3f first_seen_at=%.3f started_from_mention=%s started_from_active=%s buffered=%s",
            key,
            message.id,
            now,
            first_seen_at,
            started_from_mention,
            started_from_active_conversation,
            len(pending_messages),
        )

        return key, new_pending

    def get(self, key: ConversationKey) -> PendingReply | None:
        return self._pending.get(key)

    def pop(self, key: ConversationKey) -> PendingReply | None:
        return self._pending.pop(key, None)

    def has(self, key: ConversationKey) -> bool:
        return key in self._pending
