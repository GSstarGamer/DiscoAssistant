from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from time import monotonic

import discord

from discoassistant.runtime.keys import ConversationKey
from discoassistant.runtime.pending import PendingReply


LOGGER = logging.getLogger("discoassistant")


class MessageDebouncer:
    """Waits for the inbound user message stream to go idle before replying."""

    __slots__ = ("_window", "_pending_lookup", "_signals")

    def __init__(
        self,
        *,
        window_seconds: float,
        pending_lookup: Callable[[ConversationKey], PendingReply | None],
    ) -> None:
        self._window = window_seconds
        self._pending_lookup = pending_lookup
        self._signals: dict[ConversationKey, asyncio.Event] = {}

    def signal(self, key: ConversationKey) -> None:
        signal = self._signals.setdefault(key, asyncio.Event())
        signal.set()

    def discard(self, key: ConversationKey) -> None:
        self._signals.pop(key, None)

    async def wait_until_idle(self, key: ConversationKey) -> discord.Message | None:
        signal = self._signals.setdefault(key, asyncio.Event())
        signal.clear()

        while True:
            pending = self._pending_lookup(key)
            if pending is None:
                return None

            remaining = (pending.updated_at + self._window) - monotonic()
            if remaining <= 0:
                LOGGER.info(
                    "debounce ready key=%s message_id=%s",
                    key,
                    pending.message.id,
                )
                return pending.message

            LOGGER.debug(
                "debounce wait key=%s current_message_id=%s remaining=%.3f",
                key,
                pending.message.id,
                remaining,
            )
            try:
                await asyncio.wait_for(signal.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                pending = self._pending_lookup(key)
                if pending is None:
                    return None
                LOGGER.info(
                    "debounce ready (timeout) key=%s message_id=%s",
                    key,
                    pending.message.id,
                )
                return pending.message
            signal.clear()
