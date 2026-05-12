from __future__ import annotations

import asyncio
from typing import Any

import discord

from discoassistant.runtime.keys import ConversationKey


class TypingHeartbeat:
    """Keeps the Discord typing indicator alive while a reply is in flight."""

    __slots__ = ("channel", "_stop")

    def __init__(self, channel: Any) -> None:
        self.channel = channel
        self._stop = asyncio.Event()

    async def run(self) -> None:
        while not self._stop.is_set():
            try:
                await self.channel.trigger_typing()
            except (discord.HTTPException, AttributeError, asyncio.CancelledError):
                if self._stop.is_set():
                    return
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=8.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                return

    def stop(self) -> None:
        self._stop.set()


class TypingHeartbeatRegistry:
    """Tracks running typing heartbeats keyed by conversation."""

    __slots__ = ("_heartbeats", "_tasks")

    def __init__(self) -> None:
        self._heartbeats: dict[ConversationKey, TypingHeartbeat] = {}
        self._tasks: dict[ConversationKey, asyncio.Task[None]] = {}

    def ensure_running(self, key: ConversationKey, channel: Any) -> None:
        if key in self._heartbeats:
            return
        if not hasattr(channel, "trigger_typing"):
            return
        heartbeat = TypingHeartbeat(channel)
        self._heartbeats[key] = heartbeat
        self._tasks[key] = asyncio.create_task(heartbeat.run())

    def stop(self, key: ConversationKey) -> None:
        heartbeat = self._heartbeats.pop(key, None)
        if heartbeat is not None:
            heartbeat.stop()
        task = self._tasks.pop(key, None)
        if task is not None and not task.done():
            task.cancel()

    def cancel_all(self) -> None:
        for heartbeat in list(self._heartbeats.values()):
            heartbeat.stop()
        for task in list(self._tasks.values()):
            if not task.done():
                task.cancel()
