from __future__ import annotations

import asyncio

from discoassistant.runtime.keys import ConversationKey


class InFlightTaskRegistry:
    """Tracks the currently-running reply task per conversation."""

    __slots__ = ("_reply_tasks", "_generating")

    def __init__(self) -> None:
        self._reply_tasks: dict[ConversationKey, asyncio.Task[None]] = {}
        self._generating: set[ConversationKey] = set()

    def get(self, key: ConversationKey) -> asyncio.Task[None] | None:
        return self._reply_tasks.get(key)

    def set(self, key: ConversationKey, task: asyncio.Task[None]) -> None:
        self._reply_tasks[key] = task

    def mark_generating(self, key: ConversationKey) -> None:
        self._generating.add(key)

    def is_generating(self, key: ConversationKey) -> bool:
        return key in self._generating

    def clear_generating(self, key: ConversationKey) -> None:
        self._generating.discard(key)

    def cleanup_if_owner(
        self,
        key: ConversationKey,
        current_task: asyncio.Task[None] | None,
    ) -> bool:
        if self._reply_tasks.get(key) is current_task:
            self._reply_tasks.pop(key, None)
            return True
        return False

    def cancel_all(self) -> None:
        for task in self._reply_tasks.values():
            if not task.done():
                task.cancel()
