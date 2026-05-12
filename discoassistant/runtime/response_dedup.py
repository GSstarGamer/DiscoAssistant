from __future__ import annotations

from collections import deque


class RecentResponseIds:
    """Bounded set of recently-seen Discord message IDs for dedup."""

    __slots__ = ("_ids",)

    def __init__(self, *, maxlen: int = 200) -> None:
        self._ids: deque[int] = deque(maxlen=maxlen)

    def add(self, message_id: int) -> None:
        self._ids.append(message_id)

    def __contains__(self, message_id: object) -> bool:
        return message_id in self._ids
