from __future__ import annotations

from typing import Any


ConversationKey = tuple[int, int]


def conversation_key(message: Any) -> ConversationKey:
    return (message.channel.id, message.author.id)
