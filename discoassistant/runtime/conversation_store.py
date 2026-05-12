from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from time import monotonic

import discord

from discoassistant.runtime.keys import ConversationKey, conversation_key


@dataclass(slots=True)
class ActiveConversation:
    messages: list[dict[str, str]] = field(default_factory=list)
    expires_at: float = 0.0
    interrupted_by_other_user: bool = False


class ConversationStore:
    """Tracks recent assistant exchanges keyed by (channel_id, author_id)."""

    __slots__ = ("_window", "_messages_cap", "_conversations")

    def __init__(self, *, window_seconds: float, messages_cap: int = 12) -> None:
        self._window = window_seconds
        self._messages_cap = messages_cap
        self._conversations: dict[ConversationKey, ActiveConversation] = {}

    def get(self, key: ConversationKey) -> ActiveConversation | None:
        conversation = self._conversations.get(key)
        if conversation is None:
            return None
        if conversation.expires_at <= monotonic():
            del self._conversations[key]
            return None
        return conversation

    def keys_for_channel(self, channel_id: int) -> list[ConversationKey]:
        keys: list[ConversationKey] = []
        now = monotonic()
        for key, conversation in list(self._conversations.items()):
            if key[0] != channel_id:
                continue
            if conversation.expires_at <= now:
                del self._conversations[key]
                continue
            keys.append(key)
        return keys

    def store(
        self,
        key: ConversationKey,
        *,
        prior_messages: list[dict[str, str]] | None,
        user_message: dict[str, str],
        assistant_reply: str,
    ) -> None:
        conversation_messages: list[dict[str, str]] = []
        if prior_messages:
            conversation_messages.extend(prior_messages)

        conversation_messages.append(user_message)
        conversation_messages.append({"role": "assistant", "content": assistant_reply})
        conversation_messages = conversation_messages[-self._messages_cap :]

        self._conversations[key] = ActiveConversation(
            messages=conversation_messages,
            expires_at=monotonic() + self._window,
            interrupted_by_other_user=False,
        )

    def mark_interrupted(
        self,
        intruder_message: discord.Message,
        *,
        author_display_name: str,
    ) -> None:
        """Append a passive context note to all active conversations in the
        intruder's channel that don't belong to the intruder.
        """
        content = intruder_message.content.strip()
        if not content:
            return

        channel_id = intruder_message.channel.id
        author_id = intruder_message.author.id

        for key in self.keys_for_channel(channel_id):
            if key[1] == author_id:
                continue

            conversation = self._conversations.get(key)
            if conversation is None:
                continue

            conversation.messages.append(
                {
                    "role": "user",
                    "content": (
                        "Channel context update from another user. This message was not directed at you, "
                        "but happened in same channel during active conversation.\n"
                        f"Author username: {intruder_message.author}\n"
                        f"Author display name: {author_display_name}\n"
                        f"Channel message: {content}"
                    ),
                }
            )
            conversation.messages = conversation.messages[-self._messages_cap :]
            conversation.expires_at = monotonic() + self._window
            conversation.interrupted_by_other_user = True
