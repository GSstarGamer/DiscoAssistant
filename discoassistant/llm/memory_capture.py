from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from time import monotonic
from typing import Any

import discord

from discoassistant.config import AgentDefinition
from discoassistant.llm.openrouter import OpenRouterClient
from discoassistant.llm.response import extract_response_text
from discoassistant.memory import GuildMemoryStore, UserMemoryStore


LOGGER = logging.getLogger("discoassistant")


class OwnerMemoryCapturer:
    _MEMORY_SIGNAL_KEYWORDS = (
        "remember",
        "note that",
        "save this",
        "keep in mind",
        "don't forget",
        "do not forget",
        "from now on",
        "going forward",
        "i prefer",
        "call me",
        "my ",
        "i am ",
        "i'm a",
        "i'm an",
        "forget about",
        "stop calling",
    )
    _MEMORY_SIGNAL_PATTERN = re.compile(
        r"\b(remember|forget|store|save|delete)\b.*\byou(rself)?\b",
        re.IGNORECASE,
    )

    def __init__(
        self,
        *,
        openrouter_client: OpenRouterClient,
        user_memory_store: UserMemoryStore,
        guild_memory_store: GuildMemoryStore,
        owner_user_id: int,
        owner_agent: AgentDefinition,
        memory_enabled: bool,
        message_text_for_context: Callable[[discord.Message], str],
        is_direct_message: Callable[[discord.Message], bool],
        is_mention_without_other_text: Callable[[discord.Message], bool],
        conversation_key: Callable[[discord.Message], tuple[int, int]],
        server_memory_context_for_message: Callable[[discord.Message], str],
        display_name_for_message_author: Callable[[discord.Message], str],
    ) -> None:
        self._openrouter = openrouter_client
        self._user_memory_store = user_memory_store
        self._guild_memory_store = guild_memory_store
        self._owner_user_id = owner_user_id
        self._owner_agent = owner_agent
        self._memory_enabled = memory_enabled
        self._message_text_for_context = message_text_for_context
        self._is_direct_message = is_direct_message
        self._is_mention_without_other_text = is_mention_without_other_text
        self._conversation_key = conversation_key
        self._server_memory_context_for_message = server_memory_context_for_message
        self._display_name_for_message_author = display_name_for_message_author
        self._last_owner_capture_at: dict[tuple[int, int], float] = {}

    @classmethod
    def _message_likely_carries_memory_signal(cls, text: str) -> bool:
        if not text:
            return False
        lo = text.lower()
        if any(keyword in lo for keyword in cls._MEMORY_SIGNAL_KEYWORDS):
            return True
        return cls._MEMORY_SIGNAL_PATTERN.search(text) is not None

    async def maybe_capture(
        self,
        *,
        message: discord.Message,
        reply_text: str,
        pending_messages: list[discord.Message] | None = None,
    ) -> None:
        if not self._memory_enabled:
            return
        if message.author.id != self._owner_user_id:
            return

        latest_text = self._message_text_for_context(message).strip()
        if not latest_text:
            return
        if self._is_mention_without_other_text(message):
            return

        is_dm = self._is_direct_message(message)
        carries_signal = self._message_likely_carries_memory_signal(latest_text)
        long_form_dm = is_dm and len(latest_text) >= 200
        capture_key = self._conversation_key(message)
        now = monotonic()
        last_capture_at = self._last_owner_capture_at.get(capture_key, 0.0)
        recently_captured = (now - last_capture_at) < 30.0

        if not carries_signal and not long_form_dm:
            LOGGER.info(
                "owner memory capture skipped (no signal) key=%s text_chars=%s",
                capture_key,
                len(latest_text),
            )
            return
        if recently_captured and not carries_signal:
            LOGGER.info("owner memory capture skipped (recent capture) key=%s", capture_key)
            return

        self._last_owner_capture_at[capture_key] = now

        transcript = self._memory_capture_transcript(
            message=message,
            pending_messages=pending_messages,
        )
        memory_context = self._user_memory_store.read_for_prompt(message.author.id)
        server_memory_context = self._server_memory_context_for_message(message)
        messages = [
            {
                "role": "system",
                "content": (
                    "You extract durable memory from the owner's latest Discord messages.\n"
                    "Return strict JSON only.\n"
                    "Decide whether any reusable long-term note should be saved.\n"
                    "Use user memory only for important personal facts about the owner: identity facts, durable relationships, long-term projects, stable preferences, important plans, or other biographical facts likely to matter across servers.\n"
                    "Do not put casual style cues, small interaction habits, lightweight collaboration preferences, or routine chat details into user memory.\n"
                    "Save those casual or local details to server memory instead when they are useful in this guild.\n"
                    "Save server-wide rules, shared norms, channel habits, shared in-jokes, guild-specific behavior instructions, and most casual reusable context to server memory.\n"
                    "If the owner is speaking in a guild channel and tells the assistant how to behave toward a member, role, topic, or conversation in that server, store that in server memory, not user memory.\n"
                    "Examples that belong in server memory: 'remember to be colder to X here', 'don't joke about Y in this server', 'call Joey by that nickname here'.\n"
                    "Do not save one-off chatter, transient tasks, or details already present in memory.\n"
                    "Do not duplicate a note into both user and server memory.\n"
                    "JSON schema:\n"
                    "{\"user_memory_append\": [string], \"server_memory_append\": [string]}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Owner user id: {self._owner_user_id}\n"
                    f"Channel type: {'dm' if self._is_direct_message(message) else 'guild'}\n"
                    f"Assistant reply: {reply_text or '(none)'}\n"
                    f"{memory_context}"
                    f"{server_memory_context}"
                    "Latest owner message batch:\n"
                    f"{transcript}\n"
                    "Return JSON only."
                ),
            },
        ]

        try:
            response = await self._openrouter.chat(
                model=self._owner_agent.model,
                messages=messages,
                extra_payload={
                    "temperature": 0,
                    "max_tokens": 250,
                },
            )
            payload = self._parse_memory_capture_payload(extract_response_text(response, messages=messages))
        except Exception:
            LOGGER.exception("Owner memory capture failed for message_id=%s", message.id)
            return

        user_notes = self._dedupe_memory_notes(
            payload.get("user_memory_append"),
            existing_memory=self._user_memory_store.read_for_user(message.author.id),
        )
        server_notes: list[str] = []
        if message.guild is not None:
            server_notes = self._dedupe_memory_notes(
                payload.get("server_memory_append"),
                existing_memory=self._guild_memory_store.read_for_guild(message.guild.id),
            )
            rerouted_user_notes: list[str] = []
            kept_user_notes: list[str] = []
            for note in user_notes:
                if self._should_route_owner_note_to_server(message=message, note=note) or not self._is_important_user_fact(note):
                    rerouted_user_notes.append(note)
                else:
                    kept_user_notes.append(note)
            user_notes = kept_user_notes
            if rerouted_user_notes:
                server_existing = self._guild_memory_store.read_for_guild(message.guild.id)
                server_notes = self._dedupe_memory_notes(
                    [*server_notes, *rerouted_user_notes],
                    existing_memory=server_existing,
                )

        for note in user_notes:
            path = self._user_memory_store.append_for_user(
                user_id=message.author.id,
                note=note,
                author_display_name=self._display_name_for_message_author(message),
                source_channel_id=getattr(message.channel, "id", None),
            )
            LOGGER.info("Owner memory captured user_id=%s path=%s note=%r", message.author.id, path, note)

        for note in server_notes:
            path = self._guild_memory_store.append_for_guild(
                guild_id=message.guild.id,
                note=note,
                guild_name=message.guild.name,
                author_display_name=self._display_name_for_message_author(message),
                source_channel_id=getattr(message.channel, "id", None),
                owner_priority=True,
            )
            LOGGER.info("Owner server memory captured guild_id=%s path=%s note=%r", message.guild.id, path, note)

    def _memory_capture_transcript(
        self,
        *,
        message: discord.Message,
        pending_messages: list[discord.Message] | None = None,
    ) -> str:
        relevant_messages = pending_messages or [message]
        lines: list[str] = []
        for item in relevant_messages[-8:]:
            author_name = self._display_name_for_message_author(item)
            content = self._message_text_for_context(item).strip() or "(no text)"
            lines.append(f"{author_name}: {content}")
        return "\n".join(lines)

    @staticmethod
    def _parse_memory_capture_payload(raw_text: str) -> dict[str, list[str]]:
        text = raw_text.strip()
        if not text:
            return {"user_memory_append": [], "server_memory_append": []}

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end >= start:
            text = text[start : end + 1]

        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("Memory capture payload must be a JSON object.")

        return {
            "user_memory_append": payload.get("user_memory_append", []),
            "server_memory_append": payload.get("server_memory_append", []),
        }

    def _dedupe_memory_notes(self, raw_notes: Any, *, existing_memory: str) -> list[str]:
        if not isinstance(raw_notes, list):
            return []

        notes: list[str] = []
        seen: set[str] = set()
        for value in raw_notes:
            if not isinstance(value, str):
                continue
            note = self._normalize_memory_note(value)
            if not note:
                continue
            normalized = note.casefold()
            if normalized in seen:
                continue
            if normalized in existing_memory.casefold():
                continue
            seen.add(normalized)
            notes.append(note)
        return notes

    @staticmethod
    def _normalize_memory_note(value: str) -> str:
        note = value.strip()
        note = re.sub(r"^[-*]\s+", "", note)
        note = re.sub(r"^(?:\[[^\]]+\]\s*)+", "", note).strip()
        note = re.sub(r"^[-*]\s+", "", note)
        return note.strip()

    def _should_route_owner_note_to_server(self, *, message: discord.Message, note: str) -> bool:
        if message.guild is None:
            return False

        note_lower = note.casefold()
        message_text = self._message_text_for_context(message).casefold()
        has_member_reference = bool(re.search(r"<@!?\d+>", note)) or bool(getattr(message, "mentions", []))
        behavior_keywords = (
            "treat ",
            "be cold",
            "be distant",
            "don't be ",
            "do not be ",
            "remember to ",
            "call ",
            "avoid ",
            "never mention ",
            "nickname",
        )
        server_scope_keywords = (
            "server",
            "guild",
            "channel",
            "here",
            "in this chat",
            "in here",
        )

        if any(keyword in message_text for keyword in server_scope_keywords):
            return True
        if has_member_reference and any(keyword in note_lower for keyword in behavior_keywords):
            return True
        if message_text.startswith("remember ") and has_member_reference:
            return True
        return False

    @staticmethod
    def _is_important_user_fact(note: str) -> bool:
        note_lower = note.casefold()
        important_markers = (
            "is ",
            "works at ",
            "lives in ",
            "birthday",
            "pronouns",
            "allergic",
            "medical",
            "project",
            "building ",
            "studying ",
            "dating ",
            "married ",
            "family",
            "prefers ",
            "favorite ",
            "goal",
            "plan",
        )
        casual_markers = (
            "tone",
            "style",
            "short replies",
            "long replies",
            "formatting",
            "joke",
            "banter",
            "call them",
            "nickname",
            "be cold",
            "be distant",
            "don't be ",
            "do not be ",
            "remember to ",
        )
        if any(marker in note_lower for marker in casual_markers):
            return False
        return any(marker in note_lower for marker in important_markers)
