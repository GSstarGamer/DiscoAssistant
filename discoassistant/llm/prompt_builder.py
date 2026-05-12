from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import discord


LOGGER = logging.getLogger("discoassistant")


@dataclass
class BuiltPrompt:
    messages: list[dict[str, Any]]
    user_message: dict[str, Any] | None
    extra_payload: dict[str, Any]


class ReplyPromptBuilder:
    """Assembles the LLM request for a single reply turn.

    The builder owns the ordering and shape of the OpenRouter `messages` array
    plus the per-turn `extra_payload` (temperature, max tokens, tool wiring).

    Discord-specific data extraction stays on the bot via the callback bag —
    this keeps the builder free of `DiscoAssistant` imports while preserving
    the exact prompt structure historically produced by `generate_reply_for_message`.
    """

    def __init__(
        self,
        *,
        static_system_prompt: str,
        memory_enabled: bool,
        owner_user_id: int,
        agent_for_message: Callable[[discord.Message], Any],
        owner_context_prompt: Callable[[discord.Message], Awaitable[str]],
        assistant_identity_prompt: Callable[[], str],
        is_direct_message: Callable[[discord.Message], bool],
        is_mention_without_other_text: Callable[[discord.Message], bool],
        get_active_conversation: Callable[[discord.Message], Any],
        should_prefetch_channel_context: Callable[[discord.Message], bool],
        prefetch_channel_history_for_message: Callable[[discord.Message], Awaitable[str]],
        user_memory_context_for_message: Callable[[discord.Message], str],
        server_memory_context_for_message: Callable[[discord.Message], str],
        pending_burst_context: Callable[[list[discord.Message] | None, int], str],
        reply_reference_context: Callable[[discord.Message], Awaitable[str]],
        passive_flush_confirmation_context: Callable[[discord.Message], str],
        dm_conversation_context_for_message: Callable[[discord.Message], list[dict[str, Any]]],
        dm_summary_block_for_user: Callable[[int], str],
        message_text_for_context: Callable[[discord.Message], str],
        display_name_for_message_author: Callable[[discord.Message], str],
        mentioned_users_context: Callable[[discord.Message], str],
        mentioned_channels_context: Callable[[discord.Message], str],
        tool_schemas_for_agent: Callable[[list[str]], list[dict[str, Any]]],
        tool_choice_for_message: Callable[[discord.Message], Any],
        tool_calls_allowed: bool,
    ) -> None:
        self._static_system_prompt = static_system_prompt
        self._memory_enabled = memory_enabled
        self._owner_user_id = owner_user_id
        self._agent_for_message = agent_for_message
        self._owner_context_prompt = owner_context_prompt
        self._assistant_identity_prompt = assistant_identity_prompt
        self._is_direct_message = is_direct_message
        self._is_mention_without_other_text = is_mention_without_other_text
        self._get_active_conversation = get_active_conversation
        self._should_prefetch_channel_context = should_prefetch_channel_context
        self._prefetch_channel_history_for_message = prefetch_channel_history_for_message
        self._user_memory_context_for_message = user_memory_context_for_message
        self._server_memory_context_for_message = server_memory_context_for_message
        self._pending_burst_context = pending_burst_context
        self._reply_reference_context = reply_reference_context
        self._passive_flush_confirmation_context = passive_flush_confirmation_context
        self._dm_conversation_context_for_message = dm_conversation_context_for_message
        self._dm_summary_block_for_user = dm_summary_block_for_user
        self._message_text_for_context = message_text_for_context
        self._display_name_for_message_author = display_name_for_message_author
        self._mentioned_users_context = mentioned_users_context
        self._mentioned_channels_context = mentioned_channels_context
        self._tool_schemas_for_agent = tool_schemas_for_agent
        self._tool_choice_for_message = tool_choice_for_message
        self._tool_calls_allowed = tool_calls_allowed

    async def build(
        self,
        *,
        message: discord.Message,
        pending_messages: list[discord.Message] | None,
    ) -> BuiltPrompt:
        selected_agent = self._agent_for_message(message)
        owner_context_prompt = await self._owner_context_prompt(message)
        is_dm = self._is_direct_message(message)
        conversation = None if is_dm else self._get_active_conversation(message)
        mention_only = self._is_mention_without_other_text(message)
        prefetched_channel_context = ""
        if not is_dm and (mention_only or self._should_prefetch_channel_context(message)):
            prefetched_channel_context = await self._prefetch_channel_history_for_message(message)
        memory_context = self._user_memory_context_for_message(message)
        server_memory_context = "" if is_dm else self._server_memory_context_for_message(message)
        pending_burst_context = self._pending_burst_context(pending_messages, message.id)
        reply_reference_context = await self._reply_reference_context(message)
        passive_flush_confirmation_context = (
            "" if is_dm else self._passive_flush_confirmation_context(message)
        )

        # System block #1 — static prefix (cache-friendly).
        static_parts: list[str] = []
        if self._static_system_prompt:
            static_parts.append(self._static_system_prompt)
        identity = self._assistant_identity_prompt()
        if identity:
            static_parts.append(identity)
        if owner_context_prompt:
            static_parts.append(owner_context_prompt)
        if selected_agent.system_prompt:
            static_parts.append(selected_agent.system_prompt)
        static_prefix = "\n\n".join(part.strip() for part in static_parts if part.strip())

        # System block #2 — semi-static memory (cached until file mtime changes).
        memory_parts: list[str] = []
        if memory_context:
            memory_parts.append(memory_context.strip())
        if server_memory_context:
            memory_parts.append(server_memory_context.strip())
        if is_dm:
            dm_summary_block = self._dm_summary_block_for_user(message.author.id)
            if dm_summary_block:
                memory_parts.append(dm_summary_block.strip())
        memory_block = "\n\n".join(memory_parts) if memory_parts else ""

        # System block #3 — per-turn runtime context.
        runtime_lines: list[str] = []
        runtime_lines.append(
            f"Author: {self._display_name_for_message_author(message)} "
            f"(username {message.author}, id {message.author.id})"
        )
        runtime_lines.append(
            "Author is the owner."
            if message.author.id == self._owner_user_id
            else "Author is NOT the owner."
        )
        if is_dm:
            runtime_lines.append("Channel: DM. Use only user memory; server memory does not apply.")
        else:
            runtime_lines.append(
                f"Channel: guild={message.guild.id if message.guild else '?'} "
                f"channel_id={message.channel.id} "
                f"mode={'follow-up' if conversation is not None else 'new mention'} "
                f"mention_only={'yes' if mention_only else 'no'}"
            )
            mentioned_users = self._mentioned_users_context(message)
            if mentioned_users:
                runtime_lines.append(mentioned_users.strip())
            mentioned_channels = self._mentioned_channels_context(message)
            if mentioned_channels:
                runtime_lines.append(mentioned_channels.strip())
        if reply_reference_context:
            runtime_lines.append(reply_reference_context.strip())
        if pending_burst_context:
            runtime_lines.append(pending_burst_context.strip())
        if prefetched_channel_context:
            runtime_lines.append(prefetched_channel_context.strip())
        if passive_flush_confirmation_context:
            runtime_lines.append(passive_flush_confirmation_context.strip())
        runtime_block = "\n".join(line for line in runtime_lines if line)

        messages: list[dict[str, Any]] = []
        if static_prefix:
            messages.append({"role": "system", "content": static_prefix})
        if memory_block:
            messages.append({"role": "system", "content": memory_block})

        if is_dm:
            messages.extend(self._dm_conversation_context_for_message(message))
        elif conversation is not None:
            messages.extend(conversation.messages)

        if runtime_block:
            messages.append({"role": "system", "content": runtime_block})

        if is_dm:
            user_message: dict[str, Any] | None = None
        else:
            user_message = {
                "role": "user",
                "content": self._message_text_for_context(message) or "(no text)",
            }
            messages.append(user_message)

        extra_payload: dict[str, Any] = {
            "temperature": selected_agent.temperature,
            "max_tokens": selected_agent.max_output_tokens,
        }
        tools = self._tool_schemas_for_agent(selected_agent.tools)
        if tools:
            extra_payload["tools"] = tools
            extra_payload["tool_choice"] = self._tool_choice_for_message(message)

        return BuiltPrompt(
            messages=messages,
            user_message=user_message,
            extra_payload=extra_payload,
        )
