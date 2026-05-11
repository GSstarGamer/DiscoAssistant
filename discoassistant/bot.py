from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from time import monotonic
from typing import Any
from collections.abc import Iterable

import aiohttp
import discord

from discoassistant.config import AppConfig, load_app_config


LOGGER = logging.getLogger("discoassistant")


@dataclass(slots=True)
class ActiveConversation:
    messages: list[dict[str, str]] = field(default_factory=list)
    expires_at: float = 0.0


class DiscoAssistant(discord.Client):
    def __init__(self, app_config: AppConfig) -> None:
        super().__init__()
        self.app_config = app_config
        self._startup_announced = False
        self.http_session: aiohttp.ClientSession | None = None
        self._recent_response_ids: deque[int] = deque(maxlen=200)
        self._active_conversations: dict[tuple[int, int], ActiveConversation] = {}

    async def setup_hook(self) -> None:
        headers = {
            "Authorization": f"Bearer {self.app_config.settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "X-Title": self.app_config.runtime.openrouter.app_name,
        }
        if self.app_config.runtime.openrouter.site_url:
            headers["HTTP-Referer"] = self.app_config.runtime.openrouter.site_url

        self.http_session = aiohttp.ClientSession(
            headers=headers
        )
        LOGGER.info(
            "setup_hook complete. Async HTTP session created for model %s.",
            self.app_config.runtime.openrouter.default_model,
        )

    async def on_ready(self) -> None:
        if self._startup_announced:
            LOGGER.info("Reconnected as %s (%s)", self.user, self.user.id if self.user else "unknown")
            return

        self._startup_announced = True
        LOGGER.info("Logged in as %s (%s)", self.user, self.user.id if self.user else "unknown")
        LOGGER.info("Watching %s guild(s).", len(self.guilds))
        LOGGER.info(
            "Loaded %s agent definitions. Default model is %s.",
            len(self.app_config.runtime.agents),
            self.app_config.runtime.openrouter.default_model,
        )
        print(f"DiscoAssistant is ready as {self.user}")

    async def on_message(self, message: discord.Message) -> None:
        should_respond = self._should_respond_to_message(message)
        if not should_respond and self._has_active_conversation(message):
            should_respond = await self._should_treat_active_conversation_message_as_follow_up(message)

        if not should_respond:
            return

        try:
            self._recent_response_ids.append(message.id)
            async with message.channel.typing():
                reply_text = await self.generate_reply_for_message(message)
            await message.channel.send(reply_text, reference=message)
        except Exception:
            LOGGER.exception("Failed to generate or send reply for message %s", message.id)
            await message.channel.send(
                "I am DiscoAssistant, a Discord assistant. Something went wrong while I was thinking."
            )

    async def close(self) -> None:
        if self.http_session is not None and not self.http_session.closed:
            await self.http_session.close()
        await super().close()

    async def openrouter_chat(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, str]],
        extra_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.http_session is None:
            raise RuntimeError("HTTP session has not been created yet.")

        payload: dict[str, Any] = {
            "model": model or self.app_config.runtime.openrouter.default_model,
            "messages": messages,
        }
        if extra_payload:
            payload.update(extra_payload)

        async with self.http_session.post(
            f"{self.app_config.runtime.openrouter.base_url}/chat/completions",
            json=payload,
        ) as response:
            response.raise_for_status()
            return await response.json()

    def _should_respond_to_message(self, message: discord.Message) -> bool:
        if self.user is None:
            return False

        if message.id in self._recent_response_ids:
            return False

        if message.author.id == self.user.id:
            return False

        if message.author.bot and not self.app_config.runtime.discord.respond_to_bots:
            return False

        return any(user.id == self.user.id for user in self._iter_mentions(message))

    @staticmethod
    def _iter_mentions(message: discord.Message) -> Iterable[discord.abc.User]:
        return getattr(message, "mentions", [])

    async def generate_reply_for_message(self, message: discord.Message) -> str:
        selected_agent = self._agent_for_message(message)
        prompt_parts = [
            self.app_config.runtime.prompts.get("shared_base", ""),
            self.app_config.runtime.prompts.get("response_style", ""),
            self.app_config.runtime.prompts.get("tool_rules", ""),
            self.app_config.runtime.prompts.get("safety", ""),
            selected_agent.system_prompt,
        ]
        system_prompt = "\n\n".join(part for part in prompt_parts if part)
        conversation = self._get_active_conversation(message)
        mention_only = self._is_mention_without_other_text(message)
        prefetched_channel_context = ""
        if mention_only:
            prefetched_channel_context = await self._prefetch_channel_history_for_message(message)

        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        if conversation is not None:
            messages.extend(conversation.messages)

        user_message = {
            "role": "user",
            "content": (
                "Respond to this Discord message as DiscoAssistant.\n"
                f"Conversation mode: {'follow-up' if conversation is not None else 'new mention'}\n"
                f"Mention-only message: {'yes' if mention_only else 'no'}\n"
                f"Author username: {message.author}\n"
                f"Author display name: {self._display_name_for_message_author(message)}\n"
                f"Channel message: {message.content.strip()}\n"
                "If this is a mention-only message, do not send a generic greeting. Use recent channel history to infer what the user is asking and answer that directly.\n"
                f"{prefetched_channel_context}"
                "Keep it short."
            ),
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

        response = await self._run_chat_with_tools_if_needed(
            model=selected_agent.model,
            messages=messages,
            extra_payload=extra_payload,
            tool_context={
                "message": message,
            },
        )
        reply_text = self.extract_response_text(response)
        self._store_active_conversation(
            message=message,
            prior_conversation=conversation,
            user_message=user_message,
            assistant_reply=reply_text,
        )
        return reply_text

    async def _run_chat_with_tools_if_needed(
        self,
        *,
        model: str | None,
        messages: list[dict[str, Any]],
        extra_payload: dict[str, Any],
        tool_context: dict[str, Any],
    ) -> dict[str, Any]:
        current_payload = dict(extra_payload)
        response = await self.openrouter_chat(
            model=model,
            messages=messages,
            extra_payload=current_payload,
        )
        max_calls = self.app_config.runtime.tool_policy.max_calls_per_turn
        tool_calls_used = 0

        while tool_calls_used < max_calls:
            assistant_message = response.get("choices", [{}])[0].get("message", {})
            tool_calls = assistant_message.get("tool_calls", [])
            if not tool_calls:
                return response

            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_message.get("content") or "",
                    "tool_calls": tool_calls,
                }
            )

            current_payload["tool_choice"] = "auto"
            for tool_call in tool_calls:
                tool_calls_used += 1
                tool_result = await self._execute_tool_call(tool_call, tool_context)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": tool_call["function"]["name"],
                        "content": json.dumps(tool_result),
                    }
                )
                if tool_calls_used >= max_calls:
                    break

            response = await self.openrouter_chat(
                model=model,
                messages=messages,
                extra_payload=current_payload,
            )

        current_payload["tool_choice"] = "none"
        return await self.openrouter_chat(
            model=model,
            messages=messages,
            extra_payload=current_payload,
        )

    async def _should_treat_active_conversation_message_as_follow_up(
        self,
        message: discord.Message,
    ) -> bool:
        conversation = self._get_active_conversation(message)
        if conversation is None:
            return False

        transcript_lines: list[str] = []
        for item in conversation.messages[-6:]:
            role = item.get("role", "unknown")
            content = item.get("content", "").strip()
            if content:
                transcript_lines.append(f"{role.upper()}: {content}")

        response = await self.openrouter_chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You classify whether a same-user Discord message is a follow-up to an assistant "
                        "conversation that happened seconds ago. Answer only YES or NO. "
                        "Answer YES only if the new message is likely still directed at the assistant. "
                        "Answer NO if the user is more likely talking to someone else in channel, making a general remark, or the message is too detached from the assistant exchange."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Recent assistant conversation:\n"
                        + ("\n".join(transcript_lines) if transcript_lines else "(none)")
                        + "\n\n"
                        f"New channel message from same user: {message.content.strip()}\n\n"
                        "Is this message still directed at the assistant?"
                    ),
                },
            ],
            extra_payload={
                "temperature": 0,
                "max_tokens": 3,
            },
        )
        decision = self.extract_response_text(response).strip().upper()
        return decision.startswith("YES")

    async def _execute_tool_call(
        self,
        tool_call: dict[str, Any],
        tool_context: dict[str, Any],
    ) -> dict[str, Any]:
        function_payload = tool_call.get("function", {})
        function_name = function_payload.get("name", "")
        raw_arguments = function_payload.get("arguments", "{}")
        arguments = json.loads(raw_arguments)

        if function_name == "get_channel_history":
            return await self._tool_get_channel_history(arguments, tool_context)

        raise RuntimeError(f"Unsupported tool call: {function_name}")

    async def _tool_get_channel_history(
        self,
        arguments: dict[str, Any],
        tool_context: dict[str, Any],
    ) -> dict[str, Any]:
        message: discord.Message = tool_context["message"]
        limit = int(arguments.get("limit", self.app_config.runtime.discord.max_history_messages))
        limit = max(1, min(limit, 100))
        before_message_id = arguments.get("before_message_id", message.id)

        before_message = discord.Object(id=int(before_message_id))
        history = []
        async for item in message.channel.history(limit=limit, before=before_message, oldest_first=False):
            history.append(
                {
                    "message_id": item.id,
                    "author_username": str(item.author),
                    "author_display_name": self._display_name_for_message_author(item),
                    "content": item.content,
                    "created_at": item.created_at.isoformat(),
                }
            )

        return {
            "channel_id": message.channel.id,
            "fetched_count": len(history),
            "messages": history,
        }

    async def _prefetch_channel_history_for_message(self, message: discord.Message) -> str:
        history_payload = await self._tool_get_channel_history({}, {"message": message})
        history_messages = history_payload.get("messages", [])
        if not history_messages:
            return "Recent channel context:\n(none)\n"

        lines = ["Recent channel context:"]
        for item in reversed(history_messages):
            author_name = item.get("author_display_name") or item.get("author_username") or "unknown"
            content = (item.get("content") or "").strip() or "(no text)"
            lines.append(f"- {author_name}: {content}")
        lines.append("")
        return "\n".join(lines)

    def _tool_schemas_for_agent(self, allowed_tool_names: list[str]) -> list[dict[str, Any]]:
        if not self.app_config.runtime.tool_policy.allow_model_tool_calls:
            return []

        enabled_tools = {
            tool.name
            for tool in self.app_config.runtime.tool_policy.registry
            if tool.enabled
        }
        allowed = enabled_tools.intersection(allowed_tool_names)
        schemas: list[dict[str, Any]] = []

        if "get_channel_history" in allowed:
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": "get_channel_history",
                        "description": (
                            "Fetch past messages from current Discord channel for context. "
                            "Use default limit unless more context is needed. To paginate farther "
                            "back, call again with a smaller before_message_id from an older message."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "limit": {
                                    "type": "integer",
                                    "description": (
                                        "Number of past channel messages to fetch. "
                                        "Discord default-style history size is fine unless more context is needed."
                                    ),
                                    "minimum": 1,
                                    "maximum": 100,
                                },
                                "before_message_id": {
                                    "type": "integer",
                                    "description": (
                                        "Fetch messages before this message id. "
                                        "Use returned older message ids to paginate farther back."
                                    ),
                                },
                            },
                            "required": [],
                            "additionalProperties": False,
                        },
                    },
                }
            )

        return schemas

    def _tool_choice_for_message(self, message: discord.Message) -> str | dict[str, Any]:
        if self._is_mention_without_other_text(message):
            return {
                "type": "function",
                "function": {
                    "name": "get_channel_history",
                },
            }
        return "auto"

    def _has_active_conversation(self, message: discord.Message) -> bool:
        return self._get_active_conversation(message) is not None

    def _get_active_conversation(self, message: discord.Message) -> ActiveConversation | None:
        key = self._conversation_key(message)
        conversation = self._active_conversations.get(key)
        if conversation is None:
            return None

        if conversation.expires_at <= monotonic():
            del self._active_conversations[key]
            return None

        return conversation

    def _store_active_conversation(
        self,
        *,
        message: discord.Message,
        prior_conversation: ActiveConversation | None,
        user_message: dict[str, str],
        assistant_reply: str,
    ) -> None:
        conversation_messages: list[dict[str, str]] = []
        if prior_conversation is not None:
            conversation_messages.extend(prior_conversation.messages)

        conversation_messages.append(user_message)
        conversation_messages.append({"role": "assistant", "content": assistant_reply})
        conversation_messages = conversation_messages[-12:]

        self._active_conversations[self._conversation_key(message)] = ActiveConversation(
            messages=conversation_messages,
            expires_at=monotonic() + self.app_config.runtime.discord.conversation_window_seconds,
        )

    @staticmethod
    def _conversation_key(message: discord.Message) -> tuple[int, int]:
        return (message.channel.id, message.author.id)

    def _is_mention_without_other_text(self, message: discord.Message) -> bool:
        if self.user is None:
            return False

        content = message.content
        mention_forms = {
            self.user.mention,
            f"<@{self.user.id}>",
            f"<@!{self.user.id}>",
        }
        for mention in mention_forms:
            content = content.replace(mention, " ")

        return not content.strip()

    def _agent_for_message(self, message: discord.Message):
        if message.author.id == self.app_config.settings.owner_user_id:
            return self.app_config.runtime.agents["owner"]
        return self.app_config.runtime.agents["public"]

    @staticmethod
    def extract_response_text(response: dict[str, Any]) -> str:
        choices = response.get("choices", [])
        if not choices:
            raise RuntimeError("OpenRouter response did not include any choices.")

        message = choices[0].get("message", {})
        content = message.get("content", "")

        if isinstance(content, str):
            text = content.strip()
            if text:
                return text

        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_value = item.get("text", "").strip()
                    if text_value:
                        text_parts.append(text_value)
            if text_parts:
                return "\n".join(text_parts)

        raise RuntimeError("OpenRouter response did not include usable text content.")

    @staticmethod
    def _display_name_for_message_author(message: discord.Message) -> str:
        author = message.author
        return getattr(author, "display_name", str(author))


bot: DiscoAssistant | None = None


def main() -> None:
    global bot

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    app_config = load_app_config()
    bot = DiscoAssistant(app_config)
    bot.run(app_config.settings.discord_token)
