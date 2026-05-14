from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from time import monotonic
from typing import Any
from collections.abc import Iterable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import aiohttp
import discord
from bs4 import BeautifulSoup

from discoassistant.config import BASE_DIR, AppConfig, load_app_config
from discoassistant.dm_history import DmHistoryStore
from discoassistant.memory import GuildMemoryStore, UserMemoryStore


LOGGER = logging.getLogger("discoassistant")


def _strip_html_text(raw: str, max_chars: int) -> str:
    soup = BeautifulSoup(raw, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "form"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    if len(text) > max_chars:
        text = text[:max_chars]
    return text


@dataclass(slots=True)
class ActiveConversation:
    messages: list[dict[str, str]] = field(default_factory=list)
    expires_at: float = 0.0
    interrupted_by_other_user: bool = False


@dataclass(slots=True)
class PendingReply:
    message: discord.Message
    updated_at: float
    first_seen_at: float
    started_from_mention: bool
    started_from_active_conversation: bool
    messages: list[discord.Message] = field(default_factory=list)


class DiscoAssistant(discord.Client):
    def __init__(self, app_config: AppConfig) -> None:
        super().__init__()
        self.app_config = app_config
        self._startup_announced = False
        self.http_session: aiohttp.ClientSession | None = None
        self._owner_context_prompt_cache: str | None = None
        self._recent_response_ids: deque[int] = deque(maxlen=200)
        self._active_conversations: dict[tuple[int, int], ActiveConversation] = {}
        self._pending_replies: dict[tuple[int, int], PendingReply] = {}
        self._reply_tasks: dict[tuple[int, int], asyncio.Task[None]] = {}
        self._reply_generation_in_progress: set[tuple[int, int]] = set()
        self._session_total_tokens_used = 0
        self._presence_update_task: asyncio.Task[None] | None = None
        self._presence_refresh_requested = False
        self.user_memory_store = UserMemoryStore(
            base_dir=BASE_DIR / self.app_config.runtime.memory.user_directory,
            max_chars_in_prompt=self.app_config.runtime.memory.max_user_chars_in_prompt,
        )
        self.guild_memory_store = GuildMemoryStore(
            base_dir=BASE_DIR / self.app_config.runtime.memory.guild_directory,
            max_chars_in_prompt=self.app_config.runtime.memory.max_guild_chars_in_prompt,
        )
        self.dm_history_store = DmHistoryStore(
            db_path=BASE_DIR / self.app_config.runtime.memory.dm_history_db_path,
        )

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
        await self._apply_configured_presence()

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

    async def _apply_configured_presence(self) -> None:
        presence_config = self.app_config.runtime.discord.presence
        if not presence_config.enabled:
            return

        activity = self._build_configured_activity()
        status = self._configured_status()
        try:
            await self.change_presence(status=status, activity=activity)
        except Exception:
            LOGGER.exception(
                "Failed to apply configured Discord presence type=%s name=%r",
                presence_config.type,
                presence_config.name,
            )
            return

        LOGGER.info(
            "Applied configured presence type=%s status=%s name=%r",
            presence_config.type,
            presence_config.status,
            self._configured_presence_name(),
        )

    def _build_configured_activity(self) -> discord.BaseActivity | None:
        presence_config = self.app_config.runtime.discord.presence
        activity_type = presence_config.type.lower().strip()

        if activity_type == "streaming":
            if not presence_config.url:
                LOGGER.warning("Discord streaming presence enabled but no Twitch URL configured.")
                return None
            return discord.Activity(
                type=discord.ActivityType.streaming,
                name=self._configured_presence_name(),
                url=presence_config.url,
                details=self._configured_presence_details(),
                state=self._configured_presence_state(),
            )

        LOGGER.warning("Unsupported Discord presence type %r. Skipping activity.", presence_config.type)
        return None

    def _configured_status(self) -> discord.Status:
        status_value = self.app_config.runtime.discord.presence.status.lower().strip()
        status_map = {
            "online": discord.Status.online,
            "idle": discord.Status.idle,
            "dnd": discord.Status.dnd,
            "do_not_disturb": discord.Status.dnd,
            "invisible": discord.Status.invisible,
            "offline": discord.Status.invisible,
        }
        if status_value not in status_map:
            LOGGER.warning("Unsupported Discord status %r. Falling back to online.", status_value)
        return status_map.get(status_value, discord.Status.online)

    def _configured_presence_name(self) -> str:
        presence_config = self.app_config.runtime.discord.presence
        if not presence_config.token_usage_enabled:
            return presence_config.name

        try:
            return presence_config.token_usage_name_template.format(
                total_tokens=self._session_total_tokens_used,
            )
        except Exception:
            LOGGER.exception(
                "Invalid token usage presence template %r. Falling back to default name.",
                presence_config.token_usage_name_template,
            )
            return presence_config.name

    def _configured_presence_state(self) -> str | None:
        presence_config = self.app_config.runtime.discord.presence
        template = (
            presence_config.token_usage_state_template
            if presence_config.token_usage_enabled and presence_config.token_usage_state_template
            else None
        )
        if template is None:
            return presence_config.state

        try:
            return template.format(total_tokens=self._session_total_tokens_used)
        except Exception:
            LOGGER.exception(
                "Invalid token usage state template %r. Falling back to default state.",
                template,
            )
            return presence_config.state

    def _configured_presence_details(self) -> str | None:
        presence_config = self.app_config.runtime.discord.presence
        template = (
            presence_config.token_usage_details_template
            if presence_config.token_usage_enabled and presence_config.token_usage_details_template
            else None
        )
        if template is None:
            return presence_config.details

        try:
            return template.format(total_tokens=self._session_total_tokens_used)
        except Exception:
            LOGGER.exception(
                "Invalid token usage details template %r. Falling back to default details.",
                template,
            )
            return presence_config.details

    def _record_token_usage(self, response_payload: dict[str, Any]) -> None:
        usage = response_payload.get("usage")
        if not isinstance(usage, dict):
            return

        total_tokens = usage.get("total_tokens")
        if not isinstance(total_tokens, int) or total_tokens <= 0:
            return

        self._session_total_tokens_used += total_tokens
        self._schedule_presence_refresh()

    def _schedule_presence_refresh(self) -> None:
        presence_config = self.app_config.runtime.discord.presence
        if not (presence_config.enabled and presence_config.token_usage_enabled):
            return

        self._presence_refresh_requested = True
        if self._presence_update_task is not None and not self._presence_update_task.done():
            return
        self._presence_update_task = asyncio.create_task(self._flush_presence_refresh())

    async def _flush_presence_refresh(self) -> None:
        try:
            await asyncio.sleep(5)
            if not self._presence_refresh_requested:
                return
            self._presence_refresh_requested = False
            await self._apply_configured_presence()
        except asyncio.CancelledError:
            raise
        except Exception:
            LOGGER.exception("Failed while refreshing token-usage presence.")
        finally:
            self._presence_update_task = None

    @staticmethod
    def _message_text_for_context(message: discord.Message) -> str:
        parts: list[str] = []

        text = (message.content or "").strip()
        if text:
            parts.append(text)

        embed_text = DiscoAssistant._embed_text_for_context(getattr(message, "embeds", None) or [])
        if embed_text:
            parts.append(embed_text)

        if message.attachments:
            filenames = ", ".join(attachment.filename for attachment in message.attachments if attachment.filename)
            parts.append(f"[attachments: {filenames or len(message.attachments)}]")
        if getattr(message, "stickers", None):
            parts.append(f"[stickers: {len(message.stickers)}]")

        return "\n".join(part for part in parts if part).strip()

    @staticmethod
    def _embed_text_for_context(embeds: list[discord.Embed]) -> str:
        rendered_embeds: list[str] = []
        for index, embed in enumerate(embeds, start=1):
            lines: list[str] = [f"[embed {index}]"]

            author = getattr(getattr(embed, "author", None), "name", None)
            title = getattr(embed, "title", None)
            description = getattr(embed, "description", None)
            footer = getattr(getattr(embed, "footer", None), "text", None)

            if author:
                lines.append(f"author: {author}")
            if title:
                lines.append(f"title: {title}")
            if description:
                lines.append(f"description: {description}")

            for field in getattr(embed, "fields", []) or []:
                field_name = getattr(field, "name", None) or "field"
                field_value = getattr(field, "value", None) or ""
                field_value = field_value.strip()
                if field_value:
                    lines.append(f"{field_name}: {field_value}")

            if footer:
                lines.append(f"footer: {footer}")

            if len(lines) > 1:
                rendered_embeds.append("\n".join(lines))

        return "\n\n".join(rendered_embeds).strip()

    async def on_message(self, message: discord.Message) -> None:
        self._append_sibling_channel_context(message)

        if not self._should_consider_message(message):
            return

        self._store_incoming_dm_message(message)

        is_direct_mention = self._should_respond_to_message(message)
        has_active_conversation = self._has_active_conversation(message)
        has_pending_reply = self._has_pending_reply(message)
        if (
            not is_direct_mention
            and not has_active_conversation
            and not has_pending_reply
        ):
            return

        LOGGER.info(
            "tracked message id=%s channel=%s author=%s mention=%s active=%s pending=%s content=%r",
            message.id,
            message.channel.id,
            message.author.id,
            is_direct_mention,
            has_active_conversation,
            has_pending_reply,
            message.content,
        )
        self._queue_reply_attempt(message)

    async def close(self) -> None:
        for task in self._reply_tasks.values():
            if not task.done():
                task.cancel()
        if self._presence_update_task is not None and not self._presence_update_task.done():
            self._presence_update_task.cancel()
        if self.http_session is not None and not self.http_session.closed:
            await self.http_session.close()
        await super().close()

    async def openrouter_chat(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, str]],
        extra_payload: dict[str, Any] | None = None,
        max_attempts: int = 4,
    ) -> dict[str, Any]:
        if self.http_session is None:
            raise RuntimeError("HTTP session has not been created yet.")

        payload: dict[str, Any] = {
            "model": model or self.app_config.runtime.openrouter.default_model,
            "messages": messages,
        }
        if extra_payload:
            payload.update(extra_payload)

        max_attempts = max(1, int(max_attempts))
        last_error: Exception | None = None
        selected_model = model or self.app_config.runtime.openrouter.default_model
        for attempt in range(1, max_attempts + 1):
            request_started_at = monotonic()
            try:
                LOGGER.info(
                    "OpenRouter request start model=%s attempt=%s messages=%s tools=%s",
                    selected_model,
                    attempt,
                    len(messages),
                    "tools" in payload,
                )
                async with self.http_session.post(
                    f"{self.app_config.runtime.openrouter.base_url}/chat/completions",
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    response_payload = await response.json()
                    self._record_token_usage(response_payload)
                    LOGGER.info(
                        "OpenRouter request success model=%s attempt=%s duration=%.2fs",
                        selected_model,
                        attempt,
                        monotonic() - request_started_at,
                    )
                    return response_payload
            except aiohttp.ClientResponseError as exc:
                last_error = exc
                should_retry = exc.status in {429, 500, 502, 503, 504} and attempt < max_attempts
                if not should_retry:
                    LOGGER.error(
                        "OpenRouter request failed model=%s attempt=%s duration=%.2fs status=%s error=%s",
                        selected_model,
                        attempt,
                        monotonic() - request_started_at,
                        exc.status,
                        exc,
                    )
                    raise

                retry_after = self._retry_after_seconds(exc.headers)
                delay = retry_after if retry_after is not None else min(8.0, 1.5 * (2 ** (attempt - 1)))
                LOGGER.warning(
                    "OpenRouter request failed model=%s status=%s attempt=%s/%s duration=%.2fs retry_in=%.2fs",
                    selected_model,
                    exc.status,
                    attempt,
                    max_attempts,
                    monotonic() - request_started_at,
                    delay,
                )
                await asyncio.sleep(delay)
            except aiohttp.ClientError as exc:
                last_error = exc
                if attempt >= max_attempts:
                    LOGGER.error(
                        "OpenRouter network error model=%s attempt=%s duration=%.2fs error=%s",
                        selected_model,
                        attempt,
                        monotonic() - request_started_at,
                        exc,
                    )
                    raise
                delay = min(8.0, 1.0 * (2 ** (attempt - 1)))
                LOGGER.warning(
                    "OpenRouter network error model=%s attempt=%s/%s duration=%.2fs retry_in=%.2fs error=%s",
                    selected_model,
                    attempt,
                    max_attempts,
                    monotonic() - request_started_at,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)

        if last_error is not None:
            raise last_error
        raise RuntimeError("OpenRouter request failed without a captured exception.")

    def _should_respond_to_message(self, message: discord.Message) -> bool:
        if not self._should_consider_message(message):
            return False
        if message.guild is None:
            return True
        return any(user.id == self.user.id for user in self._iter_mentions(message))

    def _should_consider_message(self, message: discord.Message) -> bool:
        if self.user is None:
            return False

        if message.id in self._recent_response_ids:
            return False

        if message.author.id == self.user.id:
            return False

        if message.author.bot and not self.app_config.runtime.discord.respond_to_bots:
            return False

        return True

    @staticmethod
    def _iter_mentions(message: discord.Message) -> Iterable[discord.abc.User]:
        return getattr(message, "mentions", [])

    async def generate_reply_for_message(
        self,
        message: discord.Message,
        pending_messages: list[discord.Message] | None = None,
    ) -> str:
        LOGGER.info("generate_reply_for_message start id=%s", message.id)
        selected_agent = self._agent_for_message(message)
        owner_context_prompt = await self._owner_context_prompt(message)
        prompt_parts = [
            self.app_config.runtime.prompts.get("shared_base", ""),
            self.app_config.runtime.prompts.get("response_style", ""),
            self.app_config.runtime.prompts.get("tool_rules", ""),
            self.app_config.runtime.prompts.get("memory_rules", ""),
            self.app_config.runtime.prompts.get("safety", ""),
            self._assistant_identity_prompt(),
            self._current_time_prompt(),
            owner_context_prompt,
            self._owner_only_tools_prompt(selected_agent.name),
            selected_agent.system_prompt,
        ]
        system_prompt = "\n\n".join(part for part in prompt_parts if part)
        conversation = None if self._is_direct_message(message) else self._get_active_conversation(message)
        mention_only = self._is_mention_without_other_text(message)
        prefetched_channel_context = ""
        if mention_only or self._should_prefetch_channel_context(message):
            prefetched_channel_context = await self._prefetch_channel_history_for_message(message)
        memory_context = self._user_memory_context_for_message(message)
        server_memory_context = self._server_memory_context_for_message(message)
        pending_burst_context = self._pending_burst_context(pending_messages, latest_message_id=message.id)
        reply_reference_context = await self._reply_reference_context(message)

        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        if self._is_direct_message(message):
            messages.extend(self._dm_conversation_context_for_message(message))
        elif conversation is not None:
            messages.extend(conversation.messages)

        if self._is_direct_message(message):
            user_message = None
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Direct-message mode.\n"
                        f"Owner user id: {self.app_config.settings.owner_user_id}\n"
                        f"Author username: {message.author}\n"
                        f"Author display name: {self._display_name_for_message_author(message)}\n"
                        "Full DM chat history above is authoritative context. "
                        "Latest user message is already included there. "
                        "Use only user memory, not server memory. "
                        "When using send_dm, never guess recipient. "
                        "Never say a message was sent unless tool result explicitly says ok true. "
                        "If a tool call fails, keep working and retry when possible. "
                        "Keep reply short."
                        f"\n{reply_reference_context}"
                        f"\n{memory_context}"
                    ),
                }
            )
        else:
            user_message = {
                "role": "user",
                "content": (
                    "Respond to this Discord message naturally as the logged-in account.\n"
                    f"Conversation mode: {'follow-up' if conversation is not None else 'new mention'}\n"
                    f"Mention-only message: {'yes' if mention_only else 'no'}\n"
                    f"Owner user id: {self.app_config.settings.owner_user_id}\n"
                    f"Author username: {message.author}\n"
                    f"Author display name: {self._display_name_for_message_author(message)}\n"
                    f"{self._mentioned_users_context(message)}"
                    f"{self._mentioned_channels_context(message)}"
                    f"{reply_reference_context}"
                    f"Channel message: {self._message_text_for_context(message) or '(no text)'}\n"
                    "If this is a mention-only message, do not send a generic greeting. Use recent channel history to infer what the user is asking and answer that directly.\n"
                    "When using send_dm, never guess recipient. Use an explicit mentioned user id or the owner user id. If target is unclear, say so instead of sending.\n"
                    "Never say a message was sent unless the tool result explicitly says ok true.\n"
                    "If a tool call fails, keep working. Retry with corrected tool calls and do not answer user until tool work succeeds or is impossible after repeated retries.\n"
                    "Always answer the latest user message, not an older one. If the latest message contains multiple questions, requests, or lines, address all of them in your reply.\n"
                    f"{pending_burst_context}"
                    f"{memory_context}"
                    f"{server_memory_context}"
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
        reply_text = self.extract_response_text(response, messages=messages)
        if reply_text == "I couldn't produce a normal reply, but I can try again.":
            recovery_messages = [
                *messages,
                {
                    "role": "user",
                    "content": (
                        "Answer the user's actual question directly now in one short reply. "
                        "Do not narrate tool usage, do not ask what they want if the context already contains it, "
                        "and use fetched history silently if it is relevant."
                    ),
                },
            ]
            recovery_response = await self.openrouter_chat(
                model=selected_agent.model,
                messages=recovery_messages,
                extra_payload={
                    "temperature": selected_agent.temperature,
                    "max_tokens": selected_agent.max_output_tokens,
                },
            )
            reply_text = self.extract_response_text(recovery_response, messages=recovery_messages)
        await self._maybe_capture_owner_memory(
            message=message,
            reply_text=reply_text,
            pending_messages=pending_messages,
        )
        LOGGER.info("generate_reply_for_message done id=%s", message.id)
        if not self._is_direct_message(message):
            self._store_active_conversation(
                message=message,
                prior_conversation=conversation,
                user_message=user_message,
                assistant_reply=reply_text,
            )
        return reply_text

    def _queue_reply_attempt(self, message: discord.Message) -> None:
        key = self._conversation_key(message)
        now = monotonic()
        pending = self._pending_replies.get(key)
        is_direct_mention = self._should_respond_to_message(message)
        has_active_conversation = self._has_active_conversation(message)
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

        self._pending_replies[key] = PendingReply(
            message=message,
            updated_at=now,
            first_seen_at=first_seen_at,
            started_from_mention=started_from_mention,
            started_from_active_conversation=started_from_active_conversation,
            messages=pending_messages,
        )
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

        existing_task = self._reply_tasks.get(key)
        if existing_task is not None and not existing_task.done():
            if key in self._reply_generation_in_progress:
                LOGGER.info("cancel in-flight generation key=%s for newer message_id=%s", key, message.id)
                existing_task.cancel()
                self._reply_tasks[key] = asyncio.create_task(self._process_pending_reply(key))
                LOGGER.info("spawned replacement reply task key=%s", key)
            else:
                LOGGER.info("reuse existing debounce task key=%s latest_message_id=%s", key, message.id)
            return

        self._reply_tasks[key] = asyncio.create_task(self._process_pending_reply(key))
        LOGGER.info("spawned new reply task key=%s message_id=%s", key, message.id)

    async def _process_pending_reply(self, key: tuple[int, int]) -> None:
        try:
            LOGGER.info("reply task start key=%s", key)
            message = await self._wait_until_user_idle(key)
            if message is None:
                LOGGER.info("reply task exit key=%s reason=no_pending_message", key)
                return

            LOGGER.info("debounce finished key=%s message_id=%s", key, message.id)
            pending = self._pending_replies.get(key)
            pending_messages = pending.messages if pending is not None else [message]
            should_respond = self._should_respond_to_message(message)
            had_active_conversation = self._has_active_conversation(message)
            if not should_respond and pending is not None and pending.started_from_mention:
                LOGGER.info("reply task treating key=%s message_id=%s as pending mention follow-up", key, message.id)
                should_respond = True
            if (
                not should_respond
                and (
                    had_active_conversation
                    or (pending is not None and pending.started_from_active_conversation)
                )
            ):
                LOGGER.info(
                    "checking active-conversation classifier key=%s message_id=%s started_from_active=%s active_now=%s",
                    key,
                    message.id,
                    pending.started_from_active_conversation if pending is not None else False,
                    had_active_conversation,
                )
                should_respond = await self._should_treat_active_conversation_message_as_follow_up(message)

            if not should_respond:
                if had_active_conversation and self._should_send_not_my_conversation_reply(message):
                    LOGGER.info("reply task sending not-my-conversation notice key=%s message_id=%s", key, message.id)
                    sent_message = await self._send_channel_reply(
                        message,
                        "Not my conversation - carry on.",
                    )
                    self._store_outgoing_dm_message(sent_message)
                LOGGER.info("reply task exit key=%s message_id=%s reason=should_not_respond", key, message.id)
                return

            self._reply_generation_in_progress.add(key)
            self._recent_response_ids.append(message.id)
            LOGGER.info("start generation key=%s message_id=%s", key, message.id)
            async with message.channel.typing():
                reply_text = await self.generate_reply_for_message(message, pending_messages)
            LOGGER.info("generation done key=%s message_id=%s reply=%r", key, message.id, reply_text)
            if not reply_text.strip():
                LOGGER.info("reply suppressed key=%s message_id=%s", key, message.id)
                return
            sent_message = await self._send_channel_reply(message, reply_text)
            self._store_outgoing_dm_message(sent_message)
            LOGGER.info("reply sent key=%s message_id=%s", key, message.id)
        except asyncio.CancelledError:
            LOGGER.info("reply task cancelled key=%s", key)
            raise
        except Exception as exc:
            pending = self._pending_replies.get(key)
            message = pending.message if pending is not None else None
            if message is not None:
                LOGGER.exception("Failed to generate or send reply for message %s", message.id)
                await self._send_channel_reply(message, self._safe_runtime_error_reply(exc))
            else:
                LOGGER.exception("Failed to generate or send reply for conversation %s", key)
        finally:
            current_task = asyncio.current_task()
            self._reply_generation_in_progress.discard(key)
            if self._reply_tasks.get(key) is current_task:
                self._reply_tasks.pop(key, None)
                self._pending_replies.pop(key, None)
                LOGGER.info("reply task cleanup key=%s removed_task=yes", key)
            else:
                LOGGER.info("reply task cleanup key=%s removed_task=no", key)

    async def _wait_until_user_idle(self, key: tuple[int, int]) -> discord.Message | None:
        config = self.app_config.runtime.discord

        while True:
            pending = self._pending_replies.get(key)
            if pending is None:
                return None

            now = monotonic()
            debounce_ready_at = pending.updated_at + config.reply_debounce_seconds
            if now >= debounce_ready_at:
                LOGGER.info(
                    "debounce ready key=%s message_id=%s now=%.3f ready_at=%.3f",
                    key,
                    pending.message.id,
                    now,
                    debounce_ready_at,
                )
                return pending.message

            sleep_for = max(0.05, debounce_ready_at - now)
            LOGGER.info(
                "debounce wait key=%s current_message_id=%s sleep=%.3f",
                key,
                pending.message.id,
                sleep_for,
            )
            await asyncio.sleep(sleep_for)

    async def _send_channel_reply(self, message: discord.Message, content: str) -> discord.Message:
        try:
            return await message.channel.send(content, reference=message)
        except discord.HTTPException as exc:
            if "Unknown message" not in str(exc):
                raise
            LOGGER.warning(
                "Reply reference failed for message_id=%s channel_id=%s; retrying without reference",
                message.id,
                message.channel.id,
            )
            return await message.channel.send(content)

    def _should_send_not_my_conversation_reply(self, message: discord.Message) -> bool:
        conversation = self._get_active_conversation(message)
        if conversation is None:
            return False
        if not conversation.interrupted_by_other_user:
            return False
        return self._message_explicitly_targets_assistant(message)

    def _message_explicitly_targets_assistant(self, message: discord.Message) -> bool:
        if self._should_respond_to_message(message):
            return True
        if self.user is None:
            return False
        reference = getattr(message, "reference", None)
        resolved = getattr(reference, "resolved", None)
        if isinstance(resolved, discord.Message):
            return resolved.author.id == self.user.id
        return False

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
        unresolved_tool_failure: dict[str, Any] | None = None

        while tool_calls_used < max_calls:
            assistant_message = response.get("choices", [{}])[0].get("message", {})
            tool_calls = assistant_message.get("tool_calls", [])
            if not tool_calls:
                if unresolved_tool_failure is not None:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Previous tool call failed.\n"
                                f"Tool: {unresolved_tool_failure['name']}\n"
                                f"Error: {unresolved_tool_failure['error_message']}\n"
                                "Do not answer user yet. Retry with corrected tool call. "
                                "Only send final reply after tool succeeds."
                            ),
                        }
                    )
                    current_payload["tool_choice"] = "required"
                    response = await self.openrouter_chat(
                        model=model,
                        messages=messages,
                        extra_payload=current_payload,
                    )
                    continue
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
                try:
                    tool_result = await self._execute_tool_call(tool_call, tool_context)
                except Exception as exc:
                    LOGGER.exception(
                        "Tool call failed name=%s id=%s",
                        tool_call.get("function", {}).get("name", ""),
                        tool_call.get("id", ""),
                    )
                    tool_result = {
                        "ok": False,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                if tool_result.get("ok") is False:
                    unresolved_tool_failure = {
                        "name": tool_call["function"]["name"],
                        "error_message": str(tool_result.get("error_message", "Tool failed.")),
                    }
                elif (
                    unresolved_tool_failure is not None
                    and unresolved_tool_failure["name"] == tool_call["function"]["name"]
                ):
                    unresolved_tool_failure = None
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

        if unresolved_tool_failure is not None:
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "Couldn't complete tool action after repeated retries: "
                                f"{unresolved_tool_failure['error_message']}"
                            )
                        }
                    }
                ]
            }

        current_payload["tool_choice"] = "none"
        return await self.openrouter_chat(
            model=model,
            messages=messages,
            extra_payload=current_payload,
        )

    async def _maybe_capture_owner_memory(
        self,
        *,
        message: discord.Message,
        reply_text: str,
        pending_messages: list[discord.Message] | None = None,
    ) -> None:
        if not self.app_config.runtime.memory.enabled:
            return
        if message.author.id != self.app_config.settings.owner_user_id:
            return

        latest_text = self._message_text_for_context(message).strip()
        if not latest_text:
            return
        if self._is_mention_without_other_text(message):
            return

        transcript = self._memory_capture_transcript(
            message=message,
            pending_messages=pending_messages,
        )
        memory_context = self.user_memory_store.read_for_prompt(message.author.id)
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
                    f"Owner user id: {self.app_config.settings.owner_user_id}\n"
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
            response = await self.openrouter_chat(
                model=self.app_config.runtime.agents["owner"].model,
                messages=messages,
                extra_payload={
                    "temperature": 0,
                    "max_tokens": 250,
                },
            )
            payload = self._parse_memory_capture_payload(self.extract_response_text(response, messages=messages))
        except Exception:
            LOGGER.exception("Owner memory capture failed for message_id=%s", message.id)
            return

        user_notes = self._dedupe_memory_notes(
            payload.get("user_memory_append"),
            existing_memory=self.user_memory_store.read_for_user(message.author.id),
        )
        server_notes: list[str] = []
        if message.guild is not None:
            server_notes = self._dedupe_memory_notes(
                payload.get("server_memory_append"),
                existing_memory=self.guild_memory_store.read_for_guild(message.guild.id),
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
                server_existing = self.guild_memory_store.read_for_guild(message.guild.id)
                server_notes = self._dedupe_memory_notes(
                    [*server_notes, *rerouted_user_notes],
                    existing_memory=server_existing,
                )

        for note in user_notes:
            path = self.user_memory_store.append_for_user(
                user_id=message.author.id,
                note=note,
                author_display_name=self._display_name_for_message_author(message),
                source_channel_id=getattr(message.channel, "id", None),
            )
            LOGGER.info("Owner memory captured user_id=%s path=%s note=%r", message.author.id, path, note)

        for note in server_notes:
            path = self.guild_memory_store.append_for_guild(
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

    async def _should_treat_active_conversation_message_as_follow_up(
        self,
        message: discord.Message,
    ) -> bool:
        conversation = self._get_active_conversation(message)
        if conversation is None:
            return False

        heuristic = self._heuristic_active_follow_up_decision(message)
        if heuristic is not None:
            LOGGER.info(
                "active-conversation heuristic decided message_id=%s result=%s content=%r",
                message.id,
                heuristic,
                message.content,
            )
            return heuristic

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
                        f"New channel message from same user: {self._message_text_for_context(message) or '(no text)'}\n\n"
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
        if not decision:
            heuristic = self._heuristic_active_follow_up_decision(message)
            if heuristic is not None:
                LOGGER.info(
                    "active-conversation classifier returned no text; using heuristic message_id=%s result=%s",
                    message.id,
                    heuristic,
                )
                return heuristic
        return decision.startswith("YES")

    def _heuristic_active_follow_up_decision(self, message: discord.Message) -> bool | None:
        content = (message.content or "").strip()
        if not content:
            return None

        lowered = content.casefold()
        if lowered.endswith("?"):
            return True

        follow_up_starters = (
            "what",
            "why",
            "how",
            "when",
            "where",
            "who",
            "which",
            "can",
            "could",
            "would",
            "should",
            "do",
            "does",
            "did",
            "is",
            "are",
            "am",
            "will",
            "would",
            "tell",
            "show",
            "explain",
        )
        if lowered.startswith(follow_up_starters):
            return True

        acknowledgement_followups = {
            "ok",
            "okay",
            "alr",
            "alright",
            "lol",
            "lmao",
            "real",
            "true",
            "same",
            "aw",
            "aww",
            "❤️",
            "🥹",
            "🤤",
        }
        if lowered in acknowledgement_followups:
            return True

        words = [part for part in re.split(r"\s+", content) if part]
        if len(words) >= 2:
            first = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", words[0]).casefold()
            if first and first not in follow_up_starters and first not in {"i", "im", "i'm", "you", "yo", "hey", "hi"}:
                return False

        return None

    async def _execute_tool_call(
        self,
        tool_call: dict[str, Any],
        tool_context: dict[str, Any],
    ) -> dict[str, Any]:
        function_payload = tool_call.get("function", {})
        function_name = function_payload.get("name", "")
        raw_arguments = function_payload.get("arguments", "{}")
        arguments = json.loads(raw_arguments)
        started_at = monotonic()
        LOGGER.info(
            "Tool call start name=%s arguments=%s",
            function_name,
            raw_arguments,
        )

        if function_name == "read_channel":
            result = await self._tool_read_channel(arguments, tool_context)
        elif function_name == "lookup_user":
            result = await self._tool_lookup_user(arguments, tool_context)
        elif function_name == "remember":
            result = await self._tool_remember(arguments, tool_context)
        elif function_name == "edit_memory":
            result = await self._tool_edit_memory(arguments, tool_context)
        elif function_name == "read_user_memory":
            result = await self._tool_read_user_memory(arguments, tool_context)
        elif function_name == "send_dm":
            result = await self._tool_send_dm(arguments, tool_context)
        elif function_name == "web_search":
            result = await self._tool_web_search(arguments, tool_context)
        elif function_name == "web_fetch":
            result = await self._tool_web_fetch(arguments, tool_context)
        elif function_name == "get_time":
            result = await self._tool_get_time(arguments, tool_context)
        else:
            raise RuntimeError(f"Unsupported tool call: {function_name}")

        LOGGER.info(
            "Tool call success name=%s duration=%.2fs ok=%s",
            function_name,
            monotonic() - started_at,
            result.get("ok"),
        )
        return result

    async def _tool_read_channel(
        self,
        arguments: dict[str, Any],
        tool_context: dict[str, Any],
    ) -> dict[str, Any]:
        message: discord.Message = tool_context["message"]
        limit = int(arguments.get("limit", self.app_config.runtime.discord.max_history_messages))
        limit = max(1, min(limit, 100))
        requested_channel_id = arguments.get("channel_id")
        target_channel = message.channel
        before_message_id = arguments.get("before_message_id", message.id)
        if requested_channel_id is not None:
            target_channel_id = int(requested_channel_id)
            target_channel = self._channel_from_message_context(message, target_channel_id)
            if target_channel is None:
                fetched_channel = await self.fetch_channel(target_channel_id)
                target_channel = fetched_channel
            before_message_id = arguments.get("before_message_id")
        LOGGER.info(
            "read_channel requester=%s target_channel_id=%s limit=%s before_message_id=%s",
            message.author.id,
            getattr(target_channel, "id", None),
            limit,
            before_message_id,
        )

        if not hasattr(target_channel, "history"):
            raise ValueError("read_channel target channel does not support message history.")

        before_message = discord.Object(id=int(before_message_id)) if before_message_id is not None else None
        history = []
        async for item in target_channel.history(limit=limit, before=before_message, oldest_first=False):
            history.append(
                {
                    "message_id": item.id,
                    "author_user_id": item.author.id,
                    "author_username": str(item.author),
                    "author_display_name": self._display_name_for_message_author(item),
                    "content": self._message_text_for_context(item),
                    "created_at": item.created_at.isoformat(),
                }
            )
            self._store_historical_dm_message(item)

        return {
            "ok": True,
            "channel_id": target_channel.id,
            "fetched_count": len(history),
            "messages": history,
        }

    async def _prefetch_channel_history_for_message(self, message: discord.Message) -> str:
        history_payload = await self._tool_read_channel({}, {"message": message})
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

    async def _tool_remember(
        self,
        arguments: dict[str, Any],
        tool_context: dict[str, Any],
    ) -> dict[str, Any]:
        message: discord.Message = tool_context["message"]
        self._require_owner(message, "remember")
        scope = str(arguments.get("scope", "")).strip().lower()
        if scope not in {"user", "server"}:
            raise ValueError("remember.scope must be 'user' or 'server'.")
        note = str(arguments.get("note", "")).strip()
        if not note:
            raise ValueError("remember.note is required.")

        if scope == "user":
            path = self.user_memory_store.append_for_user(
                user_id=message.author.id,
                note=note,
                author_display_name=self._display_name_for_message_author(message),
            )
            return {
                "ok": True,
                "scope": "user",
                "user_id": message.author.id,
                "path": str(path),
                "appended_note": note,
            }

        if message.guild is None:
            raise ValueError("remember(scope='server') requires a guild/server channel.")
        path = self.guild_memory_store.append_for_guild(
            guild_id=message.guild.id,
            guild_name=message.guild.name,
            note=note,
            owner_priority=True,
        )
        await self._notify_owner_of_guild_memory_update(
            guild_id=message.guild.id,
            guild_name=message.guild.name,
            summary=(
                "Owner-approved server memory note:\n"
                f"- channel_id: {message.channel.id}\n"
                f"- note: {note}"
            ),
        )
        return {
            "ok": True,
            "scope": "server",
            "guild_id": message.guild.id,
            "guild_name": message.guild.name,
            "path": str(path),
            "appended_note": note,
        }

    async def _tool_edit_memory(
        self,
        arguments: dict[str, Any],
        tool_context: dict[str, Any],
    ) -> dict[str, Any]:
        message: discord.Message = tool_context["message"]
        self._require_owner(message, "edit_memory")
        scope = str(arguments.get("scope", "")).strip().lower()
        if scope not in {"user", "server"}:
            raise ValueError("edit_memory.scope must be 'user' or 'server'.")
        find = str(arguments.get("find", "")).strip()
        replace_with = str(arguments.get("replace", "")).strip()
        if not find:
            raise ValueError("edit_memory.find is required.")

        if scope == "user":
            path, updated = self.user_memory_store.replace_for_user(
                user_id=message.author.id,
                old_text=find,
                new_text=replace_with,
            )
            return {
                "ok": updated,
                "scope": "user",
                "user_id": message.author.id,
                "path": str(path),
                "find": find,
                "replace": replace_with,
                "status": "updated" if updated else "not_found",
            }

        if message.guild is None:
            raise ValueError("edit_memory(scope='server') requires a guild/server channel.")
        path, updated = self.guild_memory_store.replace_for_guild(
            guild_id=message.guild.id,
            old_text=find,
            new_text=replace_with,
        )
        if updated:
            await self._notify_owner_of_guild_memory_update(
                guild_id=message.guild.id,
                guild_name=message.guild.name,
                summary=(
                    "Owner-approved server memory edit:\n"
                    f"- find: {find}\n"
                    f"- replace: {replace_with}"
                ),
            )
        return {
            "ok": updated,
            "scope": "server",
            "guild_id": message.guild.id,
            "guild_name": message.guild.name,
            "path": str(path),
            "find": find,
            "replace": replace_with,
            "status": "updated" if updated else "not_found",
        }

    async def _tool_read_user_memory(
        self,
        arguments: dict[str, Any],
        tool_context: dict[str, Any],
    ) -> dict[str, Any]:
        message: discord.Message = tool_context["message"]
        self._require_owner(message, "read_user_memory")

        requested_target_user_id = arguments.get("target_user_id")
        if requested_target_user_id is None:
            raise ValueError("read_user_memory.target_user_id is required.")
        target_user_id = self._resolve_target_user_id(message, requested_target_user_id)
        LOGGER.info("read_user_memory requester=%s target_user_id=%s", message.author.id, target_user_id)

        path = self.user_memory_store.path_for_user(target_user_id)
        content = self.user_memory_store.read_for_user(target_user_id)
        return {
            "ok": True,
            "target_user_id": target_user_id,
            "path": str(path),
            "exists": bool(content),
            "memory": content or "",
        }

    async def _tool_lookup_user(
        self,
        arguments: dict[str, Any],
        tool_context: dict[str, Any],
    ) -> dict[str, Any]:
        message: discord.Message = tool_context["message"]
        requested_target_user_id = arguments.get("user_id")
        target_user_id = (
            self._resolve_target_user_id(message, requested_target_user_id)
            if requested_target_user_id is not None
            else message.author.id
        )

        member = self._member_from_message_context(message, target_user_id)
        user = member or self._user_from_message_context(message, target_user_id)
        if user is None:
            user = await self.fetch_user(target_user_id)

        profile: Any | None = None
        profile_error: str | None = None
        try:
            if member is not None:
                profile = await member.profile(
                    with_mutual_guilds=True,
                    with_mutual_friends_count=False,
                    with_mutual_friends=False,
                )
            elif hasattr(user, "profile"):
                profile = await user.profile(
                    with_mutual_guilds=True,
                    with_mutual_friends_count=False,
                    with_mutual_friends=False,
                )
        except discord.HTTPException as exc:
            profile_error = f"{type(exc).__name__}: {exc}"

        effective_member = member if member is not None else (user if isinstance(user, discord.Member) else None)
        display_name = (
            self._display_name_for_message_author(message)
            if target_user_id == message.author.id
            else getattr(effective_member or user, "display_name", str(user))
        )
        bio = getattr(profile, "display_bio", None)
        if bio is None:
            bio = getattr(profile, "bio", None)
        if bio is None:
            bio = getattr(user, "bio", None)

        metadata = getattr(profile, "metadata", None)
        mutual_guilds = getattr(profile, "mutual_guilds", None)
        avatar = getattr(user, "display_avatar", None)
        banner = getattr(effective_member, "display_banner", None) or getattr(user, "banner", None)
        accent_color = None
        if metadata is not None and getattr(metadata, "accent_color", None) is not None:
            accent_color = str(metadata.accent_color)
        elif getattr(user, "accent_color", None) is not None:
            accent_color = str(user.accent_color)

        result: dict[str, Any] = {
            "ok": True,
            "target_user_id": user.id,
            "username": str(user),
            "name": getattr(user, "name", None),
            "display_name": display_name,
            "global_name": getattr(user, "global_name", None),
            "mention": getattr(user, "mention", f"<@{user.id}>"),
            "bot": getattr(user, "bot", False),
            "system": getattr(user, "system", False),
            "created_at": user.created_at.isoformat() if getattr(user, "created_at", None) else None,
            "avatar_url": str(avatar.url) if avatar is not None else None,
            "banner_url": str(banner.url) if banner is not None else None,
            "accent_color": accent_color,
            "bio": bio,
            "legacy_username": getattr(profile, "legacy_username", None),
            "profile_fetch_ok": profile_error is None,
            "profile_fetch_error": profile_error,
            "guild_member": {
                "in_current_guild": effective_member is not None,
                "nick": getattr(effective_member, "nick", None) if effective_member is not None else None,
                "joined_at": (
                    effective_member.joined_at.isoformat()
                    if effective_member is not None and getattr(effective_member, "joined_at", None)
                    else None
                ),
                "guild_bio": getattr(profile, "guild_bio", None),
            },
            "mutual_guilds": (
                [
                    {
                        "id": guild.id,
                        "name": getattr(getattr(guild, "guild", None), "name", None),
                        "nick": getattr(guild, "nick", None),
                    }
                    for guild in mutual_guilds[:10]
                ]
                if mutual_guilds
                else []
            ),
        }
        if metadata is not None:
            result["profile_metadata"] = {
                "pronouns": getattr(metadata, "pronouns", None),
                "banner_url": str(metadata.banner.url) if getattr(metadata, "banner", None) is not None else None,
            }
        return result

    async def _tool_send_dm(
        self,
        arguments: dict[str, Any],
        tool_context: dict[str, Any],
    ) -> dict[str, Any]:
        message: discord.Message = tool_context["message"]
        author_id = message.author.id
        owner_user_id = self.app_config.settings.owner_user_id
        is_owner = author_id == owner_user_id

        requested_target_user_id = arguments.get("target_user_id")
        if requested_target_user_id is None:
            raise ValueError("send_dm.target_user_id is required. Do not guess recipient.")
        target_user_id = self._resolve_target_user_id(message, requested_target_user_id)

        content = str(arguments.get("content", "")).strip()
        if not content:
            raise ValueError("send_dm.content is required.")

        if not is_owner and target_user_id != owner_user_id:
            raise ValueError("Only owner can DM arbitrary users. Non-owner may only target the owner.")
        if self.user is not None and target_user_id == self.user.id:
            raise ValueError("Do not DM the logged-in assistant account.")

        user = self.get_user(target_user_id)
        if user is None:
            user = await self.fetch_user(target_user_id)

        dm_channel = user.dm_channel
        if dm_channel is None:
            dm_channel = await user.create_dm()

        sent_message = await dm_channel.send(content)
        self._store_outgoing_dm_message(sent_message)
        return {
            "ok": True,
            "from_user_id": author_id,
            "target_user_id": target_user_id,
            "dm_channel_id": dm_channel.id,
            "message_id": sent_message.id,
            "sent_content": content,
        }

    async def _tool_web_search(
        self,
        arguments: dict[str, Any],
        tool_context: dict[str, Any],
    ) -> dict[str, Any]:
        cfg = self.app_config.runtime.web_tools
        if not cfg.enabled:
            return {
                "ok": False,
                "error_type": "disabled",
                "error_message": "web_search is disabled in config",
            }
        if self.http_session is None:
            return {
                "ok": False,
                "error_type": "no_http_session",
                "error_message": "HTTP session not ready",
            }
        api_key = self.app_config.settings.tavily_api_key
        if not api_key:
            return {
                "ok": False,
                "error_type": "no_api_key",
                "error_message": "TAVILY_API_KEY not configured",
            }
        query = str(arguments.get("query", "")).strip()
        if not query:
            return {
                "ok": False,
                "error_type": "bad_args",
                "error_message": "web_search.query is required",
            }
        payload = {
            "api_key": api_key,
            "query": query,
            "max_results": cfg.tavily_results_per_query,
            "search_depth": "basic",
        }
        timeout = aiohttp.ClientTimeout(total=15)
        try:
            async with self.http_session.post(
                "https://api.tavily.com/search",
                json=payload,
                timeout=timeout,
            ) as response:
                response.raise_for_status()
                data = await response.json()
        except Exception as exc:
            LOGGER.warning("web_search Tavily error query=%r error=%s", query, exc)
            return {
                "ok": False,
                "error_type": "tavily_error",
                "error_message": f"{type(exc).__name__}: {exc}",
            }
        results: list[dict[str, str]] = []
        seen: set[str] = set()
        for hit in data.get("results") or []:
            url = (hit.get("url") or "").strip()
            title = (hit.get("title") or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            results.append({"title": title, "url": url})
        return {"ok": True, "query": query, "results": results}

    async def _tool_web_fetch(
        self,
        arguments: dict[str, Any],
        tool_context: dict[str, Any],
    ) -> dict[str, Any]:
        cfg = self.app_config.runtime.web_tools
        if not cfg.enabled:
            return {
                "ok": False,
                "error_type": "disabled",
                "error_message": "web_fetch is disabled in config",
            }
        if self.http_session is None:
            return {
                "ok": False,
                "error_type": "no_http_session",
                "error_message": "HTTP session not ready",
            }
        url = str(arguments.get("url", "")).strip()
        prompt = str(arguments.get("prompt", "")).strip()
        if not url:
            return {
                "ok": False,
                "error_type": "bad_args",
                "error_message": "web_fetch.url is required",
            }
        if not prompt:
            return {
                "ok": False,
                "error_type": "bad_args",
                "error_message": "web_fetch.prompt is required",
            }
        if not url.lower().startswith(("http://", "https://")):
            return {
                "ok": False,
                "error_type": "bad_args",
                "error_message": "web_fetch.url must start with http:// or https://",
            }
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml",
        }
        timeout = aiohttp.ClientTimeout(total=cfg.fetch_timeout_seconds)
        try:
            async with self.http_session.get(url, timeout=timeout, headers=headers) as response:
                response.raise_for_status()
                html = await response.text(errors="ignore")
        except Exception as exc:
            LOGGER.warning("web_fetch fetch error url=%s error=%s", url, exc)
            return {
                "ok": False,
                "error_type": "fetch",
                "error_message": f"{type(exc).__name__}: {exc}",
            }
        text = await asyncio.to_thread(_strip_html_text, html, cfg.max_html_chars)
        if not text:
            return {
                "ok": False,
                "error_type": "empty_page",
                "error_message": "no extractable text on page",
            }
        user_prompt = (
            f"URL: {url}\n\nQuestion: {prompt}\n\nPage content:\n{text}"
        )
        try:
            response = await self.openrouter_chat(
                model=cfg.fetch_model,
                messages=[
                    {"role": "system", "content": cfg.fetch_system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                extra_payload={
                    "temperature": cfg.temperature,
                    "max_tokens": cfg.summary_max_tokens,
                    "service_tier": cfg.service_tier,
                    "reasoning": {"effort": "minimal", "exclude": True},
                },
            )
        except Exception as exc:
            LOGGER.warning("web_fetch summarize error url=%s error=%s", url, exc)
            return {
                "ok": False,
                "error_type": "summarize",
                "error_message": f"{type(exc).__name__}: {exc}",
            }
        choices = response.get("choices") or []
        answer = ""
        if choices:
            message = choices[0].get("message") or {}
            answer = (message.get("content") or "").strip()
            if not answer:
                answer = (message.get("reasoning") or "").strip()
        if not answer:
            finish_reason = choices[0].get("finish_reason") if choices else None
            return {
                "ok": False,
                "error_type": "empty_answer",
                "error_message": f"model returned no content (finish_reason={finish_reason})",
            }
        return {"ok": True, "url": url, "answer": answer}

    async def _tool_get_time(
        self,
        arguments: dict[str, Any],
        tool_context: dict[str, Any],
    ) -> dict[str, Any]:
        tz_name = str(arguments.get("timezone", "UTC")).strip() or "UTC"
        try:
            tz = ZoneInfo(tz_name)
        except ZoneInfoNotFoundError:
            return {
                "ok": False,
                "error_type": "bad_timezone",
                "error_message": (
                    f"Unknown timezone {tz_name!r}. Use an IANA name like "
                    "'America/New_York', 'Europe/London', or 'Asia/Tokyo'."
                ),
            }
        now = datetime.now(tz)
        return {
            "ok": True,
            "timezone": tz_name,
            "iso": now.isoformat(),
            "human": now.strftime("%A, %B %d, %Y at %I:%M %p"),
            "utc_offset": now.strftime("%z"),
            "tz_abbreviation": now.strftime("%Z"),
        }

    def _require_owner(self, message: discord.Message, tool_name: str) -> None:
        if message.author.id != self.app_config.settings.owner_user_id:
            raise ValueError(f"{tool_name} is restricted to the owner.")

    def _resolve_target_user_id(self, message: discord.Message, raw_value: Any) -> int:
        if isinstance(raw_value, int):
            return raw_value

        if isinstance(raw_value, str):
            value = raw_value.strip()
            if not value:
                raise ValueError("target_user_id cannot be empty.")
            if value.isdigit():
                return int(value)

            mention_match = re.fullmatch(r"<@!?(\d+)>", value)
            if mention_match:
                return int(mention_match.group(1))

            normalized = value.casefold()
            for user in getattr(message, "mentions", []):
                candidates = {
                    str(user).casefold(),
                    getattr(user, "name", str(user)).casefold(),
                    getattr(user, "display_name", str(user)).casefold(),
                    getattr(user, "global_name", "") .casefold(),
                    getattr(user, "mention", "").casefold(),
                }
                if normalized in candidates:
                    return int(user.id)

        raise ValueError(
            "Could not resolve target_user_id. Use a numeric user id or mention the user explicitly in the message."
        )

    def _user_memory_context_for_message(self, message: discord.Message) -> str:
        if not self.app_config.runtime.memory.enabled:
            return ""
        return self.user_memory_store.read_for_prompt(message.author.id)

    async def _notify_owner_of_guild_memory_update(
        self,
        *,
        guild_id: int,
        guild_name: str,
        summary: str,
    ) -> None:
        owner_user_id = self.app_config.settings.owner_user_id
        user = self.get_user(owner_user_id)
        if user is None:
            user = await self.fetch_user(owner_user_id)

        dm_channel = user.dm_channel
        if dm_channel is None:
            dm_channel = await user.create_dm()

        content = (
            f"Guild memory updated for {guild_name} ({guild_id}).\n"
            f"{summary.strip()}"
        ).strip()
        sent_message = await dm_channel.send(content, silent=True)
        self._store_outgoing_dm_message(sent_message)

    def _server_memory_context_for_message(self, message: discord.Message) -> str:
        if not self.app_config.runtime.memory.enabled:
            return ""
        if message.guild is None:
            return ""
        return self.guild_memory_store.read_for_prompt(message.guild.id)

    def _pending_burst_context(
        self,
        pending_messages: list[discord.Message] | None,
        *,
        latest_message_id: int,
    ) -> str:
        if not pending_messages or len(pending_messages) <= 1:
            return ""

        lines = [
            "Pre-reply message burst from same user in this channel. Treat these together as one request:"
        ]
        for item in pending_messages:
            marker = "latest" if item.id == latest_message_id else "earlier"
            content = self._message_text_for_context(item) or "(no text)"
            lines.append(f"- {marker}: {content}")
        lines.append("")
        return "\n".join(lines)

    def _store_incoming_dm_message(self, message: discord.Message) -> None:
        if not self._is_direct_message(message):
            return
        self.dm_history_store.append_message(
            user_id=message.author.id,
            role="user",
            content=self._message_text_for_context(message),
            discord_message_id=message.id,
            created_at=message.created_at.isoformat() if message.created_at else None,
        )

    def _store_historical_dm_message(self, message: discord.Message) -> None:
        if not self._is_direct_message(message):
            return
        if self.user is None:
            return

        recipient = getattr(message.channel, "recipient", None)
        if recipient is None:
            return

        if message.author.id == self.user.id:
            user_id = recipient.id
            role = "assistant"
        else:
            user_id = message.author.id
            role = "user"

        self.dm_history_store.append_message(
            user_id=user_id,
            role=role,
            content=self._message_text_for_context(message),
            discord_message_id=message.id,
            created_at=message.created_at.isoformat() if message.created_at else None,
        )

    def _store_outgoing_dm_message(self, message: discord.Message) -> None:
        if not self._is_direct_message(message):
            return
        recipient = getattr(message.channel, "recipient", None)
        if recipient is None:
            return
        self.dm_history_store.append_message(
            user_id=recipient.id,
            role="assistant",
            content=self._message_text_for_context(message),
            discord_message_id=message.id,
            created_at=message.created_at.isoformat() if message.created_at else None,
        )

    def _dm_conversation_context_for_message(self, message: discord.Message) -> list[dict[str, str]]:
        return self.dm_history_store.read_conversation(user_id=message.author.id)

    def _should_prefetch_channel_context(self, message: discord.Message) -> bool:
        if self._is_direct_message(message):
            return False

        content = message.content.lower()
        history_cues = (
            "what were we talking",
            "what were we discussing",
            "what did i ask",
            "answer the question",
            "that question",
            "earlier",
            "before",
            "pick up where",
            "continue where",
            "what was i saying",
            "what were we on about",
        )
        return any(cue in content for cue in history_cues)

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

        if "read_channel" in allowed:
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": "read_channel",
                        "description": (
                            "Read recent messages from the current Discord channel (or another channel by id). "
                            "Call when the user references prior context, mentions you with no other text, asks "
                            "'what were we talking about', or anything that needs scrollback. "
                            "Default limit is fine; paginate older with before_message_id only if needed."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "limit": {
                                    "type": "integer",
                                    "description": "How many messages to fetch. Default size is usually enough.",
                                    "minimum": 1,
                                    "maximum": 100,
                                },
                                "before_message_id": {
                                    "type": "integer",
                                    "description": "Fetch messages older than this id. Use returned older ids to paginate farther.",
                                },
                                "channel_id": {
                                    "type": "integer",
                                    "description": "Optional explicit Discord channel id. Use when the user names a specific #channel.",
                                },
                            },
                            "required": [],
                            "additionalProperties": False,
                        },
                    },
                }
            )

        if "lookup_user" in allowed:
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": "lookup_user",
                        "description": (
                            "Get profile details for a Discord user (defaults to the current chatter). "
                            "Use when the user asks who someone is, wants their id or bio, or you need to verify a target before send_dm."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "user_id": {
                                    "type": "integer",
                                    "description": "Optional Discord user id. Omit to look up the current message author.",
                                }
                            },
                            "required": [],
                            "additionalProperties": False,
                        },
                    },
                }
            )

        if "remember" in allowed:
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": "remember",
                        "description": (
                            "Owner-only. Save a durable note to memory. "
                            "scope='user' appends to the current user's memory file (identity, projects, long-term preferences, biographical facts). "
                            "scope='server' appends to the current guild's shared memory (server-wide rules, channel norms, cues that should affect replies for everyone in this guild). "
                            "Notes are written under the file's '## Notes' section. Save only things that should still matter next week."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "scope": {
                                    "type": "string",
                                    "enum": ["user", "server"],
                                    "description": "Which memory file to write to.",
                                },
                                "note": {
                                    "type": "string",
                                    "description": "Concise markdown-safe note. One fact per call.",
                                },
                            },
                            "required": ["scope", "note"],
                            "additionalProperties": False,
                        },
                    },
                }
            )

        if "edit_memory" in allowed:
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": "edit_memory",
                        "description": (
                            "Owner-only. Edit a memory file by substring replacement. "
                            "scope='user' edits the current user's memory; scope='server' edits the current guild's memory. "
                            "find must be an exact substring; replace overwrites it (use empty string to delete). "
                            "Use when an existing fact is wrong, outdated, or changed."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "scope": {
                                    "type": "string",
                                    "enum": ["user", "server"],
                                    "description": "Which memory file to edit.",
                                },
                                "find": {
                                    "type": "string",
                                    "description": "Exact substring already in the memory file.",
                                },
                                "replace": {
                                    "type": "string",
                                    "description": "Replacement text. Empty string deletes the matched substring.",
                                },
                            },
                            "required": ["scope", "find", "replace"],
                            "additionalProperties": False,
                        },
                    },
                }
            )

        if "read_user_memory" in allowed:
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": "read_user_memory",
                        "description": "Owner-only. Read the memory file for a specific Discord user id.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "target_user_id": {
                                    "type": "integer",
                                    "description": "Discord user id whose memory file to read.",
                                }
                            },
                            "required": ["target_user_id"],
                            "additionalProperties": False,
                        },
                    },
                }
            )

        if "send_dm" in allowed:
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": "send_dm",
                        "description": (
                            "Send a direct message to a specific Discord user. "
                            "target_user_id MUST come from a mention, lookup_user, or prior context — never guess. "
                            "Non-owner can only target the owner; owner can target anyone. "
                            "Never claim the message was sent unless this tool returned ok=true."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "target_user_id": {
                                    "type": "integer",
                                    "description": "Discord user id to DM. Required.",
                                },
                                "content": {
                                    "type": "string",
                                    "description": "Message text to send.",
                                },
                            },
                            "required": ["target_user_id", "content"],
                            "additionalProperties": False,
                        },
                    },
                }
            )

        if "web_search" in allowed:
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "description": (
                            "Search the live web via Tavily. Returns a list of {title, url} results — no snippets, no page content. "
                            "Use for current events, recent facts, or anything that needs up-to-date sources. "
                            "After searching, pick the most promising URL(s) and call web_fetch on them to actually read the content."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The search query to send to Tavily.",
                                },
                            },
                            "required": ["query"],
                            "additionalProperties": False,
                        },
                    },
                }
            )

        if "web_fetch" in allowed:
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": "web_fetch",
                        "description": (
                            "Fetch one URL and answer a focused question about its content. "
                            "Only pass URLs that came from a prior web_search result or from the user's own message — do not invent URLs. "
                            "Returns a short grounded answer string. Typically used after web_search to actually read a page."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "url": {
                                    "type": "string",
                                    "description": "The http(s) URL to fetch.",
                                },
                                "prompt": {
                                    "type": "string",
                                    "description": "The specific question to answer using the page's content.",
                                },
                            },
                            "required": ["url", "prompt"],
                            "additionalProperties": False,
                        },
                    },
                }
            )

        if "get_time" in allowed:
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": "get_time",
                        "description": (
                            "Get the exact current time in a specific timezone. "
                            "Always call this when the user asks for the time anywhere — never guess. "
                            "Use IANA timezone names like 'America/New_York' (US East), "
                            "'America/Los_Angeles' (US West), 'Europe/London', 'Asia/Tokyo', 'UTC'."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "timezone": {
                                    "type": "string",
                                    "description": "IANA timezone name. Defaults to UTC if omitted.",
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
                    "name": "read_channel",
                },
            }
        return "auto"

    def _has_active_conversation(self, message: discord.Message) -> bool:
        if self._is_direct_message(message):
            return False
        return self._get_active_conversation(message) is not None

    def _has_pending_reply(self, message: discord.Message) -> bool:
        key = self._conversation_key(message)
        task = self._reply_tasks.get(key)
        return key in self._pending_replies or (task is not None and not task.done())

    def _get_active_conversation(self, message: discord.Message) -> ActiveConversation | None:
        if self._is_direct_message(message):
            return None
        key = self._conversation_key(message)
        conversation = self._active_conversations.get(key)
        if conversation is None:
            return None

        if conversation.expires_at <= monotonic():
            del self._active_conversations[key]
            return None

        return conversation

    def _active_conversation_keys_for_channel(self, channel_id: int) -> list[tuple[int, int]]:
        keys: list[tuple[int, int]] = []
        now = monotonic()
        for key, conversation in list(self._active_conversations.items()):
            if key[0] != channel_id:
                continue
            if conversation.expires_at <= now:
                del self._active_conversations[key]
                continue
            keys.append(key)
        return keys

    def _append_sibling_channel_context(self, message: discord.Message) -> None:
        if self.user is None:
            return

        if message.author.id == self.user.id:
            return

        content = message.content.strip()
        if not content:
            return

        for key in self._active_conversation_keys_for_channel(message.channel.id):
            if key[1] == message.author.id:
                continue

            conversation = self._active_conversations.get(key)
            if conversation is None:
                continue

            conversation.messages.append(
                {
                    "role": "user",
                    "content": (
                        "Channel context update from another user. This message was not directed at you, "
                        "but happened in same channel during active conversation.\n"
                        f"Author username: {message.author}\n"
                        f"Author display name: {self._display_name_for_message_author(message)}\n"
                        f"Channel message: {content}"
                    ),
                }
            )
            conversation.messages = conversation.messages[-12:]
            conversation.expires_at = monotonic() + self.app_config.runtime.discord.conversation_window_seconds
            conversation.interrupted_by_other_user = True

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
            interrupted_by_other_user=False,
        )

    @staticmethod
    def _conversation_key(message: discord.Message) -> tuple[int, int]:
        return (message.channel.id, message.author.id)

    @staticmethod
    def _is_direct_message(message: discord.Message) -> bool:
        return message.guild is None

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
    def extract_response_text(
        response: dict[str, Any],
        *,
        messages: list[dict[str, Any]] | None = None,
    ) -> str:
        choices = response.get("choices", [])
        if not choices:
            LOGGER.warning("OpenRouter response had no choices. Falling back to default reply.")
            return DiscoAssistant._fallback_response_text(
                messages,
                default="I couldn't produce a reply just now.",
            )

        choice = choices[0]
        finish_reason = choice.get("finish_reason")
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

        LOGGER.warning(
            "OpenRouter response had no usable text content. finish_reason=%r content_type=%s tool_calls=%s message_keys=%s",
            finish_reason,
            type(content).__name__,
            bool(message.get("tool_calls")),
            sorted(message.keys()),
        )
        return DiscoAssistant._fallback_response_text(messages)

    @staticmethod
    def _display_name_for_message_author(message: discord.Message) -> str:
        author = message.author
        return getattr(author, "display_name", str(author))

    def _mentioned_users_context(self, message: discord.Message) -> str:
        if not message.mentions:
            return "Mentioned users: none\n"

        lines = ["Mentioned users:"]
        for user in message.mentions:
            lines.append(
                f"- username: {user} | display_name: {getattr(user, 'display_name', str(user))} | user_id: {user.id}"
            )
        return "\n".join(lines) + "\n"

    def _mentioned_channels_context(self, message: discord.Message) -> str:
        channel_mentions = getattr(message, "channel_mentions", [])
        if not channel_mentions:
            return "Mentioned channels: none\n"

        lines = ["Mentioned channels:"]
        for channel in channel_mentions:
            lines.append(
                f"- channel_name: #{getattr(channel, 'name', 'unknown')} | channel_id: {channel.id}"
            )
        return "\n".join(lines) + "\n"

    @staticmethod
    def _channel_from_message_context(
        message: discord.Message,
        channel_id: int,
    ) -> discord.abc.MessageableChannel | None:
        if getattr(message.channel, "id", None) == channel_id:
            return message.channel
        for channel in getattr(message, "channel_mentions", []):
            if channel.id == channel_id:
                return channel
        guild = getattr(message, "guild", None)
        if guild is not None:
            return guild.get_channel(channel_id)
        return None

    async def _reply_reference_context(self, message: discord.Message) -> str:
        reference = getattr(message, "reference", None)
        if reference is None:
            return "Reply target: none\n"

        referenced_message = await self._resolve_referenced_message(message)
        if referenced_message is None:
            message_id = getattr(reference, "message_id", None)
            return f"Reply target: unresolved message_id={message_id}\n"

        author_name = self._display_name_for_message_author(referenced_message)
        content = self._message_text_for_context(referenced_message) or "(no text)"
        return (
            "Reply target:\n"
            f"- author: {author_name}\n"
            f"- message_id: {referenced_message.id}\n"
            f"- content: {content}\n"
        )

    async def _resolve_referenced_message(self, message: discord.Message) -> discord.Message | None:
        reference = getattr(message, "reference", None)
        if reference is None:
            return None

        resolved = getattr(reference, "resolved", None)
        if isinstance(resolved, discord.Message):
            return resolved

        message_id = getattr(reference, "message_id", None)
        if message_id is None:
            return None

        channel = getattr(message, "channel", None)
        if channel is None or not hasattr(channel, "fetch_message"):
            return None

        try:
            return await channel.fetch_message(message_id)
        except discord.HTTPException:
            return None

    async def _owner_context_prompt(self, message: discord.Message) -> str:
        if self._owner_context_prompt_cache is not None:
            return self._owner_context_prompt_cache

        owner_user_id = self.app_config.settings.owner_user_id
        owner_user = self._user_from_message_context(message, owner_user_id)
        if owner_user is None:
            owner_user = self.get_user(owner_user_id)
        if owner_user is None:
            try:
                owner_user = await self.fetch_user(owner_user_id)
            except discord.HTTPException as exc:
                LOGGER.warning("Failed to fetch owner user details for prompt: %s", exc)
                prompt = f"Owner profile:\n- owner_user_id: {owner_user_id}\n"
                self._owner_context_prompt_cache = prompt
                return prompt

        owner_bio = getattr(owner_user, "bio", None)
        if owner_bio is None and hasattr(owner_user, "profile"):
            try:
                owner_profile = await owner_user.profile(
                    with_mutual_guilds=False,
                    with_mutual_friends_count=False,
                    with_mutual_friends=False,
                )
                owner_bio = getattr(owner_profile, "display_bio", None) or getattr(owner_profile, "bio", None)
            except discord.HTTPException as exc:
                LOGGER.info("Owner profile fetch unavailable for prompt: %s", exc)

        prompt_lines = [
            "Owner profile:",
            f"- owner_user_id: {owner_user_id}",
            f"- owner_username: {owner_user}",
            f"- owner_name: {getattr(owner_user, 'name', str(owner_user))}",
            f"- owner_display_name: {getattr(owner_user, 'display_name', str(owner_user))}",
            f"- owner_global_name: {getattr(owner_user, 'global_name', None) or '(none)'}",
            f"- owner_mention: {getattr(owner_user, 'mention', f'<@{owner_user_id}>')}",
            f"- owner_bio: {owner_bio or '(none)'}",
            (
                f"- owner_created_at: {owner_user.created_at.isoformat()}"
                if getattr(owner_user, "created_at", None)
                else "- owner_created_at: (unknown)"
            ),
        ]
        self._owner_context_prompt_cache = "\n".join(prompt_lines) + "\n"
        return self._owner_context_prompt_cache

    def _owner_only_tools_prompt(self, active_agent_name: str) -> str:
        owner_agent = self.app_config.runtime.agents.get("owner")
        public_agent = self.app_config.runtime.agents.get("public")
        if owner_agent is None or public_agent is None:
            return ""

        owner_only_tools = [tool for tool in owner_agent.tools if tool not in public_agent.tools]
        if not owner_only_tools:
            return ""

        lines = [
            "Owner-only capabilities:",
            f"- real owner agent name: {owner_agent.name}",
            f"- current active agent name: {active_agent_name}",
            f"- owner-only tool calls: {', '.join(owner_only_tools)}",
            "- If current user is not owner, these tool calls are not available to you.",
            "- If a non-owner asks for an owner-only action, explain limitation briefly.",
            "- Only if escalation is genuinely needed, ping real owner in channel using owner_mention.",
            "- Never claim you used an owner-only tool unless tool result explicitly proves it.",
        ]
        return "\n".join(lines) + "\n"

    def _assistant_identity_prompt(self) -> str:
        owner_user_id = self.app_config.settings.owner_user_id
        if self.user is None:
            return (
                "Assistant identity:\n"
                "- platform: Discord\n"
                f"- owner_user_id: {owner_user_id}\n"
                "- You are the logged-in user account for this assistant.\n"
            )

        return (
            "Assistant identity:\n"
            f"- account_username: {self.user}\n"
            f"- account_user_id: {self.user.id}\n"
            f"- account_mention: {self.user.mention}\n"
            f"- owner_user_id: {owner_user_id}\n"
            "- You are this Discord account. Do not confuse yourself with message author or owner.\n"
            "- If user asks who you are, use this identity.\n"
        )

    def _current_time_prompt(self) -> str:
        now_utc = datetime.now(timezone.utc)
        return (
            "Current time (real clock — do not guess or compute from memory):\n"
            f"- UTC now: {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
            f"- Weekday: {now_utc.strftime('%A')}\n"
            "If asked for time in a specific timezone, call get_time with an IANA "
            "name (e.g. 'America/New_York' for US East, 'America/Los_Angeles' for "
            "US West, 'Europe/London', 'Asia/Tokyo'). Never invent a time."
        )

    @staticmethod
    def _fallback_response_text(
        messages: list[dict[str, Any]] | None,
        *,
        default: str = "I couldn't produce a normal reply, but I can try again.",
    ) -> str:
        if messages:
            for item in reversed(messages):
                if item.get("role") != "tool":
                    continue
                try:
                    payload = json.loads(item.get("content", "{}"))
                except json.JSONDecodeError:
                    continue

                if payload.get("ok") is False:
                    error_message = str(payload.get("error_message", "")).strip()
                    if error_message:
                        return f"Couldn't do that: {error_message}"
                    return "Couldn't do that."

                if item.get("name") == "send_dm" and payload.get("ok") is True:
                    target_user_id = payload.get("target_user_id")
                    return f"Message sent to user `{target_user_id}`."

        return default

    @staticmethod
    def _safe_runtime_error_reply(exc: Exception) -> str:
        if isinstance(exc, aiohttp.ClientResponseError) and exc.status == 429:
            if "openrouter.ai" in str(exc.request_info.real_url):
                return "Model rate-limited me. Try again in a few seconds."
            return "Rate-limited by upstream service. Try again in a few seconds."
        message = str(exc).strip()
        if message:
            return f"I hit an internal error: {type(exc).__name__}: {message}"
        return f"I hit an internal error: {type(exc).__name__}."

    def _user_from_message_context(
        self,
        message: discord.Message,
        user_id: int,
    ) -> discord.abc.User | None:
        if message.author.id == user_id:
            return message.author

        for user in self._iter_mentions(message):
            if user.id == user_id:
                return user

        member = self._member_from_message_context(message, user_id)
        if member is not None:
            return member

        return self.get_user(user_id)

    def _member_from_message_context(
        self,
        message: discord.Message,
        user_id: int,
    ) -> discord.Member | None:
        if isinstance(message.author, discord.Member) and message.author.id == user_id:
            return message.author

        guild = getattr(message, "guild", None)
        if guild is None:
            return None
        return guild.get_member(user_id)

    @staticmethod
    def _retry_after_seconds(headers: Any) -> float | None:
        if not headers:
            return None

        value = headers.get("Retry-After")
        if value is None:
            return None

        try:
            seconds = float(value)
        except (TypeError, ValueError):
            return None
        return max(0.0, seconds)


bot: DiscoAssistant | None = None


def main() -> None:
    global bot

    log_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    formatter = logging.Formatter(log_format)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        BASE_DIR / "discoassistant.runtime.log",
        maxBytes=2_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)

    app_config = load_app_config()
    bot = DiscoAssistant(app_config)
    bot.run(app_config.settings.discord_token)
