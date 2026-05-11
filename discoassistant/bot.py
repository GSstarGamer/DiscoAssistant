from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from dataclasses import dataclass, field
from time import monotonic
from typing import Any
from collections.abc import Iterable

import aiohttp
import discord

from discoassistant.config import BASE_DIR, AppConfig, load_app_config
from discoassistant.dm_history import DmHistoryStore
from discoassistant.memory import GuildMemoryStore, UserMemoryStore


LOGGER = logging.getLogger("discoassistant")


@dataclass(slots=True)
class ActiveConversation:
    messages: list[dict[str, str]] = field(default_factory=list)
    expires_at: float = 0.0


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

    async def on_message(self, message: discord.Message) -> None:
        self._append_passive_channel_context(message)

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
    ) -> dict[str, Any]:
        if self.http_session is None:
            raise RuntimeError("HTTP session has not been created yet.")

        payload: dict[str, Any] = {
            "model": model or self.app_config.runtime.openrouter.default_model,
            "messages": messages,
        }
        if extra_payload:
            payload.update(extra_payload)

        max_attempts = 4
        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                async with self.http_session.post(
                    f"{self.app_config.runtime.openrouter.base_url}/chat/completions",
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    response_payload = await response.json()
                    self._record_token_usage(response_payload)
                    return response_payload
            except aiohttp.ClientResponseError as exc:
                last_error = exc
                should_retry = exc.status in {429, 500, 502, 503, 504} and attempt < max_attempts
                if not should_retry:
                    raise

                retry_after = self._retry_after_seconds(exc.headers)
                delay = retry_after if retry_after is not None else min(8.0, 1.5 * (2 ** (attempt - 1)))
                LOGGER.warning(
                    "OpenRouter request failed status=%s attempt=%s/%s retry_in=%.2fs",
                    exc.status,
                    attempt,
                    max_attempts,
                    delay,
                )
                await asyncio.sleep(delay)
            except aiohttp.ClientError as exc:
                last_error = exc
                if attempt >= max_attempts:
                    raise
                delay = min(8.0, 1.0 * (2 ** (attempt - 1)))
                LOGGER.warning(
                    "OpenRouter network error attempt=%s/%s retry_in=%.2fs error=%s",
                    attempt,
                    max_attempts,
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
                        "When using send_message, never guess recipient. "
                        "Never say a message was sent unless tool result explicitly says ok true. "
                        "If a tool call fails, keep working and retry when possible. "
                        "Keep reply short."
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
                    f"Channel message: {message.content.strip()}\n"
                    "If this is a mention-only message, do not send a generic greeting. Use recent channel history to infer what the user is asking and answer that directly.\n"
                    "When using send_message, never guess recipient. Use an explicit mentioned user id or the owner user id. If target is unclear, say so instead of sending.\n"
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
            if not should_respond and pending is not None and pending.started_from_mention:
                LOGGER.info("reply task treating key=%s message_id=%s as pending mention follow-up", key, message.id)
                should_respond = True
            if not should_respond and pending is not None and pending.started_from_active_conversation:
                LOGGER.info("reply task treating key=%s message_id=%s as active-conversation burst", key, message.id)
                should_respond = True
            if not should_respond and self._has_active_conversation(message):
                LOGGER.info("checking follow-up classifier key=%s message_id=%s", key, message.id)
                should_respond = await self._should_treat_active_conversation_message_as_follow_up(message)

            if not should_respond:
                LOGGER.info("reply task exit key=%s message_id=%s reason=should_not_respond", key, message.id)
                return

            self._reply_generation_in_progress.add(key)
            self._recent_response_ids.append(message.id)
            LOGGER.info("start generation key=%s message_id=%s", key, message.id)
            async with message.channel.typing():
                reply_text = await self.generate_reply_for_message(message, pending_messages)
            LOGGER.info("generation done key=%s message_id=%s reply=%r", key, message.id, reply_text)
            sent_message = await message.channel.send(reply_text, reference=message)
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
                await message.channel.send(
                    self._safe_runtime_error_reply(exc)
                )
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
        if function_name == "append_user_memory":
            return await self._tool_append_user_memory(arguments, tool_context)
        if function_name == "edit_user_memory":
            return await self._tool_edit_user_memory(arguments, tool_context)
        if function_name == "append_server_memory":
            return await self._tool_append_server_memory(arguments, tool_context)
        if function_name == "edit_server_memory":
            return await self._tool_edit_server_memory(arguments, tool_context)
        if function_name == "get_user_details":
            return await self._tool_get_user_details(arguments, tool_context)
        if function_name == "send_message":
            return await self._tool_send_message(arguments, tool_context)

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
            self._store_historical_dm_message(item)

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

    async def _tool_append_user_memory(
        self,
        arguments: dict[str, Any],
        tool_context: dict[str, Any],
    ) -> dict[str, Any]:
        message: discord.Message = tool_context["message"]
        note = str(arguments.get("note", "")).strip()
        if not note:
            raise ValueError("append_user_memory.note is required.")

        path = self.user_memory_store.append_for_user(
            user_id=message.author.id,
            note=note,
            author_display_name=self._display_name_for_message_author(message),
            source_channel_id=message.channel.id,
        )
        return {
            "ok": True,
            "user_id": message.author.id,
            "path": str(path),
            "appended_note": note,
        }

    async def _tool_edit_user_memory(
        self,
        arguments: dict[str, Any],
        tool_context: dict[str, Any],
    ) -> dict[str, Any]:
        message: discord.Message = tool_context["message"]
        old_text = str(arguments.get("old_text", "")).strip()
        new_text = str(arguments.get("new_text", "")).strip()
        if not old_text or not new_text:
            raise ValueError("edit_user_memory.old_text and new_text are required.")

        path, updated = self.user_memory_store.replace_for_user(
            user_id=message.author.id,
            old_text=old_text,
            new_text=new_text,
        )
        return {
            "ok": updated,
            "user_id": message.author.id,
            "path": str(path),
            "old_text": old_text,
            "new_text": new_text,
            "status": "updated" if updated else "not_found",
        }

    async def _tool_get_user_details(
        self,
        arguments: dict[str, Any],
        tool_context: dict[str, Any],
    ) -> dict[str, Any]:
        message: discord.Message = tool_context["message"]
        requested_target_user_id = arguments.get("target_user_id")
        target_user_id = int(requested_target_user_id) if requested_target_user_id is not None else message.author.id

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

    async def _tool_append_server_memory(
        self,
        arguments: dict[str, Any],
        tool_context: dict[str, Any],
    ) -> dict[str, Any]:
        message: discord.Message = tool_context["message"]
        if message.guild is None:
            raise ValueError("append_server_memory requires a guild/server channel.")

        note = str(arguments.get("note", "")).strip()
        if not note:
            raise ValueError("append_server_memory.note is required.")

        path = self.guild_memory_store.append_for_guild(
            guild_id=message.guild.id,
            guild_name=message.guild.name,
            note=note,
            author_display_name=self._display_name_for_message_author(message),
            source_channel_id=message.channel.id,
        )
        return {
            "ok": True,
            "guild_id": message.guild.id,
            "guild_name": message.guild.name,
            "path": str(path),
            "appended_note": note,
        }

    async def _tool_edit_server_memory(
        self,
        arguments: dict[str, Any],
        tool_context: dict[str, Any],
    ) -> dict[str, Any]:
        message: discord.Message = tool_context["message"]
        if message.guild is None:
            raise ValueError("edit_server_memory requires a guild/server channel.")

        old_text = str(arguments.get("old_text", "")).strip()
        new_text = str(arguments.get("new_text", "")).strip()
        if not old_text or not new_text:
            raise ValueError("edit_server_memory.old_text and new_text are required.")

        path, updated = self.guild_memory_store.replace_for_guild(
            guild_id=message.guild.id,
            old_text=old_text,
            new_text=new_text,
        )
        return {
            "ok": updated,
            "guild_id": message.guild.id,
            "guild_name": message.guild.name,
            "path": str(path),
            "old_text": old_text,
            "new_text": new_text,
            "status": "updated" if updated else "not_found",
        }

    async def _tool_send_message(
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
            raise ValueError("send_message.target_user_id is required. Do not guess recipient.")
        target_user_id = int(requested_target_user_id)

        content = str(arguments.get("message", "")).strip()
        if not content:
            raise ValueError("send_message.message is required.")

        if not is_owner and target_user_id != owner_user_id:
            raise ValueError("Only owner can send DMs to arbitrary users.")
        if self.user is not None and target_user_id == self.user.id:
            raise ValueError("Do not DM logged-in assistant account.")

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
            "sent_message": content,
        }

    def _user_memory_context_for_message(self, message: discord.Message) -> str:
        if not self.app_config.runtime.memory.enabled:
            return ""
        return self.user_memory_store.read_for_prompt(message.author.id)

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
            content = item.content.strip() or "(no text)"
            lines.append(f"- {marker}: {content}")
        lines.append("")
        return "\n".join(lines)

    def _store_incoming_dm_message(self, message: discord.Message) -> None:
        if not self._is_direct_message(message):
            return
        self.dm_history_store.append_message(
            user_id=message.author.id,
            role="user",
            content=message.content,
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
            content=message.content,
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
            content=message.content,
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

        if "append_user_memory" in allowed:
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": "append_user_memory",
                        "description": (
                            "Append durable notes about current Discord user to that user's markdown memory file. "
                            "Use for useful long-term facts, preferences, plans, personal context, repeated habits, "
                            "or even small details likely to help in future conversations."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "note": {
                                    "type": "string",
                                    "description": (
                                        "Short markdown-safe note to append to this user's memory file. "
                                        "Write concrete facts, preferences, or context worth remembering."
                                    ),
                                }
                            },
                            "required": ["note"],
                            "additionalProperties": False,
                        },
                    },
                }
            )

        if "edit_user_memory" in allowed:
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": "edit_user_memory",
                        "description": (
                            "Update stale or changed facts inside current user's markdown memory file by replacing "
                            "specific old text with new text. Use when existing memory is wrong, outdated, or changed."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "old_text": {
                                    "type": "string",
                                    "description": "Exact existing text snippet from user memory that should be replaced.",
                                },
                                "new_text": {
                                    "type": "string",
                                    "description": "New replacement text that should overwrite old_text.",
                                },
                            },
                            "required": ["old_text", "new_text"],
                            "additionalProperties": False,
                        },
                    },
                }
            )

        if "append_server_memory" in allowed:
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": "append_server_memory",
                        "description": (
                            "Append shared durable notes about current Discord server to that server's markdown memory file. "
                            "Use for small preferences, running jokes, local conventions, recurring relationships, "
                            "or any server-level detail likely to help later, even if nobody explicitly asked you to remember it."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "note": {
                                    "type": "string",
                                    "description": (
                                        "Short markdown-safe note to append to this server's memory file."
                                    ),
                                }
                            },
                            "required": ["note"],
                            "additionalProperties": False,
                        },
                    },
                }
            )

        if "edit_server_memory" in allowed:
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": "edit_server_memory",
                        "description": (
                            "Update stale or changed facts inside current server's markdown memory file by replacing "
                            "specific old text with new text."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "old_text": {
                                    "type": "string",
                                    "description": "Exact existing text snippet from server memory that should be replaced.",
                                },
                                "new_text": {
                                    "type": "string",
                                    "description": "New replacement text that should overwrite old_text.",
                                },
                            },
                            "required": ["old_text", "new_text"],
                            "additionalProperties": False,
                        },
                    },
                }
            )

        if "get_user_details" in allowed:
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": "get_user_details",
                        "description": (
                            "Fetch profile details for current chatter by default, or for an explicit Discord user id. "
                            "Use this when user asks who someone is, wants their user id, bio, profile details, "
                            "or more context about the current chatter."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "target_user_id": {
                                    "type": "integer",
                                    "description": (
                                        "Optional Discord user id to inspect. Omit to inspect current message author."
                                    ),
                                }
                            },
                            "required": [],
                            "additionalProperties": False,
                        },
                    },
                }
            )

        if "send_message" in allowed:
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": "send_message",
                        "description": (
                            "Send direct message to Discord user. Non-owner users may only message owner. "
                            "Owner may message any user. Never use this unless target user id is explicit."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "target_user_id": {
                                    "type": "integer",
                                    "description": (
                                        "Discord user id to DM. Required. Use owner user id when user wants to message owner."
                                    ),
                                },
                                "message": {
                                    "type": "string",
                                    "description": "Direct message content to send.",
                                },
                            },
                            "required": ["target_user_id", "message"],
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

    def _append_passive_channel_context(self, message: discord.Message) -> None:
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

                if item.get("name") == "send_message" and payload.get("ok") is True:
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

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    app_config = load_app_config()
    bot = DiscoAssistant(app_config)
    bot.run(app_config.settings.discord_token)
