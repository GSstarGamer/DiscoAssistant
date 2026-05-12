from __future__ import annotations

import asyncio
import logging
import re
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from time import monotonic
from typing import Any
from collections.abc import Iterable

import discord
from pathlib import Path

from discoassistant.config import BASE_DIR, AppConfig, load_app_config
from discoassistant.dm_history import DmHistoryStore
from discoassistant.errors.formatter import (
    fallback_response_text,
    safe_runtime_error_reply,
    synthesize_deterministic_reply,
)
from discoassistant.llm.memory_capture import OwnerMemoryCapturer
from discoassistant.llm.openrouter import OpenRouterClient
from discoassistant.llm.prompt_builder import ReplyPromptBuilder
from discoassistant.llm.response import extract_response_text
from discoassistant.llm.tool_loop import ToolLoopRunner
from discoassistant.llm.tool_registry import ToolContext
from discoassistant.llm.tools import build_default_registry
from discoassistant.memory import GuildMemoryStore, UserMemoryStore
from discoassistant.passive_guild import OwnerNotifier, PassiveGuildPoller, PassiveGuildSummarizer
from discoassistant.passive_guild_history import PassiveGuildHistoryStore
from discoassistant.runtime import (
    ActiveConversation,
    ConversationKey,
    ConversationStore,
    InFlightTaskRegistry,
    MessageDebouncer,
    PassiveFlushConfirmationStore,
    PendingReplyManager,
    RecentResponseIds,
    TokenMeter,
    TypingHeartbeatRegistry,
    conversation_key,
)


LOGGER = logging.getLogger("discoassistant")

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


class DiscoAssistant(discord.Client):
    def __init__(self, app_config: AppConfig) -> None:
        super().__init__()
        self.app_config = app_config
        self._startup_announced = False
        self.token_meter = TokenMeter(on_change=self._schedule_presence_refresh)
        self.openrouter = OpenRouterClient(
            api_key=app_config.settings.openrouter_api_key,
            base_url=app_config.runtime.openrouter.base_url,
            app_name=app_config.runtime.openrouter.app_name,
            site_url=app_config.runtime.openrouter.site_url,
            default_model=app_config.runtime.openrouter.default_model,
            on_token_usage=self.token_meter.record,
        )
        self._owner_context_prompt_cache: str | None = None
        self._assistant_identity_cache: str | None = None
        self._static_system_prompt: str = self._load_static_system_prompt()
        self.response_dedup = RecentResponseIds(maxlen=200)
        self.conversation_store = ConversationStore(
            window_seconds=self.app_config.runtime.discord.conversation_window_seconds,
            messages_cap=12,
        )
        self.pending = PendingReplyManager(
            signal_callback=lambda k: self.debouncer.signal(k),
        )
        self.debouncer = MessageDebouncer(
            window_seconds=self.app_config.runtime.discord.reply_debounce_seconds,
            pending_lookup=self.pending.get,
        )
        self.in_flight = InFlightTaskRegistry()
        self.typing_registry = TypingHeartbeatRegistry()
        self._dm_summary_in_flight: set[int] = set()
        self._presence_update_task: asyncio.Task[None] | None = None
        self._presence_refresh_requested = False
        self.flush_confirmations = PassiveFlushConfirmationStore()
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
        self.passive_guild_history_store = PassiveGuildHistoryStore(
            db_path=BASE_DIR / self.app_config.runtime.passive_guild_memory.db_path,
        )
        self.owner_memory_capturer = OwnerMemoryCapturer(
            openrouter_client=self.openrouter,
            user_memory_store=self.user_memory_store,
            guild_memory_store=self.guild_memory_store,
            owner_user_id=self.app_config.settings.owner_user_id,
            owner_agent=self.app_config.runtime.agents["owner"],
            memory_enabled=self.app_config.runtime.memory.enabled,
            message_text_for_context=self._message_text_for_context,
            is_direct_message=self._is_direct_message,
            is_mention_without_other_text=self._is_mention_without_other_text,
            conversation_key=self._conversation_key,
            server_memory_context_for_message=self._server_memory_context_for_message,
            display_name_for_message_author=self._display_name_for_message_author,
        )
        self.owner_notifier = OwnerNotifier(
            send_dm_to_owner=self._send_dm_to_owner,
        )
        self.passive_summarizer = PassiveGuildSummarizer(
            openrouter_client=self.openrouter,
            passive_guild_history_store=self.passive_guild_history_store,
            guild_memory_store=self.guild_memory_store,
            passive_config=self.app_config.runtime.passive_guild_memory,
            owner_notifier=self.owner_notifier,
            get_guild_name=self._guild_name_for_id,
        )
        self.passive_poller = PassiveGuildPoller(
            passive_guild_history_store=self.passive_guild_history_store,
            passive_config=self.app_config.runtime.passive_guild_memory,
            summarizer=self.passive_summarizer,
            is_closed=self.is_closed,
        )
        self.tool_registry = build_default_registry(self)
        self.tool_loop = ToolLoopRunner(
            openrouter_client=self.openrouter,
            registry=self.tool_registry,
            max_calls_per_turn=self.app_config.runtime.tool_policy.max_calls_per_turn,
        )
        self.prompt_builder = ReplyPromptBuilder(
            static_system_prompt=self._static_system_prompt,
            memory_enabled=self.app_config.runtime.memory.enabled,
            owner_user_id=self.app_config.settings.owner_user_id,
            agent_for_message=self._agent_for_message,
            owner_context_prompt=self._owner_context_prompt,
            assistant_identity_prompt=self._assistant_identity_prompt,
            is_direct_message=self._is_direct_message,
            is_mention_without_other_text=self._is_mention_without_other_text,
            get_active_conversation=self._get_active_conversation,
            should_prefetch_channel_context=self._should_prefetch_channel_context,
            prefetch_channel_history_for_message=self._prefetch_channel_history_for_message,
            user_memory_context_for_message=self._user_memory_context_for_message,
            server_memory_context_for_message=self._server_memory_context_for_message,
            pending_burst_context=self._pending_burst_context_for_builder,
            reply_reference_context=self._reply_reference_context,
            passive_flush_confirmation_context=self.flush_confirmations.context_for_message,
            dm_conversation_context_for_message=self._dm_conversation_context_for_message,
            dm_summary_block_for_user=self._dm_summary_block_for_user,
            message_text_for_context=self._message_text_for_context,
            display_name_for_message_author=self._display_name_for_message_author,
            mentioned_users_context=self._mentioned_users_context,
            mentioned_channels_context=self._mentioned_channels_context,
            tool_schemas_for_agent=self._tool_schemas_for_agent,
            tool_choice_for_message=self._tool_choice_for_message,
            tool_calls_allowed=self.app_config.runtime.tool_policy.allow_model_tool_calls,
        )

    # ------------------------------------------------------------------
    # BotServices protocol surface (used by tool handlers and helpers).
    # ------------------------------------------------------------------

    @property
    def owner_user_id(self) -> int:
        return self.app_config.settings.owner_user_id

    @property
    def user_id(self) -> int | None:
        return self.user.id if self.user is not None else None

    def display_name_for_message_author(self, message: discord.Message) -> str:
        return self._display_name_for_message_author(message)

    def message_text_for_context(self, message: discord.Message) -> str:
        return self._message_text_for_context(message)

    def channel_from_message_context(
        self,
        message: discord.Message,
        channel_id: int,
    ) -> Any:
        return self._channel_from_message_context(message, channel_id)

    def member_from_message_context(
        self,
        message: discord.Message,
        user_id: int,
    ) -> Any:
        return self._member_from_message_context(message, user_id)

    def user_from_message_context(
        self,
        message: discord.Message,
        user_id: int,
    ) -> Any:
        return self._user_from_message_context(message, user_id)

    def resolve_target_user_id(self, message: discord.Message, raw_value: Any) -> int:
        return self._resolve_target_user_id(message, raw_value)

    def store_historical_dm_message(self, message: discord.Message) -> None:
        self._store_historical_dm_message(message)

    def store_outgoing_dm_message(self, message: discord.Message) -> None:
        self._store_outgoing_dm_message(message)

    def is_direct_message(self, message: discord.Message) -> bool:
        return self._is_direct_message(message)

    def max_history_messages(self) -> int:
        return self.app_config.runtime.discord.max_history_messages

    def register_passive_flush_confirmation(
        self,
        *,
        guild_id: int,
        requester_user_id: int,
        pending_message_count: int,
    ) -> None:
        self.flush_confirmations.register(
            guild_id=guild_id,
            requester_user_id=requester_user_id,
            pending_message_count=pending_message_count,
        )

    def consume_passive_flush_confirmation(
        self,
        *,
        guild_id: int,
        requester_user_id: int,
    ) -> Any:
        return self.flush_confirmations.consume(
            guild_id=guild_id,
            requester_user_id=requester_user_id,
        )

    def discard_passive_flush_confirmation(
        self,
        *,
        guild_id: int,
        requester_user_id: int,
    ) -> bool:
        return self.flush_confirmations.discard(
            guild_id=guild_id,
            requester_user_id=requester_user_id,
        )

    def _pending_burst_context_for_builder(
        self,
        pending_messages: list[discord.Message] | None,
        latest_message_id: int,
    ) -> str:
        return self._pending_burst_context(
            pending_messages,
            latest_message_id=latest_message_id,
        )

    async def setup_hook(self) -> None:
        await self.openrouter.setup()
        LOGGER.info(
            "setup_hook complete. OpenRouter client ready for model %s.",
            self.app_config.runtime.openrouter.default_model,
        )

    async def on_ready(self) -> None:
        await self._apply_configured_presence()
        self.passive_poller.ensure_running()

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
                total_tokens=self.token_meter.total,
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
            return template.format(total_tokens=self.token_meter.total)
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
            return template.format(total_tokens=self.token_meter.total)
        except Exception:
            LOGGER.exception(
                "Invalid token usage details template %r. Falling back to default details.",
                template,
            )
            return presence_config.details

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

    def _store_passive_guild_message(self, message: discord.Message) -> None:
        if message.guild is None:
            return
        if not self.passive_summarizer.is_enabled_for_guild(message.guild.id):
            return

        content = self._message_text_for_context(message)
        if not content:
            return

        self.passive_summarizer.store_passive_guild_message(
            guild_id=message.guild.id,
            guild_name=message.guild.name,
            channel_id=message.channel.id,
            channel_name=getattr(message.channel, "name", None),
            author_id=message.author.id,
            author_username=getattr(message.author, "name", str(message.author)),
            content=content,
            discord_message_id=message.id,
            created_at=message.created_at.isoformat() if message.created_at else None,
        )
        running_task = self.passive_summarizer.summary_tasks.get(message.guild.id)
        if running_task is None or running_task.done():
            asyncio.create_task(self.passive_summarizer.maybe_start_summary_for_guild(message.guild.id))

    def _guild_name_for_id(self, guild_id: int) -> str | None:
        guild = self.get_guild(guild_id)
        if guild is not None and getattr(guild, "name", None):
            return guild.name
        return None

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
        self._append_passive_channel_context(message)

        if not self._should_consider_message(message):
            return

        self._store_passive_guild_message(message)
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
        self.in_flight.cancel_all()
        self.typing_registry.cancel_all()
        self.passive_poller.close()
        if self._presence_update_task is not None and not self._presence_update_task.done():
            self._presence_update_task.cancel()
        await self.openrouter.close()
        await super().close()

    def _should_respond_to_message(self, message: discord.Message) -> bool:
        if not self._should_consider_message(message):
            return False
        if message.guild is None:
            return True
        return any(user.id == self.user.id for user in self._iter_mentions(message))

    def _should_consider_message(self, message: discord.Message) -> bool:
        if self.user is None:
            return False

        if message.id in self.response_dedup:
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
        is_dm = self._is_direct_message(message)
        conversation = None if is_dm else self._get_active_conversation(message)
        built = await self.prompt_builder.build(
            message=message,
            pending_messages=pending_messages,
        )
        messages = built.messages
        user_message = built.user_message
        extra_payload = built.extra_payload

        response = await self.tool_loop.run(
            model=selected_agent.model,
            messages=messages,
            extra_payload=extra_payload,
            tool_context=ToolContext(message=message, services=self),
        )
        reply_text = extract_response_text(response, messages=messages)
        if reply_text == "I couldn't produce a normal reply, but I can try again.":
            deterministic = synthesize_deterministic_reply(messages)
            if deterministic is not None:
                reply_text = deterministic
            elif len(messages) < 10:
                recovery_messages = [
                    *messages,
                    {
                        "role": "user",
                        "content": (
                            "Answer the user's actual question directly now in one short reply. "
                            "Do not narrate tool usage; use fetched context silently if relevant."
                        ),
                    },
                ]
                recovery_response = await self.openrouter.chat(
                    model=selected_agent.model,
                    messages=recovery_messages,
                    extra_payload={
                        "temperature": selected_agent.temperature,
                        "max_tokens": selected_agent.max_output_tokens,
                    },
                )
                reply_text = extract_response_text(recovery_response, messages=recovery_messages)
            else:
                reply_text = "Got it — give me a moment, then ask again if I haven't followed up."
        await self.owner_memory_capturer.maybe_capture(
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
        is_direct_mention = self._should_respond_to_message(message)
        has_active_conversation = self._has_active_conversation(message)
        key, _ = self.pending.upsert(
            message,
            is_direct_mention=is_direct_mention,
            has_active_conversation=has_active_conversation,
            now=monotonic(),
        )

        self.typing_registry.ensure_running(key, message.channel)

        existing_task = self.in_flight.get(key)
        if existing_task is not None and not existing_task.done():
            if self.in_flight.is_generating(key):
                LOGGER.info("cancel in-flight generation key=%s for newer message_id=%s", key, message.id)
                existing_task.cancel()
                self.in_flight.set(key, asyncio.create_task(self._process_pending_reply(key)))
                LOGGER.info("spawned replacement reply task key=%s", key)
            else:
                LOGGER.info("reuse existing debounce task key=%s latest_message_id=%s", key, message.id)
            return

        self.in_flight.set(key, asyncio.create_task(self._process_pending_reply(key)))
        LOGGER.info("spawned new reply task key=%s message_id=%s", key, message.id)

    async def _process_pending_reply(self, key: ConversationKey) -> None:
        try:
            LOGGER.info("reply task start key=%s", key)
            message = await self._wait_until_user_idle(key)
            if message is None:
                LOGGER.info("reply task exit key=%s reason=no_pending_message", key)
                return

            LOGGER.info("debounce finished key=%s message_id=%s", key, message.id)
            pending = self.pending.get(key)
            pending_messages = pending.messages if pending is not None else [message]
            should_respond = self._should_respond_to_message(message)
            had_active_conversation = self._has_active_conversation(message)
            if not should_respond and self.flush_confirmations.has_pending_for_message(message):
                LOGGER.info(
                    "reply task treating message_id=%s as passive flush confirmation follow-up",
                    message.id,
                )
                should_respond = True
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

            self.in_flight.mark_generating(key)
            self.response_dedup.add(message.id)
            LOGGER.info("start generation key=%s message_id=%s", key, message.id)
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
            pending = self.pending.get(key)
            message = pending.message if pending is not None else None
            if message is not None:
                LOGGER.exception("Failed to generate or send reply for message %s", message.id)
                await self._send_channel_reply(message, safe_runtime_error_reply(exc))
            else:
                LOGGER.exception("Failed to generate or send reply for conversation %s", key)
        finally:
            current_task = asyncio.current_task()
            self.in_flight.clear_generating(key)
            if self.in_flight.cleanup_if_owner(key, current_task):
                self.pending.pop(key)
                self.debouncer.discard(key)
                self.typing_registry.stop(key)
                LOGGER.info("reply task cleanup key=%s removed_task=yes", key)
            else:
                LOGGER.info("reply task cleanup key=%s removed_task=no", key)

    async def _wait_until_user_idle(self, key: ConversationKey) -> discord.Message | None:
        return await self.debouncer.wait_until_idle(key)

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

        response = await self.openrouter.chat(
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
        decision = extract_response_text(response).strip().upper()
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

    async def _prefetch_channel_history_for_message(self, message: discord.Message) -> str:
        history_payload = await self.tool_registry.dispatch(
            "read_channel_messages",
            {},
            ToolContext(message=message, services=self),
        )
        history_messages = history_payload.get("messages", [])
        if not history_messages:
            return "Recent channel context:\n(none)\n"

        lines = ["Recent channel context:"]
        for item in reversed(history_messages):
            author_name = item.get("author_display_name") or item.get("author_username") or "unknown"
            author_user_id = item.get("author_user_id")
            content = (item.get("content") or "").strip() or "(no text)"
            id_tag = f"[id={author_user_id}] " if author_user_id is not None else ""
            lines.append(f"- {id_tag}{author_name}: {content}")
        lines.append("")
        return "\n".join(lines)

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

    async def _send_dm_to_owner(self, content: str) -> None:
        owner_user_id = self.app_config.settings.owner_user_id
        user = self.get_user(owner_user_id)
        if user is None:
            user = await self.fetch_user(owner_user_id)

        dm_channel = user.dm_channel
        if dm_channel is None:
            dm_channel = await user.create_dm()

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
        self._maybe_schedule_dm_resummary(recipient.id)

    def _dm_conversation_context_for_message(self, message: discord.Message) -> list[dict[str, str]]:
        memory_config = self.app_config.runtime.memory
        max_tail = max(1, memory_config.dm_max_recent_turns)
        _, tail = self.dm_history_store.read_conversation_for_prompt(
            user_id=message.author.id,
            max_tail=max_tail,
        )
        return tail

    def _dm_summary_block_for_user(self, user_id: int) -> str:
        memory_config = self.app_config.runtime.memory
        max_tail = max(1, memory_config.dm_max_recent_turns)
        summary, _ = self.dm_history_store.read_conversation_for_prompt(
            user_id=user_id,
            max_tail=max_tail,
        )
        if not summary:
            return ""
        return f"Older DM summary:\n{summary.strip()}\n"

    def _maybe_schedule_dm_resummary(self, user_id: int) -> None:
        memory_config = self.app_config.runtime.memory
        if memory_config.dm_summary_threshold_turns <= 0:
            return
        summary_through_id, max_message_id = self.dm_history_store.get_summary_state(user_id=user_id)
        unsummarized = max_message_id - summary_through_id
        if unsummarized < memory_config.dm_summary_threshold_turns:
            return
        if user_id in self._dm_summary_in_flight:
            return
        self._dm_summary_in_flight.add(user_id)
        asyncio.create_task(self._run_dm_resummary(user_id))

    async def _run_dm_resummary(self, user_id: int) -> None:
        memory_config = self.app_config.runtime.memory
        try:
            summary_through_id, max_message_id = self.dm_history_store.get_summary_state(user_id=user_id)
            keep_recent = max(0, memory_config.dm_summary_keep_recent_turns)
            cutoff_id = max_message_id - keep_recent
            if cutoff_id <= summary_through_id:
                return
            rows = self.dm_history_store.read_messages_in_range(
                user_id=user_id,
                after_id=summary_through_id,
                before_or_equal_id=cutoff_id,
            )
            if not rows:
                return

            transcript_lines = []
            for row in rows:
                role = row.get("role") or "user"
                content = (row.get("content") or "").strip()
                if not content:
                    continue
                transcript_lines.append(f"{role}: {content}")
            if not transcript_lines:
                return

            previous_summary, _ = self.dm_history_store.read_conversation_for_prompt(
                user_id=user_id,
                max_tail=0,
            )
            summary_model = (
                memory_config.dm_summary_model
                or self.app_config.runtime.passive_guild_memory.primary_model
                or self.app_config.runtime.openrouter.default_model
            )
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Compress this DM transcript into a dense factual summary. "
                        "Preserve durable user details, ongoing topics, decisions, preferences, and unresolved threads. "
                        "Drop greetings, fillers, and small talk. Output plain prose, no headings, no JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        (f"Previous summary:\n{previous_summary.strip()}\n\n" if previous_summary else "")
                        + "New transcript to merge in:\n"
                        + "\n".join(transcript_lines)
                    ),
                },
            ]
            response = await self.openrouter.chat(
                model=summary_model,
                messages=messages,
                extra_payload={
                    "temperature": memory_config.dm_summary_temperature,
                    "max_tokens": memory_config.dm_summary_max_output_tokens,
                },
            )
            choice = response.get("choices", [{}])[0].get("message", {})
            new_summary_text = (choice.get("content") or "").strip()
            if not new_summary_text:
                LOGGER.warning("DM resummary returned empty content user_id=%s", user_id)
                return
            self.dm_history_store.upsert_summary(
                user_id=user_id,
                summary_text=new_summary_text,
                summary_through_id=cutoff_id,
                updated_at=datetime.now(UTC).replace(microsecond=0).isoformat(),
            )
            LOGGER.info(
                "DM resummary updated user_id=%s through_id=%s rows_summarized=%s",
                user_id,
                cutoff_id,
                len(rows),
            )
        except Exception:
            LOGGER.exception("DM resummary failed user_id=%s", user_id)
        finally:
            self._dm_summary_in_flight.discard(user_id)

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
        enabled_tools = {
            tool.name
            for tool in self.app_config.runtime.tool_policy.registry
            if tool.enabled
        }
        return self.tool_registry.schemas_for(
            allowed_tool_names,
            allow_model_tool_calls=self.app_config.runtime.tool_policy.allow_model_tool_calls,
            enabled_names=enabled_tools,
        )

    def _tool_choice_for_message(self, message: discord.Message) -> str | dict[str, Any]:
        if self._is_mention_without_other_text(message):
            return {
                "type": "function",
                "function": {
                    "name": "read_channel_messages",
                },
            }
        return "auto"

    def _has_active_conversation(self, message: discord.Message) -> bool:
        if self._is_direct_message(message):
            return False
        return self._get_active_conversation(message) is not None

    def _has_pending_reply(self, message: discord.Message) -> bool:
        key = self._conversation_key(message)
        task = self.in_flight.get(key)
        return self.pending.has(key) or (task is not None and not task.done())

    def _get_active_conversation(self, message: discord.Message) -> ActiveConversation | None:
        if self._is_direct_message(message):
            return None
        return self.conversation_store.get(self._conversation_key(message))

    def _store_active_conversation(
        self,
        *,
        message: discord.Message,
        prior_conversation: ActiveConversation | None,
        user_message: dict[str, str],
        assistant_reply: str,
    ) -> None:
        prior_messages = prior_conversation.messages if prior_conversation is not None else None
        self.conversation_store.store(
            self._conversation_key(message),
            prior_messages=prior_messages,
            user_message=user_message,
            assistant_reply=assistant_reply,
        )

    def _append_passive_channel_context(self, message: discord.Message) -> None:
        if self.user is None:
            return

        if message.author.id == self.user.id:
            return

        self.conversation_store.mark_interrupted(
            message,
            author_display_name=self._display_name_for_message_author(message),
        )

    @staticmethod
    def _conversation_key(message: discord.Message) -> ConversationKey:
        return conversation_key(message)

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

    def _load_static_system_prompt(self) -> str:
        path = PROMPTS_DIR / "system.md"
        owner_user_id = self.app_config.settings.owner_user_id
        try:
            template = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            LOGGER.warning("Static system prompt file missing path=%s", path)
            return ""
        return template.replace("{owner_user_id}", str(owner_user_id)).strip() + "\n"

    def _assistant_identity_prompt(self) -> str:
        if self._assistant_identity_cache is not None:
            return self._assistant_identity_cache

        owner_user_id = self.app_config.settings.owner_user_id
        if self.user is None:
            text = (
                "Assistant identity:\n"
                "- platform: Discord\n"
                f"- owner_user_id: {owner_user_id}\n"
                "- You are the logged-in user account for this assistant.\n"
            )
            return text

        text = (
            "Assistant identity:\n"
            f"- account_username: {self.user}\n"
            f"- account_user_id: {self.user.id}\n"
            f"- account_mention: {self.user.mention}\n"
            f"- owner_user_id: {owner_user_id}\n"
            "- You are this Discord account. Do not confuse yourself with message author or owner.\n"
        )
        self._assistant_identity_cache = text
        return text

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
