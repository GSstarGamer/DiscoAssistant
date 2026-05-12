from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from typing import Any

from discoassistant.config import PassiveGuildMemoryConfig
from discoassistant.llm.openrouter import OpenRouterClient
from discoassistant.llm.response import extract_response_text
from discoassistant.memory import GuildMemoryStore
from discoassistant.passive_guild.notifier import OwnerNotifier
from discoassistant.passive_guild_history import PassiveGuildHistoryStore


LOGGER = logging.getLogger("discoassistant")


class PassiveGuildSummarizer:
    def __init__(
        self,
        *,
        openrouter_client: OpenRouterClient,
        passive_guild_history_store: PassiveGuildHistoryStore,
        guild_memory_store: GuildMemoryStore,
        passive_config: PassiveGuildMemoryConfig,
        owner_notifier: OwnerNotifier,
        get_guild_name: Callable[[int], str | None],
    ) -> None:
        self._openrouter = openrouter_client
        self._passive_guild_history_store = passive_guild_history_store
        self._guild_memory_store = guild_memory_store
        self._passive_config = passive_config
        self._owner_notifier = owner_notifier
        self._get_guild_name = get_guild_name
        self._summary_tasks: dict[int, asyncio.Task[None]] = {}

    @property
    def summary_tasks(self) -> dict[int, asyncio.Task[None]]:
        return self._summary_tasks

    def is_enabled_for_guild(self, guild_id: int) -> bool:
        if not self._passive_config.enabled:
            return False
        if self._passive_config.enabled_guild_ids is None:
            return True
        return guild_id in self._passive_config.enabled_guild_ids

    def pending_count(self, guild_id: int) -> int:
        return len(self._passive_guild_history_store.read_batch_for_guild(guild_id=guild_id))

    def flush_threshold_tokens(self) -> int:
        return max(
            1024,
            int(self._passive_config.effective_context_limit_tokens * self._passive_config.flush_ratio),
        )

    def estimate_prompt_tokens(
        self,
        *,
        guild_id: int,
        guild_name: str,
        current_memory: str,
        rows: list[dict[str, Any]],
    ) -> int:
        prompt_text = self._summary_prompt_text(
            guild_id=guild_id,
            guild_name=guild_name,
            current_memory=current_memory,
            rows=rows,
        )
        return (len(prompt_text) // 4) + self._passive_config.max_output_tokens

    def latest_guild_name(self, guild_id: int, rows: list[dict[str, Any]]) -> str:
        guild_name = self._get_guild_name(guild_id)
        if guild_name:
            return guild_name
        for row in reversed(rows):
            row_guild_name = str(row.get("guild_name") or "").strip()
            if row_guild_name:
                return row_guild_name
        return "unknown"

    def store_passive_guild_message(
        self,
        *,
        guild_id: int,
        guild_name: str | None,
        channel_id: int,
        channel_name: str | None,
        author_id: int,
        author_username: str,
        content: str,
        discord_message_id: int | None,
        created_at: str | None,
    ) -> None:
        if not self.is_enabled_for_guild(guild_id):
            return
        if not content:
            return

        self._passive_guild_history_store.append_message(
            guild_id=guild_id,
            guild_name=guild_name,
            channel_id=channel_id,
            channel_name=channel_name,
            author_id=author_id,
            author_username=author_username,
            content=content,
            discord_message_id=discord_message_id,
            created_at=created_at,
        )

    async def maybe_start_summary_for_guild(self, guild_id: int) -> None:
        if not self.is_enabled_for_guild(guild_id):
            return

        running_task = self._summary_tasks.get(guild_id)
        if running_task is not None and not running_task.done():
            return

        rows = self._passive_guild_history_store.read_batch_for_guild(guild_id=guild_id)
        if not rows:
            return

        current_memory = self._guild_memory_store.read_for_guild(guild_id)
        estimated_tokens = self.estimate_prompt_tokens(
            guild_id=guild_id,
            guild_name=self.latest_guild_name(guild_id, rows),
            current_memory=current_memory,
            rows=rows,
        )
        if estimated_tokens < self.flush_threshold_tokens():
            return

        self._summary_tasks[guild_id] = asyncio.create_task(
            self.run_summary(guild_id)
        )

    async def run_summary(self, guild_id: int, *, force: bool = False) -> dict[str, Any]:
        try:
            rows = self._passive_guild_history_store.read_batch_for_guild(guild_id=guild_id)
            if not rows:
                return {
                    "ok": True,
                    "guild_id": guild_id,
                    "status": "no_pending_rows",
                    "messages_processed": 0,
                }

            guild_name = self.latest_guild_name(guild_id, rows)
            current_memory = self._guild_memory_store.read_for_guild(guild_id)
            summarizer_memory = self._guild_memory_store.strip_owner_rules_section(current_memory).strip()
            if (
                not force
                and self.estimate_prompt_tokens(
                    guild_id=guild_id,
                    guild_name=guild_name,
                    current_memory=summarizer_memory,
                    rows=rows,
                ) < self.flush_threshold_tokens()
            ):
                return {
                    "ok": True,
                    "guild_id": guild_id,
                    "status": "below_threshold",
                    "messages_processed": 0,
                }

            selected_rows = self._select_rows_for_flush(
                guild_id=guild_id,
                guild_name=guild_name,
                current_memory=summarizer_memory,
                rows=rows,
            )
            if not selected_rows:
                return {
                    "ok": False,
                    "guild_id": guild_id,
                    "status": "selection_failed",
                    "messages_processed": 0,
                    "error_message": "Passive flush could not select any rows for summarization.",
                }

            summary_payload = await self._summarize_rows(
                guild_id=guild_id,
                guild_name=guild_name,
                current_memory=summarizer_memory,
                rows=selected_rows,
            )
            rendered_memory = self._render_memory_snapshot(
                guild_id=guild_id,
                guild_name=guild_name,
                summary_payload=summary_payload,
            )
            self._guild_memory_store.write_for_guild(
                guild_id=guild_id,
                content=rendered_memory,
            )
            await self._owner_notifier.notify_owner_of_guild_memory_update(
                guild_id=guild_id,
                guild_name=guild_name,
                summary=self._owner_summary(summary_payload),
            )
            self._passive_guild_history_store.delete_messages_through_row_id(
                guild_id=guild_id,
                max_row_id=int(selected_rows[-1]["row_id"]),
            )
            LOGGER.info(
                "Passive guild memory updated guild_id=%s messages=%s",
                guild_id,
                len(selected_rows),
            )
            return {
                "ok": True,
                "guild_id": guild_id,
                "status": "updated",
                "messages_processed": len(selected_rows),
                "last_processed_row_id": int(selected_rows[-1]["row_id"]),
            }
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            LOGGER.exception("Passive guild summarization failed guild_id=%s", guild_id)
            return {
                "ok": False,
                "guild_id": guild_id,
                "status": "failed",
                "messages_processed": 0,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        finally:
            current_task = self._summary_tasks.get(guild_id)
            if current_task is asyncio.current_task():
                self._summary_tasks.pop(guild_id, None)

    async def flush_now(self, guild_id: int) -> dict[str, Any]:
        total_processed = 0
        iterations = 0
        max_iterations = 50

        while iterations < max_iterations:
            iterations += 1
            running_task = self._summary_tasks.get(guild_id)
            if running_task is not None and not running_task.done():
                result = await running_task
            else:
                result = await self.run_summary(guild_id, force=True)

            processed = int(result.get("messages_processed", 0) or 0)
            total_processed += processed
            pending_rows = self._passive_guild_history_store.read_batch_for_guild(guild_id=guild_id)

            if result.get("ok") is False:
                return {
                    "ok": False,
                    "guild_id": guild_id,
                    "status": "failed",
                    "iterations": iterations,
                    "messages_processed": total_processed,
                    "pending_message_count": len(pending_rows),
                    "error_type": result.get("error_type"),
                    "error_message": result.get("error_message", "Passive flush failed."),
                }

            if not pending_rows:
                return {
                    "ok": True,
                    "guild_id": guild_id,
                    "status": "flushed",
                    "iterations": iterations,
                    "messages_processed": total_processed,
                    "pending_message_count": 0,
                }

            if processed <= 0:
                return {
                    "ok": False,
                    "guild_id": guild_id,
                    "status": "stalled",
                    "iterations": iterations,
                    "messages_processed": total_processed,
                    "pending_message_count": len(pending_rows),
                    "error_message": "Passive flush made no progress while messages still remained queued.",
                }

        pending_rows = self._passive_guild_history_store.read_batch_for_guild(guild_id=guild_id)
        return {
            "ok": False,
            "guild_id": guild_id,
            "status": "iteration_limit",
            "iterations": iterations,
            "messages_processed": total_processed,
            "pending_message_count": len(pending_rows),
            "error_message": "Passive flush hit iteration limit before draining the queue.",
        }

    def _select_rows_for_flush(
        self,
        *,
        guild_id: int,
        guild_name: str,
        current_memory: str,
        rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        prompt_budget_tokens = max(
            1024,
            self.flush_threshold_tokens() - self._passive_config.max_output_tokens,
        )
        prompt_budget_chars = prompt_budget_tokens * 4
        base_chars = len(
            self._summary_prompt_text(
                guild_id=guild_id,
                guild_name=guild_name,
                current_memory=current_memory,
                rows=[],
            )
        )

        selected_rows: list[dict[str, Any]] = []
        active_channel_id: int | None = None
        current_chars = base_chars
        for row in rows:
            channel_header = ""
            if row["channel_id"] != active_channel_id:
                channel_name = row.get("channel_name") or f"channel-{row['channel_id']}"
                channel_header = f"\n## Channel: {channel_name} ({row['channel_id']})\n"

            row_line = self._row_text(row)
            projected_chars = current_chars + len(channel_header) + len(row_line)
            if selected_rows and projected_chars > prompt_budget_chars:
                break

            selected_rows.append(row)
            current_chars = projected_chars
            active_channel_id = int(row["channel_id"])

        if not selected_rows and rows:
            return [rows[0]]
        return selected_rows

    async def _summarize_rows(
        self,
        *,
        guild_id: int,
        guild_name: str,
        current_memory: str,
        rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        models = [self._passive_config.primary_model]
        if self._passive_config.fallback_model and self._passive_config.fallback_model not in models:
            models.append(self._passive_config.fallback_model)

        messages = [
            {
                "role": "system",
                "content": (
                    "You maintain long-term guild memory for a Discord assistant.\n"
                    "You receive existing guild memory plus a chronological batch of new messages.\n"
                    "Revise memory carefully: preserve useful old facts, remove stale claims, merge duplicates, and add new useful observations.\n"
                    "Capture guild-level norms, channel habits, recurring topics, jokes, relationships, interaction styles, and lightweight observable member traits.\n"
                    "Allowed member traits are only observable behavior such as talkative, kind, likes certain topics, often helps, often jokes, recurring habits, preferred level of detail, or recognizable communication style.\n"
                    "Prefer small reusable cues that would help future replies land better, not just major facts.\n"
                    "Do not infer sensitive traits or hidden attributes.\n"
                    "Return strict JSON only with keys guild_notes, channel_notes, member_notes.\n"
                    "Each key must map to a list.\n"
                    "channel_notes items must contain channel_id, channel_name, notes.\n"
                    "member_notes items must contain user_id, username, notes."
                ),
            },
            {
                "role": "user",
                "content": self._summary_prompt_text(
                    guild_id=guild_id,
                    guild_name=guild_name,
                    current_memory=current_memory,
                    rows=rows,
                ),
            },
        ]

        last_error: Exception | None = None
        for model in models:
            try:
                response = await self._openrouter.chat(
                    model=model,
                    messages=messages,
                    extra_payload={
                        "temperature": self._passive_config.temperature,
                        "max_tokens": self._passive_config.max_output_tokens,
                    },
                    max_attempts=1,
                )
                text = extract_response_text(response, messages=messages)
                return self._parse_summary_json(text)
            except Exception as exc:
                last_error = exc
                LOGGER.warning(
                    "Passive guild summarizer failed model=%s guild_id=%s error=%s",
                    model,
                    guild_id,
                    exc,
                )

        if last_error is not None:
            raise last_error
        raise RuntimeError("Passive guild summarizer failed without a captured exception.")

    def _summary_prompt_text(
        self,
        *,
        guild_id: int,
        guild_name: str,
        current_memory: str,
        rows: list[dict[str, Any]],
    ) -> str:
        memory_text = current_memory.strip() or "(none)"
        messages_text = self._serialize_rows(rows).strip() or "(no messages)"
        return (
            f"Guild ID: {guild_id}\n"
            f"Guild name: {guild_name}\n\n"
            "Current guild memory:\n"
            f"{memory_text}\n\n"
            "New raw guild messages, grouped by channel:\n"
            f"{messages_text}\n"
        )

    def _serialize_rows(self, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return ""

        lines: list[str] = []
        active_channel_id: int | None = None
        for row in rows:
            channel_id = int(row["channel_id"])
            if channel_id != active_channel_id:
                channel_name = row.get("channel_name") or f"channel-{channel_id}"
                lines.append(f"## Channel: {channel_name} ({channel_id})")
                active_channel_id = channel_id
            lines.append(self._row_text(row).rstrip())
        return "\n".join(lines)

    @staticmethod
    def _row_text(row: dict[str, Any]) -> str:
        created_at = row.get("created_at") or "unknown-time"
        author_username = row.get("author_username") or "unknown-user"
        author_id = row.get("author_id")
        content = row.get("content") or ""
        return f"- [{created_at}] {author_username} ({author_id}): {content}\n"

    def _parse_summary_json(self, text: str) -> dict[str, Any]:
        candidate_texts = [text.strip()]
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped.strip("`")
            if stripped.startswith("json"):
                stripped = stripped[4:].lstrip()
            candidate_texts.append(stripped)
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            candidate_texts.append(text[first_brace : last_brace + 1].strip())

        parsed: dict[str, Any] | None = None
        for candidate in candidate_texts:
            if not candidate:
                continue
            try:
                value = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                parsed = value
                break

        if parsed is None:
            raise ValueError("Passive guild summarizer did not return valid JSON.")

        guild_notes = parsed.get("guild_notes")
        channel_notes = parsed.get("channel_notes")
        member_notes = parsed.get("member_notes")
        if not isinstance(guild_notes, list) or not isinstance(channel_notes, list) or not isinstance(member_notes, list):
            raise ValueError("Passive guild summary JSON is missing required list keys.")

        return {
            "guild_notes": self._normalize_notes(guild_notes),
            "channel_notes": [
                {
                    "channel_id": int(item.get("channel_id")),
                    "channel_name": str(item.get("channel_name", "")).strip() or f"channel-{item.get('channel_id')}",
                    "notes": self._normalize_notes(item.get("notes", [])),
                }
                for item in channel_notes
                if isinstance(item, dict) and item.get("channel_id") is not None
            ],
            "member_notes": [
                {
                    "user_id": int(item.get("user_id")),
                    "username": str(item.get("username", "")).strip() or f"user-{item.get('user_id')}",
                    "notes": self._normalize_notes(item.get("notes", [])),
                }
                for item in member_notes
                if isinstance(item, dict) and item.get("user_id") is not None
            ],
        }

    @staticmethod
    def _normalize_notes(raw_notes: Any) -> list[str]:
        if isinstance(raw_notes, str):
            note = raw_notes.strip()
            return [note] if note else []
        if not isinstance(raw_notes, list):
            return []
        return [str(item).strip() for item in raw_notes if str(item).strip()]

    def _render_memory_snapshot(
        self,
        *,
        guild_id: int,
        guild_name: str,
        summary_payload: dict[str, Any],
    ) -> str:
        lines = [
            f"# Server Memory: {guild_id}",
            "",
            f"- Guild ID: {guild_id}",
            f"- Last known guild name: {guild_name}",
            "",
            "## Guild Notes",
        ]

        guild_notes = summary_payload.get("guild_notes", [])
        if guild_notes:
            for note in guild_notes:
                lines.append(f"- {note}")
        else:
            lines.append("- (none)")

        lines.extend(["", "## Channel Notes"])
        channel_notes = summary_payload.get("channel_notes", [])
        if channel_notes:
            for item in channel_notes:
                lines.append(f"### {item['channel_name']} ({item['channel_id']})")
                if item["notes"]:
                    for note in item["notes"]:
                        lines.append(f"- {note}")
                else:
                    lines.append("- (none)")
        else:
            lines.append("- (none)")

        lines.extend(["", "## Member Notes"])
        member_notes = summary_payload.get("member_notes", [])
        if member_notes:
            for item in member_notes:
                lines.append(f"### {item['username']} ({item['user_id']})")
                if item["notes"]:
                    for note in item["notes"]:
                        lines.append(f"- {note}")
                else:
                    lines.append("- (none)")
        else:
            lines.append("- (none)")

        return "\n".join(lines)

    def _owner_summary(self, summary_payload: dict[str, Any]) -> str:
        lines: list[str] = []

        guild_notes = summary_payload.get("guild_notes", [])
        if guild_notes:
            lines.append("Guild notes:")
            for note in guild_notes[:3]:
                lines.append(f"- {note}")

        channel_notes = summary_payload.get("channel_notes", [])
        if channel_notes:
            lines.append("Channel notes:")
            for item in channel_notes[:2]:
                channel_name = item.get("channel_name") or item.get("channel_id")
                note_list = item.get("notes", [])
                if note_list:
                    lines.append(f"- {channel_name}: {note_list[0]}")

        member_notes = summary_payload.get("member_notes", [])
        if member_notes:
            lines.append("Member notes:")
            for item in member_notes[:3]:
                username = item.get("username") or item.get("user_id")
                note_list = item.get("notes", [])
                if note_list:
                    lines.append(f"- {username}: {note_list[0]}")

        if not lines:
            return "Memory snapshot updated, but there were no non-empty summary notes."
        return "\n".join(lines)
