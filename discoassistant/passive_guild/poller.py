from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

from discoassistant.config import PassiveGuildMemoryConfig
from discoassistant.passive_guild.summarizer import PassiveGuildSummarizer
from discoassistant.passive_guild_history import PassiveGuildHistoryStore


LOGGER = logging.getLogger("discoassistant")


class PassiveGuildPoller:
    def __init__(
        self,
        *,
        passive_guild_history_store: PassiveGuildHistoryStore,
        passive_config: PassiveGuildMemoryConfig,
        summarizer: PassiveGuildSummarizer,
        is_closed: Callable[[], bool],
    ) -> None:
        self._passive_guild_history_store = passive_guild_history_store
        self._passive_config = passive_config
        self._summarizer = summarizer
        self._is_closed = is_closed
        self._poller_task: asyncio.Task[None] | None = None

    @property
    def poller_task(self) -> asyncio.Task[None] | None:
        return self._poller_task

    def ensure_running(self) -> None:
        if not self._passive_config.enabled:
            return
        if self._poller_task is not None and not self._poller_task.done():
            return
        self._poller_task = asyncio.create_task(self._poller_loop())

    async def _poller_loop(self) -> None:
        poll_interval = max(5, self._passive_config.poll_interval_seconds)
        try:
            while not self._is_closed():
                try:
                    await self._scan()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    LOGGER.exception("Passive guild memory scan failed.")
                await asyncio.sleep(poll_interval)
        except asyncio.CancelledError:
            raise
        finally:
            self._poller_task = None

    async def _scan(self) -> None:
        enabled_guild_ids = self._passive_config.enabled_guild_ids
        guild_ids = self._passive_guild_history_store.list_guild_ids_with_pending_messages(
            enabled_guild_ids=enabled_guild_ids,
        )
        for guild_id in guild_ids:
            await self._summarizer.maybe_start_summary_for_guild(guild_id)

    def cancel_all_tasks(self) -> None:
        for task in self._summarizer.summary_tasks.values():
            if not task.done():
                task.cancel()
        if self._poller_task is not None and not self._poller_task.done():
            self._poller_task.cancel()

    def close(self) -> None:
        self.cancel_all_tasks()
