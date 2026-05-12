from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable


LOGGER = logging.getLogger("discoassistant")


class OwnerNotifier:
    def __init__(
        self,
        *,
        send_dm_to_owner: Callable[[str], Awaitable[None]],
    ) -> None:
        self._send_dm_to_owner = send_dm_to_owner

    async def notify_owner_of_guild_memory_update(
        self,
        *,
        guild_id: int,
        guild_name: str,
        summary: str,
    ) -> None:
        content = (
            f"Guild memory updated for {guild_name} ({guild_id}).\n"
            f"{summary.strip()}"
        ).strip()
        await self._send_dm_to_owner(content)
