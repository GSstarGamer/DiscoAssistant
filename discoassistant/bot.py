from __future__ import annotations

import logging
from typing import Any

import aiohttp
import discord
from discord.ext import commands

from discoassistant.config import Settings, load_settings


LOGGER = logging.getLogger("discoassistant")


class DiscoAssistant(commands.Bot):
    def __init__(self, settings: Settings) -> None:
        intents = discord.Intents.default()
        intents.message_content = True

        super().__init__(
            command_prefix=settings.command_prefix,
            self_bot=True,
            intents=intents,
        )
        self.settings = settings
        self._startup_announced = False
        self.http_session: aiohttp.ClientSession | None = None

    async def setup_hook(self) -> None:
        self.http_session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {self.settings.openrouter_api_key}",
                "Content-Type": "application/json",
            }
        )
        LOGGER.info("setup_hook complete. Async HTTP session created and startup is preparing.")

    async def on_ready(self) -> None:
        if self._startup_announced:
            LOGGER.info("Reconnected as %s (%s)", self.user, self.user.id if self.user else "unknown")
            return

        self._startup_announced = True
        LOGGER.info("Logged in as %s (%s)", self.user, self.user.id if self.user else "unknown")
        LOGGER.info("Watching %s guild(s). Prefix is %r.", len(self.guilds), self.command_prefix)
        print(f"DiscoAssistant is ready as {self.user}")

    async def on_command_error(self, ctx: commands.Context, error: commands.CommandError) -> None:
        if isinstance(error, commands.CommandNotFound):
            return

        LOGGER.exception("Command error in %s", getattr(ctx.command, "qualified_name", "unknown"), exc_info=error)

    async def close(self) -> None:
        if self.http_session is not None and not self.http_session.closed:
            await self.http_session.close()
        await super().close()

    async def openrouter_chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        extra_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.http_session is None:
            raise RuntimeError("HTTP session has not been created yet.")

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if extra_payload:
            payload.update(extra_payload)

        async with self.http_session.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json=payload,
        ) as response:
            response.raise_for_status()
            return await response.json()


bot: DiscoAssistant | None = None


def main() -> None:
    global bot

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    settings = load_settings()
    bot = DiscoAssistant(settings)
    bot.run(settings.discord_token)
