from __future__ import annotations

import logging
from collections import deque
from typing import Any
from collections.abc import Iterable

import aiohttp
import discord

from discoassistant.config import AppConfig, load_app_config


LOGGER = logging.getLogger("discoassistant")


class DiscoAssistant(discord.Client):
    def __init__(self, app_config: AppConfig) -> None:
        super().__init__()
        self.app_config = app_config
        self._startup_announced = False
        self.http_session: aiohttp.ClientSession | None = None
        self._recent_response_ids: deque[int] = deque(maxlen=200)

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
        if not self._should_respond_to_message(message):
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

        response = await self.openrouter_chat(
            model=selected_agent.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Respond to this Discord message as DiscoAssistant.\n"
                        f"Author username: {message.author}\n"
                        f"Author display name: {self._display_name_for_message_author(message)}\n"
                        f"Channel message: {message.content.strip()}\n"
                        "Keep it short."
                    ),
                },
            ],
            extra_payload={
                "temperature": selected_agent.temperature,
                "max_tokens": selected_agent.max_output_tokens,
            },
        )
        return self.extract_response_text(response)

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
