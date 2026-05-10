from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"


@dataclass(slots=True)
class Settings:
    discord_token: str
    openrouter_api_key: str
    command_prefix: str = "!"


def load_settings() -> Settings:
    load_dotenv(ENV_PATH)

    discord_token = os.getenv("DISCORD_TOKEN", "").strip()
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    command_prefix = os.getenv("COMMAND_PREFIX", "!").strip() or "!"

    if not discord_token:
        raise RuntimeError(
            "DISCORD_TOKEN is missing. Add it to the project .env file before starting the client."
        )

    if not openrouter_api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is missing. Add it to the project .env file before starting the client."
        )

    return Settings(
        discord_token=discord_token,
        openrouter_api_key=openrouter_api_key,
        command_prefix=command_prefix,
    )
