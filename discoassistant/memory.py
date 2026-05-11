from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(slots=True)
class UserMemoryStore:
    base_dir: Path
    max_chars_in_prompt: int = 6000

    def __post_init__(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def path_for_user(self, user_id: int) -> Path:
        return self.base_dir / f"{user_id}.md"

    def read_for_user(self, user_id: int) -> str:
        path = self.path_for_user(user_id)
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8").strip()

    def read_for_prompt(self, user_id: int) -> str:
        content = self.read_for_user(user_id)
        if not content:
            return "User memory:\n(none)\n"

        if len(content) > self.max_chars_in_prompt:
            content = content[-self.max_chars_in_prompt :]
            first_newline = content.find("\n")
            if first_newline != -1:
                content = content[first_newline + 1 :]

        return f"User memory:\n{content}\n"

    def append_for_user(
        self,
        *,
        user_id: int,
        note: str,
        author_display_name: str | None = None,
        source_channel_id: int | None = None,
    ) -> Path:
        path = self.path_for_user(user_id)
        timestamp = datetime.now(UTC).replace(microsecond=0).isoformat()
        safe_note = note.strip()
        if not safe_note:
            raise ValueError("Memory note cannot be empty.")

        if path.exists():
            existing = path.read_text(encoding="utf-8").rstrip()
        else:
            existing = self._new_file_header(
                user_id=user_id,
                author_display_name=author_display_name,
            ).rstrip()

        channel_suffix = f" | channel {source_channel_id}" if source_channel_id is not None else ""
        entry = f"- [{timestamp}{channel_suffix}] {safe_note}"
        new_content = f"{existing}\n{entry}\n"
        path.write_text(new_content, encoding="utf-8")
        return path

    def replace_for_user(
        self,
        *,
        user_id: int,
        old_text: str,
        new_text: str,
    ) -> tuple[Path, bool]:
        path = self.path_for_user(user_id)
        if not path.exists():
            return path, False

        old_value = old_text.strip()
        new_value = new_text.strip()
        if not old_value or not new_value:
            raise ValueError("Both old_text and new_text are required.")

        content = path.read_text(encoding="utf-8")
        if old_value not in content:
            return path, False

        updated = content.replace(old_value, new_value, 1)
        path.write_text(updated, encoding="utf-8")
        return path, True

    @staticmethod
    def _new_file_header(*, user_id: int, author_display_name: str | None) -> str:
        display_name = author_display_name or "unknown"
        return (
            f"# User Memory: {user_id}\n\n"
            f"- User ID: {user_id}\n"
            f"- Last known display name: {display_name}\n"
            f"- Notes:\n"
        )


@dataclass(slots=True)
class GuildMemoryStore:
    base_dir: Path
    max_chars_in_prompt: int = 6000

    def __post_init__(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def path_for_guild(self, guild_id: int) -> Path:
        return self.base_dir / f"{guild_id}.md"

    def read_for_guild(self, guild_id: int) -> str:
        path = self.path_for_guild(guild_id)
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8").strip()

    def read_for_prompt(self, guild_id: int) -> str:
        content = self.read_for_guild(guild_id)
        if not content:
            return "Server memory:\n(none)\n"

        if len(content) > self.max_chars_in_prompt:
            content = content[-self.max_chars_in_prompt :]
            first_newline = content.find("\n")
            if first_newline != -1:
                content = content[first_newline + 1 :]

        return f"Server memory:\n{content}\n"

    def append_for_guild(
        self,
        *,
        guild_id: int,
        note: str,
        guild_name: str | None = None,
        author_display_name: str | None = None,
        source_channel_id: int | None = None,
    ) -> Path:
        path = self.path_for_guild(guild_id)
        timestamp = datetime.now(UTC).replace(microsecond=0).isoformat()
        safe_note = note.strip()
        if not safe_note:
            raise ValueError("Server memory note cannot be empty.")

        if path.exists():
            existing = path.read_text(encoding="utf-8").rstrip()
        else:
            existing = self._new_file_header(
                guild_id=guild_id,
                guild_name=guild_name,
            ).rstrip()

        author_suffix = f" | by {author_display_name}" if author_display_name else ""
        channel_suffix = f" | channel {source_channel_id}" if source_channel_id is not None else ""
        entry = f"- [{timestamp}{channel_suffix}{author_suffix}] {safe_note}"
        new_content = f"{existing}\n{entry}\n"
        path.write_text(new_content, encoding="utf-8")
        return path

    def replace_for_guild(
        self,
        *,
        guild_id: int,
        old_text: str,
        new_text: str,
    ) -> tuple[Path, bool]:
        path = self.path_for_guild(guild_id)
        if not path.exists():
            return path, False

        old_value = old_text.strip()
        new_value = new_text.strip()
        if not old_value or not new_value:
            raise ValueError("Both old_text and new_text are required.")

        content = path.read_text(encoding="utf-8")
        if old_value not in content:
            return path, False

        updated = content.replace(old_value, new_value, 1)
        path.write_text(updated, encoding="utf-8")
        return path, True

    @staticmethod
    def _new_file_header(*, guild_id: int, guild_name: str | None) -> str:
        safe_guild_name = guild_name or "unknown"
        return (
            f"# Server Memory: {guild_id}\n\n"
            f"- Guild ID: {guild_id}\n"
            f"- Last known guild name: {safe_guild_name}\n"
            f"- Notes:\n"
        )
