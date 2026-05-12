from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path


LOGGER = logging.getLogger("discoassistant")


@dataclass(slots=True)
class UserMemoryStore:
    base_dir: Path
    max_chars_in_prompt: int = 6000
    _prompt_cache: dict[int, tuple[float, str]] = field(default_factory=dict, init=False, repr=False)

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
        path = self.path_for_user(user_id)
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            mtime = 0.0
        cached = self._prompt_cache.get(user_id)
        if cached is not None and cached[0] == mtime:
            LOGGER.debug("user memory cache hit user_id=%s", user_id)
            return cached[1]

        content = self.read_for_user(user_id)
        if not content:
            rendered = "User memory:\n(none)\n"
        else:
            if len(content) > self.max_chars_in_prompt:
                content = content[-self.max_chars_in_prompt :]
                first_newline = content.find("\n")
                if first_newline != -1:
                    content = content[first_newline + 1 :]
            rendered = f"User memory:\n{content}\n"

        self._prompt_cache[user_id] = (mtime, rendered)
        LOGGER.debug("user memory cache miss user_id=%s", user_id)
        return rendered

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
