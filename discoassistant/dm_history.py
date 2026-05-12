from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


class DmHistoryStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS dm_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    discord_message_id INTEGER,
                    created_at TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_dm_messages_user_id_id
                ON dm_messages (user_id, id)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS dm_summary (
                    user_id INTEGER PRIMARY KEY,
                    summary_text TEXT NOT NULL,
                    summary_through_id INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def append_message(
        self,
        *,
        user_id: int,
        role: str,
        content: str,
        discord_message_id: int | None = None,
        created_at: str | None = None,
    ) -> None:
        text = content.strip()
        if not text:
            return

        with self._connect() as connection:
            if discord_message_id is not None:
                existing = connection.execute(
                    "SELECT 1 FROM dm_messages WHERE discord_message_id = ? LIMIT 1",
                    (discord_message_id,),
                ).fetchone()
                if existing is not None:
                    return

            connection.execute(
                """
                INSERT INTO dm_messages (user_id, role, content, discord_message_id, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user_id, role, text, discord_message_id, created_at),
            )

    def read_conversation(self, *, user_id: int) -> list[dict[str, str]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT role, content
                FROM dm_messages
                WHERE user_id = ?
                ORDER BY id ASC
                """,
                (user_id,),
            ).fetchall()

        return [{"role": role, "content": content} for role, content in rows]

    def read_conversation_for_prompt(
        self,
        *,
        user_id: int,
        max_tail: int,
    ) -> tuple[str | None, list[dict[str, str]]]:
        with self._connect() as connection:
            summary_row = connection.execute(
                "SELECT summary_text, summary_through_id FROM dm_summary WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            summary_through_id = summary_row[1] if summary_row else 0
            tail_rows = connection.execute(
                """
                SELECT role, content
                FROM dm_messages
                WHERE user_id = ? AND id > ?
                ORDER BY id ASC
                """,
                (user_id, summary_through_id),
            ).fetchall()
        summary_text = summary_row[0] if summary_row else None
        tail = [{"role": role, "content": content} for role, content in tail_rows]
        if max_tail > 0 and len(tail) > max_tail:
            tail = tail[-max_tail:]
        return summary_text, tail

    def get_summary_state(self, *, user_id: int) -> tuple[int, int]:
        """Return (summary_through_id, max_message_id) for the user."""
        with self._connect() as connection:
            summary_row = connection.execute(
                "SELECT summary_through_id FROM dm_summary WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            max_row = connection.execute(
                "SELECT COALESCE(MAX(id), 0) FROM dm_messages WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        summary_through_id = summary_row[0] if summary_row else 0
        max_message_id = max_row[0] if max_row else 0
        return summary_through_id, max_message_id

    def read_messages_in_range(
        self,
        *,
        user_id: int,
        after_id: int,
        before_or_equal_id: int,
    ) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, role, content, created_at
                FROM dm_messages
                WHERE user_id = ? AND id > ? AND id <= ?
                ORDER BY id ASC
                """,
                (user_id, after_id, before_or_equal_id),
            ).fetchall()
        return [
            {"id": row[0], "role": row[1], "content": row[2], "created_at": row[3]}
            for row in rows
        ]

    def upsert_summary(
        self,
        *,
        user_id: int,
        summary_text: str,
        summary_through_id: int,
        updated_at: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO dm_summary (user_id, summary_text, summary_through_id, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    summary_text = excluded.summary_text,
                    summary_through_id = excluded.summary_through_id,
                    updated_at = excluded.updated_at
                """,
                (user_id, summary_text, summary_through_id, updated_at),
            )
