from __future__ import annotations

import sqlite3
from pathlib import Path


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
