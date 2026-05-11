from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


class PassiveGuildHistoryStore:
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
                CREATE TABLE IF NOT EXISTS guild_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    guild_id INTEGER NOT NULL,
                    guild_name TEXT,
                    channel_id INTEGER NOT NULL,
                    channel_name TEXT,
                    author_id INTEGER NOT NULL,
                    author_username TEXT NOT NULL,
                    content TEXT NOT NULL,
                    discord_message_id INTEGER,
                    created_at TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_guild_messages_discord_message_id
                ON guild_messages (discord_message_id)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_guild_messages_guild_id_id
                ON guild_messages (guild_id, id)
                """
            )

    def append_message(
        self,
        *,
        guild_id: int,
        guild_name: str | None,
        channel_id: int,
        channel_name: str | None,
        author_id: int,
        author_username: str,
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
                    "SELECT 1 FROM guild_messages WHERE discord_message_id = ? LIMIT 1",
                    (discord_message_id,),
                ).fetchone()
                if existing is not None:
                    return

            connection.execute(
                """
                INSERT INTO guild_messages (
                    guild_id,
                    guild_name,
                    channel_id,
                    channel_name,
                    author_id,
                    author_username,
                    content,
                    discord_message_id,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    guild_id,
                    guild_name,
                    channel_id,
                    channel_name,
                    author_id,
                    author_username,
                    text,
                    discord_message_id,
                    created_at,
                ),
            )

    def list_guild_ids_with_pending_messages(self, *, enabled_guild_ids: list[int] | None = None) -> list[int]:
        query = "SELECT DISTINCT guild_id FROM guild_messages"
        params: tuple[Any, ...] = ()
        if enabled_guild_ids:
            placeholders = ",".join("?" for _ in enabled_guild_ids)
            query += f" WHERE guild_id IN ({placeholders})"
            params = tuple(enabled_guild_ids)
        query += " ORDER BY guild_id ASC"

        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [int(row[0]) for row in rows]

    def read_batch_for_guild(self, *, guild_id: int) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    id,
                    guild_id,
                    guild_name,
                    channel_id,
                    channel_name,
                    author_id,
                    author_username,
                    content,
                    discord_message_id,
                    created_at
                FROM guild_messages
                WHERE guild_id = ?
                ORDER BY id ASC
                """,
                (guild_id,),
            ).fetchall()

        return [
            {
                "row_id": row_id,
                "guild_id": stored_guild_id,
                "guild_name": guild_name,
                "channel_id": channel_id,
                "channel_name": channel_name,
                "author_id": author_id,
                "author_username": author_username,
                "content": content,
                "discord_message_id": discord_message_id,
                "created_at": created_at,
            }
            for (
                row_id,
                stored_guild_id,
                guild_name,
                channel_id,
                channel_name,
                author_id,
                author_username,
                content,
                discord_message_id,
                created_at,
            ) in rows
        ]

    def delete_messages_through_row_id(self, *, guild_id: int, max_row_id: int) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                DELETE FROM guild_messages
                WHERE guild_id = ? AND id <= ?
                """,
                (guild_id, max_row_id),
            )
