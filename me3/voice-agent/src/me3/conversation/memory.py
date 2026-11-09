"""SQLite session/turn logging for conversation context."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from me3.state.db import apply_migrations, connect, database_path


class ConversationMemory:
    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or database_path()
        self._lock = threading.RLock()
        self.conn = connect(self.db_path)
        apply_migrations(self.conn)
        self.session_id: str | None = None

    def ensure_session(self) -> str:
        with self._lock:
            if self.session_id:
                return self.session_id
            sid = str(uuid4())
            now = datetime.now(timezone.utc).isoformat()
            self.conn.execute(
                "INSERT INTO sessions (id, started_at, status) VALUES (?, ?, ?)",
                (sid, now, "active"),
            )
            self.conn.commit()
            self.session_id = sid
            return sid

    def add_turn(self, role: str, content: str, meta: dict[str, Any] | None = None) -> str:
        with self._lock:
            sid = self.ensure_session()
            tid = str(uuid4())
            now = datetime.now(timezone.utc).isoformat()
            self.conn.execute(
                """
                INSERT INTO turns (id, session_id, role, content, created_at, token_estimate, meta_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tid,
                    sid,
                    role,
                    content,
                    now,
                    max(1, len(content.split())),
                    json.dumps(meta or {}),
                ),
            )
            self.conn.commit()
            return tid

    def recent_messages(self, limit: int = 12) -> list[dict[str, str]]:
        with self._lock:
            sid = self.ensure_session()
            rows = self.conn.execute(
                """
                SELECT role, content FROM turns
                WHERE session_id = ? AND role IN ('user', 'assistant')
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (sid, limit),
            ).fetchall()
            messages = [{"role": r["role"], "content": r["content"]} for r in reversed(list(rows))]
            return messages

    def close_session(self) -> None:
        with self._lock:
            if not self.session_id:
                return
            now = datetime.now(timezone.utc).isoformat()
            self.conn.execute(
                "UPDATE sessions SET ended_at = ?, status = ? WHERE id = ?",
                (now, "ended", self.session_id),
            )
            self.conn.commit()
            self.session_id = None

    def close(self) -> None:
        with self._lock:
            try:
                self.conn.close()
            except sqlite3.Error:
                pass
