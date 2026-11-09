"""SQLite-backed timers, notes, notifications with idempotency."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from me3.state.db import apply_migrations, connect, database_path


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class TaskStore:
    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or database_path()
        self._lock = threading.RLock()
        self.conn = connect(self.db_path)
        apply_migrations(self.conn)

    def close(self) -> None:
        with self._lock:
            try:
                self.conn.close()
            except sqlite3.Error:
                pass

    def get_idempotent(self, key: str | None) -> dict[str, Any] | None:
        if not key:
            return None
        with self._lock:
            row = self.conn.execute(
                "SELECT result_json FROM tool_idempotency WHERE key = ?",
                (key,),
            ).fetchone()
            if not row:
                return None
            return json.loads(row["result_json"])

    def put_idempotent(self, key: str | None, tool: str, result: dict[str, Any]) -> None:
        if not key:
            return
        with self._lock:
            self.conn.execute(
                """
                INSERT OR IGNORE INTO tool_idempotency (key, tool, result_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (key, tool, json.dumps(result), _utcnow()),
            )
            self.conn.commit()

    # --- timers ---

    def create_timer(
        self,
        *,
        label: str | None,
        deadline_at: str,
        idempotency_key: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if idempotency_key:
                existing = self.conn.execute(
                    "SELECT * FROM timers WHERE idempotency_key = ?",
                    (idempotency_key,),
                ).fetchone()
                if existing:
                    return self._timer_row(existing)
            tid = str(uuid4())
            now = _utcnow()
            self.conn.execute(
                """
                INSERT INTO timers
                  (id, label, deadline_at, status, created_at, updated_at, notified_at, idempotency_key, meta_json)
                VALUES (?, ?, ?, 'active', ?, ?, NULL, ?, ?)
                """,
                (tid, label, deadline_at, now, now, idempotency_key, json.dumps(meta or {})),
            )
            self.conn.commit()
            row = self.conn.execute("SELECT * FROM timers WHERE id = ?", (tid,)).fetchone()
            return self._timer_row(row)

    def list_timers(self, *, status: str | None = "active") -> list[dict[str, Any]]:
        with self._lock:
            if status:
                rows = self.conn.execute(
                    "SELECT * FROM timers WHERE status = ? ORDER BY deadline_at ASC",
                    (status,),
                ).fetchall()
            else:
                rows = self.conn.execute(
                    "SELECT * FROM timers ORDER BY deadline_at ASC"
                ).fetchall()
            return [self._timer_row(r) for r in rows]

    def get_timer(self, timer_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self.conn.execute("SELECT * FROM timers WHERE id = ?", (timer_id,)).fetchone()
            return self._timer_row(row) if row else None

    def update_timer(
        self,
        timer_id: str,
        *,
        label: str | None = None,
        deadline_at: str | None = None,
        status: str | None = None,
    ) -> dict[str, Any] | None:
        with self._lock:
            row = self.conn.execute("SELECT * FROM timers WHERE id = ?", (timer_id,)).fetchone()
            if not row:
                return None
            new_label = label if label is not None else row["label"]
            new_deadline = deadline_at if deadline_at is not None else row["deadline_at"]
            new_status = status if status is not None else row["status"]
            now = _utcnow()
            self.conn.execute(
                """
                UPDATE timers SET label = ?, deadline_at = ?, status = ?, updated_at = ?
                WHERE id = ?
                """,
                (new_label, new_deadline, new_status, now, timer_id),
            )
            self.conn.commit()
            return self._timer_row(
                self.conn.execute("SELECT * FROM timers WHERE id = ?", (timer_id,)).fetchone()
            )

    def cancel_timer(self, timer_id: str) -> dict[str, Any] | None:
        return self.update_timer(timer_id, status="cancelled")

    def due_active_timers(self, now_iso: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT * FROM timers
                WHERE status = 'active' AND deadline_at <= ?
                ORDER BY deadline_at ASC
                """,
                (now_iso,),
            ).fetchall()
            return [self._timer_row(r) for r in rows]

    def mark_timer_fired(self, timer_id: str) -> dict[str, Any] | None:
        with self._lock:
            now = _utcnow()
            self.conn.execute(
                """
                UPDATE timers SET status = 'fired', notified_at = ?, updated_at = ?
                WHERE id = ? AND status = 'active'
                """,
                (now, now, timer_id),
            )
            self.conn.commit()
            row = self.conn.execute("SELECT * FROM timers WHERE id = ?", (timer_id,)).fetchone()
            return self._timer_row(row) if row else None

    # --- notes ---

    def add_note(
        self,
        *,
        body: str,
        title: str | None = None,
        tags: list[str] | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if idempotency_key:
                existing = self.conn.execute(
                    "SELECT * FROM notes WHERE idempotency_key = ?",
                    (idempotency_key,),
                ).fetchone()
                if existing:
                    return self._note_row(existing)
            nid = str(uuid4())
            now = _utcnow()
            self.conn.execute(
                """
                INSERT INTO notes
                  (id, title, body, tags_json, created_at, updated_at, deleted_at, idempotency_key)
                VALUES (?, ?, ?, ?, ?, ?, NULL, ?)
                """,
                (nid, title, body, json.dumps(tags or []), now, now, idempotency_key),
            )
            self.conn.commit()
            return self._note_row(
                self.conn.execute("SELECT * FROM notes WHERE id = ?", (nid,)).fetchone()
            )

    def search_notes(self, query: str = "", *, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            q = f"%{(query or '').strip()}%"
            if query.strip():
                rows = self.conn.execute(
                    """
                    SELECT * FROM notes
                    WHERE deleted_at IS NULL
                      AND (body LIKE ? OR IFNULL(title, '') LIKE ?)
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    (q, q, limit),
                ).fetchall()
            else:
                rows = self.conn.execute(
                    """
                    SELECT * FROM notes WHERE deleted_at IS NULL
                    ORDER BY updated_at DESC LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            return [self._note_row(r) for r in rows]

    def edit_note(
        self,
        note_id: str,
        *,
        body: str | None = None,
        title: str | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any] | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM notes WHERE id = ? AND deleted_at IS NULL",
                (note_id,),
            ).fetchone()
            if not row:
                return None
            new_body = body if body is not None else row["body"]
            new_title = title if title is not None else row["title"]
            new_tags = json.dumps(tags) if tags is not None else row["tags_json"]
            now = _utcnow()
            self.conn.execute(
                """
                UPDATE notes SET body = ?, title = ?, tags_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (new_body, new_title, new_tags, now, note_id),
            )
            self.conn.commit()
            return self._note_row(
                self.conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
            )

    def delete_note(self, note_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM notes WHERE id = ? AND deleted_at IS NULL",
                (note_id,),
            ).fetchone()
            if not row:
                return None
            now = _utcnow()
            self.conn.execute(
                "UPDATE notes SET deleted_at = ?, updated_at = ? WHERE id = ?",
                (now, now, note_id),
            )
            self.conn.commit()
            return self._note_row(
                self.conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
            )

    # --- notifications ---

    def create_notification(
        self,
        *,
        kind: str,
        payload: dict[str, Any],
        source_kind: str | None = None,
        source_id: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            nid = str(uuid4())
            now = _utcnow()
            self.conn.execute(
                """
                INSERT INTO notifications
                  (id, kind, payload_json, status, created_at, delivered_at, source_kind, source_id)
                VALUES (?, ?, ?, 'pending', ?, NULL, ?, ?)
                """,
                (nid, kind, json.dumps(payload), now, source_kind, source_id),
            )
            self.conn.commit()
            return self._notif_row(
                self.conn.execute("SELECT * FROM notifications WHERE id = ?", (nid,)).fetchone()
            )

    def pending_notifications(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT * FROM notifications
                WHERE status = 'pending'
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [self._notif_row(r) for r in rows]

    def mark_notification_delivered(self, notif_id: str) -> dict[str, Any] | None:
        with self._lock:
            now = _utcnow()
            self.conn.execute(
                """
                UPDATE notifications SET status = 'delivered', delivered_at = ?
                WHERE id = ? AND status = 'pending'
                """,
                (now, notif_id),
            )
            self.conn.commit()
            row = self.conn.execute(
                "SELECT * FROM notifications WHERE id = ?", (notif_id,)
            ).fetchone()
            return self._notif_row(row) if row else None

    def _timer_row(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "label": row["label"],
            "deadline_at": row["deadline_at"],
            "status": row["status"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "notified_at": row["notified_at"],
            "idempotency_key": row["idempotency_key"],
            "meta": json.loads(row["meta_json"] or "{}"),
        }

    def _note_row(self, row: sqlite3.Row) -> dict[str, Any]:
        keys = row.keys()
        return {
            "id": row["id"],
            "title": row["title"],
            "body": row["body"],
            "tags": json.loads(row["tags_json"] or "[]"),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "deleted_at": row["deleted_at"],
            "idempotency_key": row["idempotency_key"] if "idempotency_key" in keys else None,
        }

    def _notif_row(self, row: sqlite3.Row) -> dict[str, Any]:
        keys = row.keys()
        return {
            "id": row["id"],
            "kind": row["kind"],
            "payload": json.loads(row["payload_json"] or "{}"),
            "status": row["status"],
            "created_at": row["created_at"],
            "delivered_at": row["delivered_at"],
            "source_kind": row["source_kind"] if "source_kind" in keys else None,
            "source_id": row["source_id"] if "source_id" in keys else None,
        }
