"""SQLite job registry with revisions and idempotency."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from me3.state.db import apply_migrations, connect, database_path

ACTIVE_STATUSES = frozenset({"queued", "running", "needs_input", "needs_approval"})


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobRegistry:
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

    def active_job(self) -> dict[str, Any] | None:
        with self._lock:
            row = self.conn.execute(
                f"""
                SELECT * FROM jobs
                WHERE status IN ({",".join("?" * len(ACTIVE_STATUSES))})
                ORDER BY created_at ASC
                LIMIT 1
                """,
                tuple(ACTIVE_STATUSES),
            ).fetchone()
            return self._job_row(row) if row else None

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self.conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            return self._job_row(row) if row else None

    def get_by_idempotency(self, key: str | None) -> dict[str, Any] | None:
        if not key:
            return None
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM jobs WHERE idempotency_key = ?", (key,)
            ).fetchone()
            return self._job_row(row) if row else None

    def create(
        self,
        *,
        specification: str,
        harness: str = "mock",
        model: str | None = None,
        workspace_path: str | None = None,
        instructions: str | None = None,
        idempotency_key: str | None = None,
        budget: dict[str, Any] | None = None,
        meta_status: str = "queued",
    ) -> dict[str, Any]:
        with self._lock:
            if idempotency_key:
                existing = self.get_by_idempotency(idempotency_key)
                if existing:
                    return existing
            active = self.active_job()
            if active:
                raise RuntimeError(
                    f"one active job only; busy with {active['id']} ({active['status']})"
                )
            jid = str(uuid4())
            now = _utcnow()
            self.conn.execute(
                """
                INSERT INTO jobs (
                  id, status, specification, harness, model, workspace_path, instructions,
                  attempts, budget_json, pending_user_request, artifacts_json, checks_json,
                  outcome, created_at, updated_at, idempotency_key
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, NULL, '[]', '[]', NULL, ?, ?, ?)
                """,
                (
                    jid,
                    meta_status,
                    specification,
                    harness,
                    model,
                    workspace_path,
                    instructions,
                    json.dumps(budget or {}),
                    now,
                    now,
                    idempotency_key,
                ),
            )
            self.conn.commit()
            job = self.get(jid)
            assert job
            self.add_revision(jid, f"created: {specification[:200]}")
            return job

    def update(
        self,
        job_id: str,
        *,
        status: str | None = None,
        pending_user_request: str | None = None,
        clear_pending: bool = False,
        artifacts: list[Any] | None = None,
        checks: list[Any] | None = None,
        outcome: str | None = None,
        attempts_inc: bool = False,
        workspace_path: str | None = None,
        harness: str | None = None,
    ) -> dict[str, Any] | None:
        with self._lock:
            row = self.conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if not row:
                return None
            now = _utcnow()
            new_status = status if status is not None else row["status"]
            new_pending = row["pending_user_request"]
            if clear_pending:
                new_pending = None
            if pending_user_request is not None:
                new_pending = pending_user_request
            new_artifacts = json.dumps(artifacts) if artifacts is not None else row["artifacts_json"]
            new_checks = json.dumps(checks) if checks is not None else row["checks_json"]
            new_outcome = outcome if outcome is not None else row["outcome"]
            attempts = int(row["attempts"]) + (1 if attempts_inc else 0)
            new_ws = workspace_path if workspace_path is not None else row["workspace_path"]
            new_harness = harness if harness is not None else row["harness"]
            self.conn.execute(
                """
                UPDATE jobs SET
                  status = ?, pending_user_request = ?, artifacts_json = ?, checks_json = ?,
                  outcome = ?, attempts = ?, workspace_path = ?, harness = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    new_status,
                    new_pending,
                    new_artifacts,
                    new_checks,
                    new_outcome,
                    attempts,
                    new_ws,
                    new_harness,
                    now,
                    job_id,
                ),
            )
            self.conn.commit()
            return self.get(job_id)

    def add_revision(self, job_id: str, content: str) -> dict[str, Any]:
        with self._lock:
            row = self.conn.execute(
                "SELECT COALESCE(MAX(revision_no), 0) AS m FROM job_revisions WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            rev_no = int(row["m"]) + 1
            rid = str(uuid4())
            now = _utcnow()
            self.conn.execute(
                """
                INSERT INTO job_revisions (id, job_id, revision_no, content, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (rid, job_id, rev_no, content, now),
            )
            self.conn.commit()
            return {"id": rid, "job_id": job_id, "revision_no": rev_no, "content": content, "created_at": now}

    def revisions(self, job_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT * FROM job_revisions WHERE job_id = ?
                ORDER BY revision_no ASC
                """,
                (job_id,),
            ).fetchall()
            return [
                {
                    "id": r["id"],
                    "job_id": r["job_id"],
                    "revision_no": r["revision_no"],
                    "content": r["content"],
                    "created_at": r["created_at"],
                }
                for r in rows
            ]

    def list_jobs(self, *, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [self._job_row(r) for r in rows]

    def cleanup_terminal(self, *, keep: int = 20) -> int:
        """Drop oldest terminal jobs beyond keep count (and their revisions)."""
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT id FROM jobs
                WHERE status IN ('completed', 'cancelled', 'failed')
                ORDER BY updated_at DESC
                """
            ).fetchall()
            drop = [r["id"] for r in rows[keep:]]
            for jid in drop:
                self.conn.execute("DELETE FROM job_revisions WHERE job_id = ?", (jid,))
                self.conn.execute("DELETE FROM artifacts WHERE job_id = ?", (jid,))
                self.conn.execute("DELETE FROM jobs WHERE id = ?", (jid,))
            self.conn.commit()
            return len(drop)

    def _job_row(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "status": row["status"],
            "specification": row["specification"],
            "harness": row["harness"],
            "model": row["model"],
            "workspace_path": row["workspace_path"],
            "instructions": row["instructions"],
            "attempts": row["attempts"],
            "budget": json.loads(row["budget_json"] or "{}"),
            "pending_user_request": row["pending_user_request"],
            "artifacts": json.loads(row["artifacts_json"] or "[]"),
            "checks": json.loads(row["checks_json"] or "[]"),
            "outcome": row["outcome"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "idempotency_key": row["idempotency_key"],
        }
