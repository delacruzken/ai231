"""Publication service: validate → commit → Pages check."""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from me3.config import ROOT, load_secrets, resolve_path
from me3.mcp.common import ErrorKind, RequestMeta, ToolError, ToolResult
from me3.publication.github import GitHubPublisher
from me3.publication.mock import MockPublisher
from me3.publication.sanitize import validate_source

logger = logging.getLogger("me3.publication.service")

Publisher = GitHubPublisher | MockPublisher


def make_publication(
    *,
    force_mock: bool | None = None,
    db_path: Path | None = None,
) -> tuple[Publisher, str]:
    secrets = load_secrets()
    pat = (secrets.github_pat or "").strip()
    owner = (secrets.github_owner or "").strip()
    repo = (secrets.github_repo or "").strip()
    branch = (secrets.github_branch or "main").strip() or "main"
    site_root = (getattr(secrets, "github_site_root", None) or "docs").strip() or "docs"
    pages_url = (getattr(secrets, "github_pages_url", None) or "").strip() or None

    use_mock = force_mock if force_mock is not None else not (pat and owner and repo)
    if use_mock:
        return MockPublisher(owner=owner or "mock-owner", repo=repo or "mock-site", site_root=site_root), "mock"
    return (
        GitHubPublisher(
            token=pat,
            owner=owner,
            repo=repo,
            branch=branch,
            site_root=site_root,
            pages_url=pages_url,
        ),
        "github",
    )


class PublicationService:
    def __init__(
        self,
        *,
        force_mock: bool | None = None,
        db_path: Path | None = None,
        publisher: Publisher | None = None,
        provider: str | None = None,
    ):
        if publisher is not None:
            self.backend = publisher
            self.provider = provider or getattr(publisher, "provider", "mock")
        else:
            self.backend, self.provider = make_publication(force_mock=force_mock)
        self.db_path = db_path or resolve_path("data/me3.sqlite")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()
        logger.info("publication provider=%s", self.provider)

    def _ensure_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS publications (
                    id TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    source_dir TEXT,
                    commit_sha TEXT,
                    pages_url TEXT,
                    status TEXT NOT NULL,
                    file_count INTEGER DEFAULT 0,
                    message TEXT,
                    meta_json TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def health(self) -> dict[str, Any]:
        h = self.backend.health()
        h["provider"] = self.provider
        return h

    def handle(self, tool: str, meta: RequestMeta | None = None, **kwargs: Any) -> ToolResult:
        meta = meta or RequestMeta()
        try:
            data = self._dispatch(tool, meta=meta, **kwargs)
        except ValueError as exc:
            return ToolResult(
                ok=False,
                request_id=meta.request_id,
                error=ToolError(kind=ErrorKind.NEEDS_INPUT, message=str(exc)),
            )
        except RuntimeError as exc:
            kind = ErrorKind.NEEDS_APPROVAL if "confirm=true" in str(exc) else ErrorKind.FAILED
            if "private content" in str(exc).lower() or "blocked name" in str(exc).lower():
                kind = ErrorKind.FAILED
            return ToolResult(
                ok=False,
                request_id=meta.request_id,
                error=ToolError(kind=kind, message=str(exc)),
            )
        except Exception as exc:
            logger.exception("publication %s failed", tool)
            return ToolResult(
                ok=False,
                request_id=meta.request_id,
                error=ToolError(kind=ErrorKind.FAILED, message=str(exc)),
            )
        return ToolResult(ok=True, request_id=meta.request_id, data=data)

    def _dispatch(self, tool: str, *, meta: RequestMeta, **kwargs: Any) -> dict[str, Any]:
        if tool == "publish.validate":
            source = kwargs.get("source_dir") or kwargs.get("path") or ""
            if not source:
                raise ValueError("source_dir is required")
            report = validate_source(Path(source))
            report["provider"] = self.provider
            return report
        if tool == "publish.status":
            pub_id = kwargs.get("publication_id") or kwargs.get("id")
            if pub_id:
                row = self._get(pub_id)
                if not row:
                    raise RuntimeError(f"publication not found: {pub_id}")
                return {"publication": row, "provider": self.provider, "health": self.health()}
            return {
                "provider": self.provider,
                "health": self.health(),
                "pages": self.backend.pages_info(),
                "recent": self.list_recent(limit=5),
            }
        if tool == "publish.site":
            return self.publish_site(
                source_dir=kwargs.get("source_dir") or kwargs.get("path") or "",
                message=kwargs.get("message"),
                confirm=bool(kwargs.get("confirm", False)),
                expect_substring=kwargs.get("expect_substring"),
                idempotency_key=meta.idempotency_key,
            )
        raise ValueError(f"unsupported tool: {tool}")

    def publish_site(
        self,
        *,
        source_dir: str,
        message: str | None = None,
        confirm: bool = False,
        expect_substring: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        if not (source_dir or "").strip():
            raise ValueError("source_dir is required")
        src = Path(source_dir)
        if not src.is_absolute():
            src = (ROOT / src).resolve()
        else:
            src = src.resolve()

        # Always validate first (also surfaces private-data rejects)
        report = validate_source(src)
        if not report["ok"]:
            raise RuntimeError(
                "publish blocked by validation: " + "; ".join(report.get("rejects") or ["unknown"])
            )

        if not confirm:
            raise RuntimeError(
                "Publication needs confirmation. Re-call publish.site with confirm=true "
                f"after reviewing {report['file_count']} file(s)."
            )

        if idempotency_key:
            existing = self._get_by_message_key(idempotency_key)
            if existing:
                return {"publication": existing, "idempotent_replay": True, "provider": self.provider}

        result = self.backend.publish_directory(src, message=message, confirm=True)
        pages_check = self.backend.validate_pages(
            expect_substring=expect_substring,
            retries=2 if self.provider == "github" else 1,
            delay_s=2.0 if self.provider == "github" else 0.0,
        )
        pub_id = uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        status = "published" if pages_check.get("ok") or self.provider == "mock" else "committed_pages_pending"
        if self.provider == "mock":
            status = "published"
            pages_check["ok"] = True

        row = {
            "id": pub_id,
            "provider": self.provider,
            "source_dir": str(src),
            "commit_sha": result.get("commit_sha"),
            "pages_url": result.get("pages_url"),
            "status": status,
            "file_count": result.get("file_count") or 0,
            "message": message or result.get("message") or "",
            "meta_json": _json(
                {
                    "files": result.get("files"),
                    "commit_url": result.get("commit_url"),
                    "pages_check": pages_check,
                    "idempotency_key": idempotency_key,
                    "validation": report,
                }
            ),
            "created_at": now,
        }
        self._insert(row)
        return {
            "publication": row,
            "commit": {
                "sha": result.get("commit_sha"),
                "url": result.get("commit_url"),
                "files": result.get("files"),
            },
            "pages": pages_check,
            "provider": self.provider,
            "validation": report,
        }

    def list_recent(self, limit: int = 5) -> list[dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM publications ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
            return [dict(r) for r in cur.fetchall()]

    def _insert(self, row: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO publications
                (id, provider, source_dir, commit_sha, pages_url, status, file_count, message, meta_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["id"],
                    row["provider"],
                    row["source_dir"],
                    row["commit_sha"],
                    row["pages_url"],
                    row["status"],
                    row["file_count"],
                    row["message"],
                    row["meta_json"],
                    row["created_at"],
                ),
            )
            self._conn.commit()

    def _get(self, pub_id: str) -> dict[str, Any] | None:
        with self._lock:
            cur = self._conn.execute("SELECT * FROM publications WHERE id = ?", (pub_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def _get_by_message_key(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM publications WHERE meta_json LIKE ? ORDER BY created_at DESC LIMIT 1",
                (f'%\"idempotency_key\": \"{key}\"%',),
            )
            row = cur.fetchone()
            return dict(row) if row else None


def _json(obj: Any) -> str:
    import json

    return json.dumps(obj, default=str)
