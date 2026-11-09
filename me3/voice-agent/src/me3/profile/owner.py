"""Approved owner public profile — never biometric identity."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from me3.state.db import apply_migrations, connect, database_path

DEFAULT_OWNER_ID = "owner"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class OwnerProfileStore:
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

    def get(self, owner_id: str = DEFAULT_OWNER_ID) -> dict[str, Any]:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM owner_profile WHERE owner_id = ?",
                (owner_id,),
            ).fetchone()
            if not row:
                return {
                    "owner_id": owner_id,
                    "public": {},
                    "preferences": {},
                    "updated_at": None,
                    "approved": False,
                }
            public = json.loads(row["public_json"] or "{}")
            return {
                "owner_id": row["owner_id"],
                "public": public,
                "preferences": json.loads(row["preferences_json"] or "{}"),
                "updated_at": row["updated_at"],
                "approved": bool(public.get("approved")),
            }

    def set_public(
        self,
        public: dict[str, Any],
        *,
        owner_id: str = DEFAULT_OWNER_ID,
        preferences: dict[str, Any] | None = None,
        approved: bool = True,
    ) -> dict[str, Any]:
        """Replace approved public fields. Does not invent private biography."""
        payload = dict(public)
        payload["approved"] = approved
        # Strip fields that must never be treated as identity claims
        for banned in ("biometric", "voiceprint", "speaker_id"):
            payload.pop(banned, None)
        with self._lock:
            existing = self.get(owner_id)
            prefs = preferences if preferences is not None else existing.get("preferences") or {}
            now = _utcnow()
            self.conn.execute(
                """
                INSERT INTO owner_profile (owner_id, public_json, preferences_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(owner_id) DO UPDATE SET
                  public_json = excluded.public_json,
                  preferences_json = excluded.preferences_json,
                  updated_at = excluded.updated_at
                """,
                (owner_id, json.dumps(payload), json.dumps(prefs), now),
            )
            self.conn.commit()
            return self.get(owner_id)

    def clear(self, owner_id: str = DEFAULT_OWNER_ID) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM owner_profile WHERE owner_id = ?", (owner_id,))
            self.conn.commit()


def load_owner_public(db_path: Path | None = None) -> dict[str, Any]:
    store = OwnerProfileStore(db_path)
    try:
        return store.get()
    finally:
        store.close()


def format_owner_for_prompt(profile: dict[str, Any] | None) -> str:
    if not profile or not profile.get("approved"):
        return (
            "Owner public profile: not approved yet. Do not guess personal identity or "
            "private biography. If asked about 'me'/'my site', say the approved profile "
            "is not configured."
        )
    public = profile.get("public") or {}
    safe = {
        k: public[k]
        for k in ("display_name", "headline", "links", "about", "locale")
        if k in public and public[k]
    }
    prefs = profile.get("preferences") or {}
    lines = [
        "You are speaking with the session owner. Treat 'me'/'my'/'I' as this person — "
        "not other web-search name matches.",
        "Owner approved public profile (use only these fields; never invent private details):",
        json.dumps(safe, ensure_ascii=False),
    ]
    if prefs:
        lines.append(
            "Owner preferences (follow these when relevant): "
            + json.dumps(prefs, ensure_ascii=False)
        )
    return "\n".join(lines)


def format_publication_for_prompt(target: dict[str, Any] | None) -> str:
    if not target or not target.get("configured"):
        return (
            "Publication target: not configured. If the owner asks to publish, say GitHub "
            "publication env is missing (owner/repo/PAT)."
        )
    owner = target.get("owner") or ""
    repo = target.get("repo") or ""
    branch = target.get("branch") or "main"
    site_root = target.get("site_root") or "docs"
    pages_url = target.get("pages_url") or ""
    parts = [
        f"Publication target is already configured: GitHub {owner}/{repo} "
        f"(branch {branch}, site root {site_root})."
    ]
    if pages_url:
        parts.append(f"Pages URL: {pages_url}.")
    parts.append(
        "Do not ask for the GitHub account or repo URL. Call publish_status if you need "
        "live Pages health. Still require explicit owner confirmation before publish_site."
    )
    return " ".join(parts)
