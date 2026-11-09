"""SQLite connection helpers and migration runner."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from me3.config import load_config, resolve_path

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def database_path(data_dir: Path | None = None) -> Path:
    if data_dir is None:
        cfg = load_config()
        data_dir = resolve_path(cfg.paths.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "me3.db"


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or database_path()
    # Conversation LLM worker may log turns off the main thread.
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def apply_migrations(conn: sqlite3.Connection, migrations_dir: Path | None = None) -> list[str]:
    directory = migrations_dir or MIGRATIONS_DIR
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            id TEXT PRIMARY KEY,
            applied_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    applied = {
        row["id"]
        for row in conn.execute("SELECT id FROM schema_migrations").fetchall()
    }
    newly: list[str] = []
    for sql_path in sorted(directory.glob("*.sql")):
        mid = sql_path.name
        if mid in applied:
            continue
        script = sql_path.read_text(encoding="utf-8")
        conn.executescript(script)
        conn.execute("INSERT INTO schema_migrations (id) VALUES (?)", (mid,))
        newly.append(mid)
    conn.commit()
    return newly
