"""Smoke checks: imports, config, migrations, MCP handles."""

from __future__ import annotations

from pathlib import Path

from me3 import __version__
from me3.config import AppConfig, load_config, resolve_path
from me3.mcp.asr.server import handle as asr_handle
from me3.mcp.common import ErrorKind, RequestMeta
from me3.mcp.delegation.server import handle as delegation_handle
from me3.mcp.tasks.server import handle as tasks_handle
from me3.mcp.tts.server import handle as tts_handle
from me3.state.db import apply_migrations, connect, database_path


ROOT = Path(__file__).resolve().parents[1]


def test_version():
    assert __version__ == "0.1.0"


def test_load_example_config():
    cfg = load_config(ROOT / "config" / "settings.example.yaml")
    assert isinstance(cfg, AppConfig)
    assert cfg.audio.input_sample_rate == 16000
    assert cfg.pending.anthropic_api_key is True
    assert cfg.paths.fd_benchmark_dir == "../fd-benchmark"


def test_fd_benchmark_path_exists():
    cfg = load_config(ROOT / "config" / "settings.example.yaml")
    path = resolve_path(cfg.paths.fd_benchmark_dir, root=ROOT)
    assert path.is_dir(), f"expected fd-benchmark at {path}"
    assert path.name == "fd-benchmark"


def test_migrations_apply(tmp_path: Path):
    db_file = database_path(tmp_path)
    conn = connect(db_file)
    try:
        newly = apply_migrations(conn)
        assert "001_initial.sql" in newly
        assert "002_phase4_tools.sql" in newly
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        for required in (
            "sessions",
            "notes",
            "timers",
            "jobs",
            "artifacts",
            "owner_profile",
            "notifications",
            "tool_idempotency",
            "schema_migrations",
        ):
            assert required in tables
        # idempotent
        assert apply_migrations(conn) == []
    finally:
        conn.close()


def test_mcp_tool_handles():
    meta = RequestMeta(request_id="req-test", idempotency_key="smoke-timer")
    assert asr_handle("session.start", meta=meta).ok is True
    assert tts_handle("speech.start", meta=meta).ok is True
    # Phase 4–5: timer + web tools live (web may be mock without TAVILY_API_KEY)
    created = tasks_handle("timer.create", meta=meta, duration_seconds=5, label="smoke")
    assert created.ok is True
    assert created.data["timer"]["label"] == "smoke"
    web = tasks_handle("web.search", meta=RequestMeta(idempotency_key="smoke-web"), query="ME3 smoke")
    assert web.ok is True
    assert web.data.get("count", 0) >= 1
    assert web.data.get("evidence")
    # Phase 6: delegation live (mock worker if OpenCode down)
    job = delegation_handle(
        "jobs.submit",
        meta=RequestMeta(idempotency_key="smoke-job"),
        specification="smoke workspace",
    )
    assert job.ok is True
    assert job.data.get("job", {}).get("id")


def test_app_main_exits_zero():
    from me3.app.main import main

    assert main([]) == 0
