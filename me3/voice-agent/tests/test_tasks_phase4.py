"""Phase 4: timer/note idempotency, fire, restart reconcile, OFF queue."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from me3.mcp.common import RequestMeta
from me3.mcp.tasks import server as tasks_mcp
from me3.tasks.scheduler import TimerScheduler
from me3.tasks.service import TaskService, parse_duration_seconds


def test_parse_duration():
    assert parse_duration_seconds(90) == 90
    assert parse_duration_seconds("5 minutes") == 300
    assert parse_duration_seconds("2s") == 2
    assert parse_duration_seconds("1h30m") == 5400


def test_timer_note_crud_and_idempotency(tmp_path: Path):
    db = tmp_path / "t.db"
    svc = TaskService(db_path=db)
    meta = RequestMeta(idempotency_key="create-once")
    r1 = svc.handle("timer.create", meta=meta, duration_seconds=120, label="tea")
    assert r1.ok
    tid = r1.data["timer"]["id"]
    r2 = svc.handle("timer.create", meta=RequestMeta(idempotency_key="create-once"), duration_seconds=120, label="tea")
    assert r2.ok
    assert r2.data.get("idempotent_replay") is True or r2.data["timer"]["id"] == tid
    assert r2.data["timer"]["id"] == tid

    listed = svc.handle("timer.list")
    assert listed.ok and len(listed.data["timers"]) == 1

    n1 = svc.handle("notes.add", meta=RequestMeta(idempotency_key="note-1"), body="buy milk")
    n2 = svc.handle("notes.add", meta=RequestMeta(idempotency_key="note-1"), body="buy milk")
    assert n1.data["note"]["id"] == n2.data["note"]["id"]
    found = svc.handle("notes.search", query="milk")
    assert found.ok and len(found.data["notes"]) == 1

    cancelled = svc.handle("timer.cancel", timer_id=tid)
    assert cancelled.ok and cancelled.data["timer"]["status"] == "cancelled"
    svc.close()


def test_timer_fire_and_restart_reconcile(tmp_path: Path):
    db = tmp_path / "fire.db"
    svc = TaskService(db_path=db)
    past = (datetime.now(timezone.utc) - timedelta(seconds=2)).isoformat()
    created = svc.store.create_timer(label="past", deadline_at=past, idempotency_key="past-1")
    fired = svc.fire_due_timers()
    assert len(fired) == 1
    assert fired[0]["timer"]["id"] == created["id"]
    assert fired[0]["timer"]["status"] == "fired"
    pending = svc.drain_pending_notifications()
    assert len(pending) == 1
    assert pending[0]["kind"] == "timer_fired"
    # second fire is no-op
    assert svc.fire_due_timers() == []
    svc.close()


def test_scheduler_fires_short_timer(tmp_path: Path):
    db = tmp_path / "sched.db"
    svc = TaskService(db_path=db)
    seen: list[dict] = []
    sched = TimerScheduler(svc, interval_s=0.1, on_notification=seen.append)
    svc.handle("timer.create", duration_seconds=1, label="quick")
    sched.start()
    deadline = time.time() + 3.0
    while time.time() < deadline and not seen:
        time.sleep(0.1)
    sched.stop()
    assert seen, "expected timer notification"
    assert seen[0]["kind"] == "timer_fired"
    svc.close()


def test_notification_stays_pending_concept(tmp_path: Path):
    """Timers fire into notifications; delivery is session's job while ON."""
    db = tmp_path / "notif.db"
    svc = TaskService(db_path=db)
    past = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    svc.store.create_timer(label="off", deadline_at=past)
    svc.fire_due_timers()
    pending = svc.drain_pending_notifications()
    assert len(pending) == 1
    # simulate OFF: do not mark delivered
    still = svc.drain_pending_notifications()
    assert len(still) == 1
    svc.mark_delivered(still[0]["id"])
    assert svc.drain_pending_notifications() == []
    svc.close()


def test_mcp_tasks_timer_path(tmp_path: Path, monkeypatch):
    db = tmp_path / "mcp.db"
    svc = TaskService(db_path=db)
    monkeypatch.setattr(tasks_mcp, "_service", svc)
    r = tasks_mcp.handle(
        "timer.create",
        RequestMeta(idempotency_key="mcp-1"),
        duration_seconds=10,
        label="mcp",
    )
    assert r.ok
    # Phase 5: web.search is live (mock or Tavily depending on env)
    web = tasks_mcp.handle("web.search", RequestMeta(), query="x")
    assert web.ok and web.data.get("count", 0) >= 1
    svc.close()


def test_mock_llm_timer_tool(tmp_path: Path):
    from me3.conversation.llm import MockLLM
    from me3.tasks.tools import dispatch_tool

    svc = TaskService(db_path=tmp_path / "llm.db")
    llm = MockLLM()

    def executor(name, args):
        return dispatch_tool(svc, name, args, turn_id="t1")

    chunks = list(
        llm.stream(
            [{"role": "user", "content": "Set a timer for 2 seconds"}],
            system="test",
            turn_id="t1",
            tools_enabled=True,
            tool_executor=executor,
        )
    )
    text = "".join(c.text for c in chunks if c.text)
    assert "timer" in text.lower() or "Okay" in text
    assert any(c.tool_results for c in chunks)
    timers = svc.store.list_timers()
    assert len(timers) == 1
    svc.close()
