"""Phase 4 sample bank: idempotency, fire, restart reconcile, mock tool turn."""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from me3.config import ROOT, load_config
from me3.conversation.llm import MockLLM
from me3.mcp.common import RequestMeta
from me3.tasks.service import TaskService
from me3.tasks.tools import dispatch_tool


def run(out_dir: Path | None = None) -> dict:
    cfg = load_config()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = out_dir or (ROOT / "artifacts" / "tools_phase4" / stamp)
    out.mkdir(parents=True, exist_ok=True)
    db = out / "sample.db"
    svc = TaskService(db_path=db, timezone_name=cfg.session.timezone)
    cases: list[dict] = []

    # Idempotent timer create
    r1 = svc.handle("timer.create", RequestMeta(idempotency_key="sb-timer"), duration_seconds=30, label="sample")
    r2 = svc.handle("timer.create", RequestMeta(idempotency_key="sb-timer"), duration_seconds=30, label="sample")
    cases.append(
        {
            "case": "idempotent_timer_create",
            "ok": r1.ok and r2.ok and r1.data["timer"]["id"] == r2.data["timer"]["id"],
            "timer_id": r1.data["timer"]["id"] if r1.ok else None,
        }
    )

    # Notes idempotency + search
    n1 = svc.handle("notes.add", RequestMeta(idempotency_key="sb-note"), body="phase4 milk")
    n2 = svc.handle("notes.add", RequestMeta(idempotency_key="sb-note"), body="phase4 milk")
    search = svc.handle("notes.search", query="milk")
    cases.append(
        {
            "case": "idempotent_note_add",
            "ok": n1.ok
            and n2.ok
            and n1.data["note"]["id"] == n2.data["note"]["id"]
            and len(search.data.get("notes") or []) == 1,
        }
    )

    # Restart reconcile: past deadline fires once
    past = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    t = svc.store.create_timer(label="missed", deadline_at=past, idempotency_key="sb-missed")
    fired = svc.fire_due_timers()
    fired2 = svc.fire_due_timers()
    pending = svc.drain_pending_notifications()
    cases.append(
        {
            "case": "restart_reconcile_fire_once",
            "ok": len(fired) == 1
            and fired[0]["timer"]["id"] == t["id"]
            and fired2 == []
            and len(pending) >= 1,
        }
    )

    # Short live timeout
    live = svc.handle("timer.create", duration_seconds=1, label="live")
    deadline = time.time() + 3.0
    saw = []
    while time.time() < deadline:
        saw = svc.fire_due_timers()
        if saw:
            break
        time.sleep(0.1)
    cases.append({"case": "timeout_fire", "ok": bool(saw) and live.ok})

    # Mock LLM tool path
    llm = MockLLM()

    def executor(name, args):
        return dispatch_tool(svc, name, args, turn_id="sb-turn")

    chunks = list(
        llm.stream(
            [{"role": "user", "content": "Remember that the gate code is 42"}],
            system="test",
            turn_id="sb-turn",
            tools_enabled=True,
            tool_executor=executor,
        )
    )
    notes = svc.store.search_notes("gate")
    cases.append(
        {
            "case": "mock_llm_note_tool",
            "ok": any(c.tool_results for c in chunks) and len(notes) >= 1,
            "reply": "".join(c.text for c in chunks if c.text)[:120],
        }
    )

    # Cancel remaining active sample timer
    svc.handle("timer.cancel", timer_id=r1.data["timer"]["id"])

    report = {
        "phase": 4,
        "structural_ok": all(c["ok"] for c in cases),
        "cases": cases,
        "db": str(db),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    svc.close()
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    run()


if __name__ == "__main__":
    main()
