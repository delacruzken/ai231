"""Phase 6 sample bank: one-job queue, needs_input, cancel, cleanup."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from me3.config import ROOT
from me3.delegation.service import DelegationService
from me3.mcp.common import RequestMeta


def _wait(svc: DelegationService, job_id: str, want: set[str], timeout: float = 4.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = svc.registry.get(job_id)
        if job and job["status"] in want:
            return job
        time.sleep(0.05)
    return svc.registry.get(job_id)


def run(out_dir: Path | None = None) -> dict:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = out_dir or (ROOT / "artifacts" / "delegation_phase6" / stamp)
    out.mkdir(parents=True, exist_ok=True)
    svc = DelegationService(
        db_path=out / "sample.db",
        workspaces_dir=out / "workspaces",
        force_mock=True,
        mock_delay_s=0.08,
    )
    cases: list[dict] = []

    # Complete path
    r = svc.handle(
        "jobs.submit",
        RequestMeta(idempotency_key="p6-1"),
        specification="Build a tiny hello workspace",
    )
    jid = r.data["job"]["id"]
    done = _wait(svc, jid, {"completed"})
    cases.append(
        {
            "case": "submit_complete",
            "ok": r.ok and done is not None and done["status"] == "completed",
            "provider": svc.provider,
            "job_id": jid,
        }
    )

    # One-job busy
    waiting_svc = DelegationService(
        db_path=out / "busy.db",
        workspaces_dir=out / "ws2",
        force_mock=True,
        mock_delay_s=0.05,
        mock_ask_input=True,
    )
    a = waiting_svc.handle("jobs.submit", specification="clarify please")
    busy = waiting_svc.handle("jobs.submit", specification="second")
    cases.append(
        {
            "case": "one_job_rejects_second",
            "ok": a.ok and (not busy.ok),
            "error": busy.error.message if busy.error else None,
        }
    )
    # respond + complete
    j2 = a.data["job"]["id"]
    _wait(waiting_svc, j2, {"needs_input"})
    waiting_svc.handle("jobs.update", job_id=j2, response="plain html", approve=True)
    done2 = _wait(waiting_svc, j2, {"completed"})
    cases.append({"case": "needs_input_resume", "ok": bool(done2 and done2["status"] == "completed")})

    # cancel + cleanup
    csvc = DelegationService(
        db_path=out / "cancel.db",
        workspaces_dir=out / "ws3",
        force_mock=True,
        mock_delay_s=0.05,
        mock_ask_input=True,
    )
    c = csvc.handle("jobs.submit", specification="to cancel")
    cid = c.data["job"]["id"]
    _wait(csvc, cid, {"needs_input", "running"})
    csvc.handle("jobs.cancel", job_id=cid)
    cancelled = csvc.registry.get(cid)
    dropped = csvc.registry.cleanup_terminal(keep=0)
    cases.append(
        {
            "case": "cancel_and_cleanup",
            "ok": cancelled is not None
            and cancelled["status"] == "cancelled"
            and dropped >= 1
            and csvc.registry.get(cid) is None,
        }
    )

    live = svc.opencode.health()
    report = {
        "phase": 6,
        "structural_ok": all(c["ok"] for c in cases),
        "cases": cases,
        "opencode_probe": live,
        "verdict_hint": "improved" if all(c["ok"] for c in cases) else "blocked",
        "note": (
            "No OpenCode cloud API key; local opencode serve optional. "
            "Structural path uses MockWorker. Live OpenCode when /global/health is healthy."
        ),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    waiting_svc.close()
    csvc.close()
    svc.close()
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    run()


if __name__ == "__main__":
    main()
