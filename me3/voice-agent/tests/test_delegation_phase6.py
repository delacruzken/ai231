"""Phase 6: one-job queue, needs_input, cancel, cleanup."""

from __future__ import annotations

import time
from pathlib import Path

from me3.delegation.service import DelegationService
from me3.mcp.common import RequestMeta
from me3.mcp.delegation import server as del_mcp


def _wait_status(svc: DelegationService, job_id: str, want: set[str], timeout: float = 3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = svc.registry.get(job_id)
        if job and job["status"] in want:
            return job
        time.sleep(0.05)
    return svc.registry.get(job_id)


def test_one_job_queue_and_complete(tmp_path: Path):
    svc = DelegationService(
        db_path=tmp_path / "j.db",
        workspaces_dir=tmp_path / "ws",
        force_mock=True,
        mock_delay_s=0.05,
    )
    r = svc.handle("jobs.submit", RequestMeta(idempotency_key="j1"), specification="Build a hello page")
    assert r.ok
    jid = r.data["job"]["id"]
    # second submit rejected while active
    busy = svc.handle("jobs.submit", specification="another")
    assert not busy.ok
    done = _wait_status(svc, jid, {"completed"})
    assert done and done["status"] == "completed"
    # after complete, new job ok
    r2 = svc.handle("jobs.submit", RequestMeta(idempotency_key="j2"), specification="second")
    assert r2.ok
    _wait_status(svc, r2.data["job"]["id"], {"completed"})
    svc.close()


def test_needs_input_then_respond(tmp_path: Path):
    svc = DelegationService(
        db_path=tmp_path / "n.db",
        workspaces_dir=tmp_path / "ws",
        force_mock=True,
        mock_delay_s=0.05,
        mock_ask_input=True,
    )
    r = svc.handle("jobs.submit", specification="Build site please clarify theme")
    jid = r.data["job"]["id"]
    waiting = _wait_status(svc, jid, {"needs_input"})
    assert waiting and waiting["status"] == "needs_input"
    upd = svc.handle("jobs.update", job_id=jid, response="Use plain HTML", approve=True)
    assert upd.ok
    done = _wait_status(svc, jid, {"completed"})
    assert done and done["status"] == "completed"
    svc.close()


def test_cancel_and_cleanup(tmp_path: Path):
    svc = DelegationService(
        db_path=tmp_path / "c.db",
        workspaces_dir=tmp_path / "ws",
        force_mock=True,
        mock_delay_s=0.2,
        mock_ask_input=True,
    )
    r = svc.handle("jobs.submit", specification="slow clarify job")
    jid = r.data["job"]["id"]
    _wait_status(svc, jid, {"needs_input", "running", "queued"})
    cancel = svc.handle("jobs.cancel", job_id=jid)
    assert cancel.ok
    job = svc.registry.get(jid)
    assert job and job["status"] == "cancelled"
    dropped = svc.registry.cleanup_terminal(keep=0)
    assert dropped >= 1
    assert svc.registry.get(jid) is None
    svc.close()


def test_idempotent_submit(tmp_path: Path):
    svc = DelegationService(
        db_path=tmp_path / "i.db",
        workspaces_dir=tmp_path / "ws",
        force_mock=True,
        mock_delay_s=0.05,
    )
    a = svc.handle("jobs.submit", RequestMeta(idempotency_key="same"), specification="x")
    b = svc.handle("jobs.submit", RequestMeta(idempotency_key="same"), specification="x")
    assert a.data["job"]["id"] == b.data["job"]["id"]
    _wait_status(svc, a.data["job"]["id"], {"completed"})
    svc.close()


def test_mcp_delegation_handle(tmp_path: Path, monkeypatch):
    svc = DelegationService(
        db_path=tmp_path / "m.db",
        workspaces_dir=tmp_path / "ws",
        force_mock=True,
        mock_delay_s=0.05,
    )
    monkeypatch.setattr(del_mcp, "_service", svc)
    r = del_mcp.handle("jobs.submit", RequestMeta(idempotency_key="mcp"), specification="mcp job")
    assert r.ok
    _wait_status(svc, r.data["job"]["id"], {"completed"})
    svc.close()
