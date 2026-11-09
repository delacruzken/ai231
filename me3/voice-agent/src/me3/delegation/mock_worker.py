"""In-process worker for structural tests when OpenCode is unreachable."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable

from me3.delegation.registry import JobRegistry

ProgressCallback = Callable[[dict[str, Any]], None]


class MockWorker:
    """Simulates queued → running → optional needs_input → completed."""

    name = "mock"

    def __init__(
        self,
        registry: JobRegistry,
        *,
        workspaces_dir: Path,
        on_progress: ProgressCallback | None = None,
        delay_s: float = 0.15,
        ask_input: bool = False,
    ):
        self.registry = registry
        self.workspaces_dir = workspaces_dir
        self.on_progress = on_progress or (lambda _e: None)
        self.delay_s = delay_s
        self.ask_input = ask_input
        self._threads: dict[str, threading.Thread] = {}
        self._lock = threading.Lock()

    def health(self) -> dict[str, Any]:
        return {"healthy": True, "provider": "mock", "version": "mock-1"}

    def start(self, job_id: str) -> None:
        t = threading.Thread(target=self._run, args=(job_id,), name=f"mock-job-{job_id[:8]}", daemon=True)
        with self._lock:
            self._threads[job_id] = t
        t.start()

    def cancel(self, job_id: str) -> bool:
        job = self.registry.get(job_id)
        if not job:
            return False
        if job["status"] in ("completed", "cancelled", "failed"):
            return True
        self.registry.update(job_id, status="cancelled", outcome="cancelled by user", clear_pending=True)
        self.registry.add_revision(job_id, "cancelled")
        self.on_progress({"type": "job_update", "job": self.registry.get(job_id)})
        return True

    def respond(self, job_id: str, *, response: str, approve: bool = True) -> dict[str, Any] | None:
        job = self.registry.get(job_id)
        if not job:
            return None
        if job["status"] not in ("needs_input", "needs_approval"):
            raise RuntimeError(f"job {job_id} is not waiting for input (status={job['status']})")
        note = f"{'approved' if approve else 'clarified'}: {response[:200]}"
        self.registry.add_revision(job_id, note)
        self.registry.update(job_id, status="running", clear_pending=True)
        self.on_progress({"type": "job_update", "job": self.registry.get(job_id)})
        # finish shortly after
        threading.Thread(target=self._finish, args=(job_id, response), daemon=True).start()
        return self.registry.get(job_id)

    def _run(self, job_id: str) -> None:
        try:
            time.sleep(self.delay_s)
            job = self.registry.get(job_id)
            if not job or job["status"] == "cancelled":
                return
            ws = self.workspaces_dir / job_id
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "README.md").write_text(
                f"# Mock job\n\nSpec: {job['specification']}\n",
                encoding="utf-8",
            )
            self.registry.update(
                job_id,
                status="running",
                workspace_path=str(ws),
                attempts_inc=True,
                harness="mock",
            )
            self.registry.add_revision(job_id, "mock worker started")
            self.on_progress({"type": "job_update", "job": self.registry.get(job_id)})
            time.sleep(self.delay_s)
            job = self.registry.get(job_id)
            if not job or job["status"] == "cancelled":
                return
            if self.ask_input or "clarify" in (job["specification"] or "").lower():
                self.registry.update(
                    job_id,
                    status="needs_input",
                    pending_user_request="Which framework should the mock site use?",
                )
                self.registry.add_revision(job_id, "needs_input")
                self.on_progress({"type": "job_update", "job": self.registry.get(job_id)})
                return
            self._finish(job_id, "ok")
        except Exception:
            # Registry may close while daemon thread still finishing (tests/shutdown).
            return

    def _finish(self, job_id: str, note: str) -> None:
        try:
            time.sleep(self.delay_s)
            job = self.registry.get(job_id)
            if not job or job["status"] == "cancelled":
                return
            artifacts = [{"kind": "file", "path": f"{job.get('workspace_path')}/README.md"}]
            self.registry.update(
                job_id,
                status="completed",
                outcome=f"mock completed ({note})",
                artifacts=artifacts,
                checks=[{"name": "mock_readme", "ok": True}],
                clear_pending=True,
            )
            self.registry.add_revision(job_id, "completed")
            self.on_progress({"type": "job_update", "job": self.registry.get(job_id)})
        except Exception:
            return
