"""In-process worker for structural tests when OpenCode is unreachable."""

from __future__ import annotations

import html
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable

from me3.delegation.registry import JobRegistry

ProgressCallback = Callable[[dict[str, Any]], None]


def _title_from_spec(spec: str) -> str:
    s = (spec or "").strip()
    m = re.search(
        r"(?:landing\s+page|website|page|site)\s+(?:about|for|on)\s+(.+?)(?:\.|$)",
        s,
        re.I,
    )
    if m:
        title = m.group(1).strip(" .")
        # Cut at clause boundaries for a readable H1.
        title = re.split(r",\s*(?:the|who|which|an?)\b", title, maxsplit=1)[0].strip()
        if 3 <= len(title) <= 80:
            return title
    # Fallback: first ~10 words.
    words = s.split()
    return " ".join(words[:10]).strip(" .,") or "Mock site"


def write_mock_site(ws: Path, specification: str) -> list[dict[str, str]]:
    """Write a publishable placeholder landing page (index.html + styles)."""
    ws.mkdir(parents=True, exist_ok=True)
    title = _title_from_spec(specification)
    safe_title = html.escape(title)
    safe_spec = html.escape((specification or "").strip()[:1200])
    (ws / "README.md").write_text(
        f"# {title}\n\nMock worker placeholder (OpenCode was offline).\n\nSpec:\n\n{specification}\n",
        encoding="utf-8",
    )
    (ws / "styles.css").write_text(
        """
:root { --bg:#0f1419; --ink:#e8eef7; --muted:#8b9bb4; --accent:#3d9cf0; --font: Georgia, serif; }
* { box-sizing: border-box; }
body { margin:0; font-family: var(--font); background: radial-gradient(900px 500px at 10% -10%, #1e3a5f, var(--bg)); color: var(--ink); }
main { max-width: 42rem; margin: 0 auto; padding: 3rem 1.25rem 4rem; }
.eyebrow { text-transform: uppercase; letter-spacing: .12em; color: var(--muted); font-size: .75rem; }
h1 { font-size: clamp(2rem, 5vw, 3rem); line-height: 1.15; margin: .4rem 0 1rem; }
.lede { font-size: 1.15rem; color: #c5d0e0; line-height: 1.5; }
.card { margin-top: 2rem; padding: 1.25rem 1.35rem; border: 1px solid #2c3a4f; border-radius: .75rem; background: rgba(26,35,50,.7); }
.card h2 { margin: 0 0 .6rem; font-size: 1rem; color: var(--accent); }
.card p { margin: 0; white-space: pre-wrap; color: var(--muted); font-size: .95rem; line-height: 1.45; }
footer { margin-top: 2.5rem; color: var(--muted); font-size: .85rem; }
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (ws / "index.html").write_text(
        f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{safe_title}</title>
  <link rel="stylesheet" href="styles.css" />
</head>
<body>
  <main>
    <p class="eyebrow">ME3 mock site</p>
    <h1>{safe_title}</h1>
    <p class="lede">Placeholder landing page generated because OpenCode was offline. Restart <code>opencode serve</code> for a fuller build.</p>
    <section class="card">
      <h2>Job specification</h2>
      <p>{safe_spec}</p>
    </section>
    <footer>Built by ME3 MockWorker · publishable to GitHub Pages</footer>
  </main>
</body>
</html>
""",
        encoding="utf-8",
    )
    return [
        {"kind": "file", "path": str(ws / "index.html")},
        {"kind": "file", "path": str(ws / "styles.css")},
        {"kind": "file", "path": str(ws / "README.md")},
    ]


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
            write_mock_site(ws, str(job.get("specification") or ""))
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
            ws = Path(str(job.get("workspace_path") or (self.workspaces_dir / job_id)))
            artifacts = write_mock_site(ws, str(job.get("specification") or ""))
            self.registry.update(
                job_id,
                status="completed",
                outcome=(
                    f"mock completed ({note}) — placeholder index.html "
                    "(start opencode serve for a full build)"
                ),
                artifacts=artifacts,
                checks=[
                    {"name": "mock_index_html", "ok": (ws / "index.html").is_file()},
                    {"name": "mock_readme", "ok": (ws / "README.md").is_file()},
                ],
                clear_pending=True,
            )
            self.registry.add_revision(job_id, "completed")
            self.on_progress({"type": "job_update", "job": self.registry.get(job_id)})
        except Exception:
            return
