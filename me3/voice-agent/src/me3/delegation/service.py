"""Delegation service: one-job queue over MockWorker or OpenCode HTTP."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Callable

from me3.config import load_config, load_secrets, resolve_path
from me3.delegation.mock_worker import MockWorker
from me3.delegation.opencode import OpenCodeClient
from me3.delegation.registry import JobRegistry
from me3.mcp.common import ErrorKind, RequestMeta, ToolError, ToolResult

logger = logging.getLogger("me3.delegation.service")

EventCallback = Callable[[dict[str, Any]], None]


class DelegationService:
    def __init__(
        self,
        db_path: Path | None = None,
        *,
        workspaces_dir: Path | None = None,
        on_event: EventCallback | None = None,
        force_mock: bool | None = None,
        mock_ask_input: bool = False,
        mock_delay_s: float = 0.15,
    ):
        cfg = load_config()
        secrets = load_secrets()
        self.registry = JobRegistry(db_path)
        self.workspaces_dir = workspaces_dir or resolve_path(cfg.paths.workspaces_dir)
        self.workspaces_dir.mkdir(parents=True, exist_ok=True)
        self.on_event = on_event or (lambda _e: None)
        self._lock = threading.RLock()
        self._poll_stop = threading.Event()
        self._poll_thread: threading.Thread | None = None

        self.opencode_url = (getattr(secrets, "opencode_url", None) or "").strip() or "http://127.0.0.1:4096"
        self.opencode = OpenCodeClient(
            self.opencode_url,
            username=getattr(secrets, "opencode_server_username", None) or "opencode",
            password=getattr(secrets, "opencode_server_password", None) or "",
        )
        health = self.opencode.health()
        use_mock = force_mock if force_mock is not None else not health.get("healthy")
        self.provider = "mock" if use_mock else "opencode"
        self.mock = MockWorker(
            self.registry,
            workspaces_dir=self.workspaces_dir,
            on_progress=self._emit,
            delay_s=mock_delay_s,
            ask_input=mock_ask_input,
        )
        self._session_by_job: dict[str, str] = {}
        if self.provider == "opencode":
            self._start_poller()
        logger.info("delegation provider=%s health=%s", self.provider, health)

    def _emit(self, event: dict[str, Any]) -> None:
        try:
            self.on_event(event)
        except Exception:
            logger.exception("delegation on_event failed")

    def close(self) -> None:
        self._poll_stop.set()
        if self._poll_thread and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=2.0)
        self.registry.close()

    def health(self) -> dict[str, Any]:
        if self.provider == "opencode":
            h = self.opencode.health()
            h["provider"] = "opencode"
            return h
        return self.mock.health()

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
            kind = ErrorKind.NEEDS_APPROVAL if "one active job" in str(exc) else ErrorKind.FAILED
            if "not waiting" in str(exc).lower():
                kind = ErrorKind.NEEDS_INPUT
            return ToolResult(
                ok=False,
                request_id=meta.request_id,
                error=ToolError(kind=kind, message=str(exc)),
            )
        except Exception as exc:
            logger.exception("delegation %s failed", tool)
            return ToolResult(
                ok=False,
                request_id=meta.request_id,
                error=ToolError(kind=ErrorKind.FAILED, message=str(exc)),
            )
        return ToolResult(ok=True, request_id=meta.request_id, data=data)

    def _dispatch(self, tool: str, *, meta: RequestMeta, **kwargs: Any) -> dict[str, Any]:
        if tool == "jobs.submit":
            return self.submit(
                specification=kwargs.get("specification") or kwargs.get("task") or kwargs.get("prompt") or "",
                instructions=kwargs.get("instructions"),
                idempotency_key=meta.idempotency_key,
                model=kwargs.get("model"),
            )
        if tool == "jobs.status":
            job_id = kwargs.get("job_id") or kwargs.get("id")
            if job_id:
                job = self.registry.get(job_id)
                if not job:
                    raise RuntimeError(f"job not found: {job_id}")
                return {"job": job, "provider": self.provider}
            active = self.registry.active_job()
            return {"job": active, "jobs": self.registry.list_jobs(limit=10), "provider": self.provider}
        if tool == "jobs.update":
            job_id = kwargs.get("job_id") or kwargs.get("id")
            if not job_id:
                raise ValueError("job_id is required")
            response = kwargs.get("response") or kwargs.get("text") or kwargs.get("answer") or ""
            approve = bool(kwargs.get("approve", True))
            if kwargs.get("status") == "needs_approval" or kwargs.get("request_approval"):
                job = self.registry.update(
                    job_id,
                    status="needs_approval",
                    pending_user_request=kwargs.get("pending_user_request") or "Please approve to continue.",
                )
                self._emit({"type": "job_update", "job": job})
                return {"job": job}
            return {"job": self.respond(job_id, response=response, approve=approve)}
        if tool == "jobs.cancel":
            job_id = kwargs.get("job_id") or kwargs.get("id")
            if not job_id:
                raise ValueError("job_id is required")
            return {"cancelled": self.cancel(job_id), "job": self.registry.get(job_id)}
        if tool == "jobs.result":
            job_id = kwargs.get("job_id") or kwargs.get("id")
            if not job_id:
                raise ValueError("job_id is required")
            job = self.registry.get(job_id)
            if not job:
                raise RuntimeError(f"job not found: {job_id}")
            return {
                "job": job,
                "revisions": self.registry.revisions(job_id),
                "provider": self.provider,
            }
        raise ValueError(f"unsupported tool: {tool}")

    def submit(
        self,
        *,
        specification: str,
        instructions: str | None = None,
        idempotency_key: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        spec = (specification or "").strip()
        if not spec:
            raise ValueError("specification is required")
        if idempotency_key:
            existing = self.registry.get_by_idempotency(idempotency_key)
            if existing:
                return {"job": existing, "idempotent_replay": True, "provider": self.provider}

        job = self.registry.create(
            specification=spec,
            instructions=instructions,
            idempotency_key=idempotency_key,
            model=model,
            harness=self.provider,
            workspace_path=str(self.workspaces_dir),
        )
        self._emit({"type": "job_update", "job": job})
        if self.provider == "mock":
            self.mock.start(job["id"])
        else:
            self._start_opencode_job(job)
        return {"job": self.registry.get(job["id"]), "provider": self.provider}

    def respond(self, job_id: str, *, response: str, approve: bool = True) -> dict[str, Any]:
        if self.provider == "mock":
            job = self.mock.respond(job_id, response=response or ("approved" if approve else "clarified"), approve=approve)
            if not job:
                raise RuntimeError(f"job not found: {job_id}")
            return job
        job = self.registry.get(job_id)
        if not job:
            raise RuntimeError(f"job not found: {job_id}")
        if job["status"] not in ("needs_input", "needs_approval"):
            raise RuntimeError(f"job {job_id} is not waiting for input")
        session_id = self._session_by_job.get(job_id)
        # Prefer answering via a follow-up async prompt; permission API needs permissionID.
        if session_id and response:
            try:
                self.opencode.send_message_async(session_id, response)
            except Exception:
                logger.exception("opencode follow-up failed")
        self.registry.update(job_id, status="running", clear_pending=True)
        self.registry.add_revision(job_id, f"user response: {(response or '')[:200]}")
        job = self.registry.get(job_id)
        self._emit({"type": "job_update", "job": job})
        return job  # type: ignore[return-value]

    def cancel(self, job_id: str) -> bool:
        if self.provider == "mock":
            return self.mock.cancel(job_id)
        session_id = self._session_by_job.get(job_id)
        if session_id:
            try:
                self.opencode.abort(session_id)
            except Exception:
                logger.exception("opencode abort failed")
        ok = self.registry.update(job_id, status="cancelled", outcome="cancelled", clear_pending=True) is not None
        self.registry.add_revision(job_id, "cancelled")
        self._emit({"type": "job_update", "job": self.registry.get(job_id)})
        return ok

    def _start_opencode_job(self, job: dict[str, Any]) -> None:
        try:
            session = self.opencode.create_session(title=f"ME3 {job['id'][:8]}")
            session_id = session.get("id") or session.get("sessionID") or ""
            if not session_id:
                raise RuntimeError(f"OpenCode session create returned no id: {session}")
            self._session_by_job[job["id"]] = session_id
            ws = self.workspaces_dir / job["id"]
            ws.mkdir(parents=True, exist_ok=True)
            prompt = (
                f"ME3 delegated job {job['id']}.\n"
                f"Specification:\n{job['specification']}\n"
                f"Instructions:\n{job.get('instructions') or 'Complete the request in the workspace.'}\n"
                f"Workspace hint: {ws}\n"
                "Return a short outcome summary when done. Do not publish privately."
            )
            self.opencode.send_message_async(session_id, prompt)
            self.registry.update(
                job["id"],
                status="running",
                workspace_path=str(ws),
                attempts_inc=True,
                harness="opencode",
            )
            self.registry.add_revision(job["id"], f"opencode session {session_id}")
            self._emit({"type": "job_update", "job": self.registry.get(job["id"])})
        except Exception as exc:
            logger.exception("opencode start failed")
            self.registry.update(job["id"], status="failed", outcome=str(exc)[:300])
            self.registry.add_revision(job["id"], f"failed: {exc}")
            self._emit({"type": "job_update", "job": self.registry.get(job["id"])})

    def _start_poller(self) -> None:
        if self._poll_thread and self._poll_thread.is_alive():
            return
        self._poll_stop.clear()
        self._poll_thread = threading.Thread(target=self._poll_loop, name="opencode-poll", daemon=True)
        self._poll_thread.start()

    def _poll_loop(self) -> None:
        """Rate-limited progress poll — must not block audio."""
        while not self._poll_stop.wait(2.0):
            try:
                active = self.registry.active_job()
                if not active or active["harness"] != "opencode":
                    continue
                sid = self._session_by_job.get(active["id"])
                if not sid:
                    continue
                msgs = self.opencode.list_messages(sid, limit=5)
                if msgs:
                    self.registry.add_revision(active["id"], f"opencode messages={len(msgs)}")
                    # Heuristic completion: last assistant text contains DONE/complete
                    text = json_dump_messages(msgs)
                    if "ME3_JOB_DONE" in text or active["status"] == "running" and "completed successfully" in text.lower():
                        self.registry.update(
                            active["id"],
                            status="completed",
                            outcome=text[-500:],
                            clear_pending=True,
                        )
                        self._emit({"type": "job_update", "job": self.registry.get(active["id"])})
            except Exception:
                logger.exception("opencode poll failed")


def json_dump_messages(msgs: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for m in msgs:
        info = m.get("info") or m
        for p in m.get("parts") or []:
            if isinstance(p, dict) and p.get("text"):
                parts.append(str(p["text"]))
            elif isinstance(p, dict) and p.get("type") == "text":
                parts.append(str(p.get("text") or ""))
        if info.get("content"):
            parts.append(str(info["content"]))
    return "\n".join(parts)
