"""High-level timer/note operations used by MCP and conversation tools."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from me3.mcp.common import ErrorKind, RequestMeta, ToolError, ToolResult
from me3.retrieval.service import RetrievalService, make_retrieval
from me3.tasks.store import TaskStore

logger = logging.getLogger("me3.tasks.service")

DURATION_RE = re.compile(
    r"(?:(\d+)\s*h(?:ours?)?)?\s*(?:(\d+)\s*m(?:in(?:utes?)?)?)?\s*(?:(\d+)\s*s(?:ec(?:onds?)?)?)?",
    re.I,
)


def parse_duration_seconds(text: str | int | float | None) -> int | None:
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return max(0, int(text))
    s = str(text).strip().lower()
    if s.isdigit():
        return int(s)
    m = DURATION_RE.fullmatch(s.replace(",", " ").strip())
    if not m:
        # try "5 minutes" style already covered; also "for 5 minutes"
        m2 = re.search(
            r"(\d+)\s*(hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s)\b",
            s,
        )
        if not m2:
            return None
        n = int(m2.group(1))
        unit = m2.group(2)
        if unit.startswith("h"):
            return n * 3600
        if unit.startswith("m"):
            return n * 60
        return n
    h = int(m.group(1) or 0)
    mi = int(m.group(2) or 0)
    sec = int(m.group(3) or 0)
    total = h * 3600 + mi * 60 + sec
    return total if total > 0 else None


class TaskService:
    def __init__(
        self,
        db_path: Path | None = None,
        *,
        timezone_name: str = "Asia/Manila",
        retrieval: RetrievalService | None = None,
        force_mock_retrieval: bool = False,
    ):
        self.store = TaskStore(db_path)
        self.timezone_name = timezone_name
        if retrieval is not None:
            self.retrieval = retrieval
        else:
            backend, provider = make_retrieval(force_mock=force_mock_retrieval)
            self.retrieval = RetrievalService(backend, provider=provider)
        self.retrieval_provider = self.retrieval.provider

    def close(self) -> None:
        self.store.close()

    def _tz(self) -> ZoneInfo | timezone:
        try:
            return ZoneInfo(self.timezone_name)
        except Exception:
            return timezone.utc

    def _now(self) -> datetime:
        return datetime.now(self._tz())

    def handle(self, tool: str, meta: RequestMeta | None = None, **kwargs: Any) -> ToolResult:
        meta = meta or RequestMeta()
        cached = self.store.get_idempotent(meta.idempotency_key)
        if cached is not None:
            return ToolResult(ok=True, request_id=meta.request_id, data={**cached, "idempotent_replay": True})

        try:
            data = self._dispatch(tool, meta=meta, **kwargs)
        except ValueError as exc:
            return ToolResult(
                ok=False,
                request_id=meta.request_id,
                error=ToolError(kind=ErrorKind.NEEDS_INPUT, message=str(exc)),
            )
        except Exception as exc:
            logger.exception("tool %s failed", tool)
            return ToolResult(
                ok=False,
                request_id=meta.request_id,
                error=ToolError(kind=ErrorKind.FAILED, message=str(exc)),
            )

        result = ToolResult(ok=True, request_id=meta.request_id, data=data)
        if meta.idempotency_key and result.ok:
            self.store.put_idempotent(meta.idempotency_key, tool, data)
        return result

    def _dispatch(self, tool: str, *, meta: RequestMeta, **kwargs: Any) -> dict[str, Any]:
        if tool == "timer.create":
            return self.create_timer(
                label=kwargs.get("label"),
                duration_seconds=kwargs.get("duration_seconds"),
                duration=kwargs.get("duration"),
                deadline_at=kwargs.get("deadline_at"),
                idempotency_key=meta.idempotency_key,
            )
        if tool == "timer.list":
            return {"timers": self.store.list_timers(status=kwargs.get("status", "active"))}
        if tool == "timer.update":
            timer_id = kwargs.get("timer_id") or kwargs.get("id")
            if not timer_id:
                raise ValueError("timer_id is required")
            updated = self.store.update_timer(
                timer_id,
                label=kwargs.get("label"),
                deadline_at=kwargs.get("deadline_at"),
                status=kwargs.get("status"),
            )
            if not updated:
                raise ValueError(f"timer not found: {timer_id}")
            return {"timer": updated}
        if tool == "timer.cancel":
            timer_id = kwargs.get("timer_id") or kwargs.get("id")
            if not timer_id:
                raise ValueError("timer_id is required")
            cancelled = self.store.cancel_timer(timer_id)
            if not cancelled:
                raise ValueError(f"timer not found: {timer_id}")
            return {"timer": cancelled}
        if tool == "notes.add":
            body = (kwargs.get("body") or kwargs.get("text") or "").strip()
            if not body:
                raise ValueError("note body is required")
            note = self.store.add_note(
                body=body,
                title=kwargs.get("title"),
                tags=kwargs.get("tags"),
                idempotency_key=meta.idempotency_key,
            )
            return {"note": note}
        if tool == "notes.search":
            return {"notes": self.store.search_notes(kwargs.get("query") or "", limit=int(kwargs.get("limit") or 20))}
        if tool == "notes.edit":
            note_id = kwargs.get("note_id") or kwargs.get("id")
            if not note_id:
                raise ValueError("note_id is required")
            edited = self.store.edit_note(
                note_id,
                body=kwargs.get("body") or kwargs.get("text"),
                title=kwargs.get("title"),
                tags=kwargs.get("tags"),
            )
            if not edited:
                raise ValueError(f"note not found: {note_id}")
            return {"note": edited}
        if tool == "notes.delete":
            note_id = kwargs.get("note_id") or kwargs.get("id")
            if not note_id:
                raise ValueError("note_id is required")
            deleted = self.store.delete_note(note_id)
            if not deleted:
                raise ValueError(f"note not found: {note_id}")
            return {"note": deleted}
        if tool == "web.search":
            return self.retrieval.search(
                kwargs.get("query") or kwargs.get("q") or "",
                max_results=int(kwargs.get("max_results") or 5),
                search_depth=str(kwargs.get("search_depth") or "basic"),
                topic=kwargs.get("topic"),
            )
        if tool == "web.fetch":
            return self.retrieval.fetch(
                url=kwargs.get("url"),
                urls=kwargs.get("urls"),
            )
        if tool == "web.crawl":
            return self.retrieval.crawl(
                kwargs.get("url") or "",
                max_depth=int(kwargs.get("max_depth") or 1),
                limit=int(kwargs.get("limit") or 10),
                instructions=kwargs.get("instructions"),
            )
        raise ValueError(f"unsupported tool: {tool}")

    def create_timer(
        self,
        *,
        label: str | None = None,
        duration_seconds: int | float | None = None,
        duration: str | None = None,
        deadline_at: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        secs = parse_duration_seconds(duration_seconds)
        if secs is None and duration:
            secs = parse_duration_seconds(duration)
        now = self._now()
        if deadline_at:
            deadline = datetime.fromisoformat(deadline_at.replace("Z", "+00:00"))
            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=self._tz())
        elif secs is not None and secs > 0:
            deadline = now + timedelta(seconds=secs)
        else:
            raise ValueError("Provide duration_seconds/duration or deadline_at")
        meta = {"duration_seconds": secs, "created_local": now.isoformat()}
        timer = self.store.create_timer(
            label=label or "timer",
            deadline_at=deadline.astimezone(timezone.utc).isoformat(),
            idempotency_key=idempotency_key,
            meta=meta,
        )
        return {"timer": timer}

    def fire_due_timers(self) -> list[dict[str, Any]]:
        """Mark due active timers as fired and enqueue notifications. Restart-safe."""
        now_iso = datetime.now(timezone.utc).isoformat()
        due = self.store.due_active_timers(now_iso)
        fired: list[dict[str, Any]] = []
        for t in due:
            updated = self.store.mark_timer_fired(t["id"])
            if not updated or updated["status"] != "fired":
                continue
            label = updated.get("label") or "timer"
            notif = self.store.create_notification(
                kind="timer_fired",
                payload={"message": f"Timer finished: {label}", "timer": updated},
                source_kind="timer",
                source_id=updated["id"],
            )
            fired.append({"timer": updated, "notification": notif})
        return fired

    def drain_pending_notifications(self) -> list[dict[str, Any]]:
        return self.store.pending_notifications()

    def mark_delivered(self, notif_id: str) -> dict[str, Any] | None:
        return self.store.mark_notification_delivered(notif_id)
