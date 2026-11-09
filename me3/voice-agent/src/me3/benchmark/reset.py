"""Explicit trial reset — silence alone is not a reset (architecture §11)."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("me3.benchmark.reset")


def reset_controller_for_trial(ctrl: Any) -> dict[str, Any]:
    """Clear conversational + tool state between fdbench trials.

    Cancels speech/LLM, turns session OFF then ON, cancels active jobs,
    cancels active timers, and clears the UI transcript buffer. Does not
    wipe the owner profile or publication history.
    """
    summary: dict[str, Any] = {"ok": True}

    try:
        if getattr(ctrl, "conversation", None):
            ctrl.conversation.cancel_active("trial_reset")
            ctrl.conversation.reset_listening()
            try:
                ctrl.conversation.memory.close_session()
            except Exception:
                logger.debug("memory close_session failed", exc_info=True)
    except Exception as exc:
        summary["conversation_error"] = str(exc)

    try:
        ctrl._cancel_speech()
    except Exception:
        pass

    # Jobs
    cancelled_jobs = 0
    try:
        active = ctrl.delegation.registry.active_job()
        if active:
            ctrl.delegation.cancel(active["id"])
            cancelled_jobs = 1
        dropped = ctrl.delegation.registry.cleanup_terminal(keep=0)
        summary["jobs_cancelled"] = cancelled_jobs
        summary["jobs_cleaned"] = dropped
    except Exception as exc:
        summary["jobs_error"] = str(exc)

    # Timers
    cancelled_timers = 0
    try:
        for t in list(ctrl.tasks.store.list_timers(status="active") or []):
            ctrl.tasks.handle("timer.cancel", timer_id=t["id"])
            cancelled_timers += 1
        summary["timers_cancelled"] = cancelled_timers
    except Exception as exc:
        summary["timers_error"] = str(exc)

    # Transcript + OFF/ON flush
    try:
        with ctrl._lock:
            ctrl.transcript.clear()
            ctrl.last_error = None
        if ctrl.enabled:
            ctrl.set_enabled(False)
        ctrl.set_enabled(True)
        summary["session"] = "on"
    except Exception as exc:
        summary["session_error"] = str(exc)
        summary["ok"] = False

    logger.info("trial reset: %s", summary)
    return summary
