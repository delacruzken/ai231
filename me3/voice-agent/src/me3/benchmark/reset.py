"""Explicit trial reset — silence alone is not a reset (architecture §11)."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

logger = logging.getLogger("me3.benchmark.reset")


def reset_controller_for_trial(ctrl: Any) -> dict[str, Any]:
    """Clear conversational + audio + tool state between fdbench trials.

    Cancels speech/LLM, drains capture, resets ASR/VAD/turn-taker, clears echo
    and barge trackers, turns session OFF then ON, cancels active jobs/timers,
    and clears the UI transcript buffer. Does not wipe owner profile / publications.
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

    try:
        if getattr(ctrl, "capture", None):
            ctrl.capture.drain()
    except Exception:
        pass

    lock = getattr(ctrl, "_announce_lock", None)
    if lock is not None:
        try:
            with lock:
                if hasattr(ctrl, "_pending_announcements"):
                    ctrl._pending_announcements.clear()
                if hasattr(ctrl, "_announce_flushing"):
                    ctrl._announce_flushing = False
        except Exception:
            pass

    try:
        if hasattr(ctrl, "_reset_barge_tracker"):
            ctrl._reset_barge_tracker()
        ctrl._echo_gate_until = 0.0
        ctrl._was_gating_asr = False
        ctrl._vad_buf = np.zeros(0, dtype=np.float32)
        ctrl._vad_in_speech = False
    except Exception:
        pass

    try:
        if getattr(ctrl, "asr", None):
            try:
                ctrl.asr.force_finalize(emit=False)
            except Exception:
                pass
            ctrl.asr.reset_segment()
    except Exception as exc:
        summary["asr_error"] = str(exc)

    try:
        if getattr(ctrl, "vad", None):
            ctrl.vad.reset()
    except Exception:
        pass

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

    cancelled_timers = 0
    try:
        for t in list(ctrl.tasks.store.list_timers(status="active") or []):
            ctrl.tasks.handle("timer.cancel", timer_id=t["id"])
            cancelled_timers += 1
        summary["timers_cancelled"] = cancelled_timers
    except Exception as exc:
        summary["timers_error"] = str(exc)

    try:
        with ctrl._lock:
            ctrl.transcript.clear()
            ctrl.last_error = None
        if ctrl.enabled:
            ctrl.set_enabled(False)
        time.sleep(0.35)
        ctrl.set_enabled(True)
        if getattr(ctrl, "asr", None):
            try:
                ctrl.asr.reset_segment()
            except Exception:
                pass
        if getattr(ctrl, "capture", None):
            try:
                ctrl.capture.drain()
            except Exception:
                pass
        summary["session"] = "on"
    except Exception as exc:
        summary["session_error"] = str(exc)
        summary["ok"] = False

    logger.info("trial reset: %s", summary)
    return summary
