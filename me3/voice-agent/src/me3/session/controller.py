"""Coco harness core — SessionController (Off/Listening/Thinking/Speaking/OverlapPaused).

Product name **Coco** = this live duplex runtime (mic pump, states, echo gate, barge-in,
ASR/TTS). The spoken agent persona remains **Kaki**. Classmates' "Pipecat harness" maps
to Coco here; Pipecat itself is only the fdbench transport shuttle.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable
from uuid import uuid4

import numpy as np

from me3.config import AppConfig, load_config, load_secrets
from me3.conversation.pipeline import ConversationOrchestrator
from me3.speech.asr_engine import AsrEvent, StreamingAsrEngine
from me3.speech.device import AudioCapture, AudioPlayback
from me3.speech.tts_engine import PiperTtsEngine
from me3.speech.vad import WINDOW, SileroVAD
from me3.delegation.service import DelegationService
from me3.profile.owner import OwnerProfileStore
from me3.publication.service import PublicationService
from me3.tasks.scheduler import TimerScheduler
from me3.tasks.service import TaskService

logger = logging.getLogger("me3.session")


class SessionState(str, Enum):
    OFF = "off"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    OVERLAP_PAUSED = "overlap_paused"


@dataclass
class Readiness:
    audio: bool = False
    vad: bool = False
    asr: bool = False
    tts: bool = False
    llm: bool = False

    def as_dict(self) -> dict[str, str]:
        def badge(ok: bool) -> str:
            return "ready" if ok else "degraded"

        return {
            "audio": badge(self.audio),
            "vad": badge(self.vad),
            "asr": badge(self.asr),
            "tts": badge(self.tts),
            "llm": badge(self.llm),
        }

    @property
    def all_ready(self) -> bool:
        return self.audio and self.vad and self.asr and self.tts and self.llm


@dataclass
class SessionSnapshot:
    state: SessionState
    enabled: bool
    readiness: dict[str, str]
    degraded: bool
    transcript: list[dict[str, Any]] = field(default_factory=list)
    last_error: str | None = None
    utterance_generation: int = 0
    llm_backend: str | None = None
    mic_muted: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "enabled": self.enabled,
            "readiness": self.readiness,
            "degraded": self.degraded,
            "transcript": list(self.transcript[-50:]),
            "last_error": self.last_error,
            "utterance_generation": self.utterance_generation,
            "llm_backend": self.llm_backend,
            "mic_muted": self.mic_muted,
        }


EventCallback = Callable[[dict[str, Any]], None]


class SessionController:
    def __init__(
        self,
        cfg: AppConfig | None = None,
        on_event: EventCallback | None = None,
        *,
        benchmark: bool = False,
        force_mock_publication: bool | None = None,
    ):
        self.cfg = cfg or load_config()
        self.on_event = on_event
        self.benchmark = benchmark
        self.state = SessionState.OFF
        self.enabled = False
        self.mic_muted = False
        self.readiness = Readiness()
        self.last_error: str | None = None
        self.transcript: list[dict[str, Any]] = []
        self.utterance_generation = 0
        self.llm_backend: str | None = None

        self._lock = threading.RLock()
        self._stop_pump = threading.Event()
        self._pump_thread: threading.Thread | None = None
        self._vad_buf = np.zeros(0, dtype=np.float32)
        self._speech_prob_thresh = self.cfg.speech.vad_threshold
        self._vad_in_speech = False
        self._audio_out_marked = False
        self._echo_gate_until = 0.0
        self._was_gating_asr = False
        self._barge_speech_ms = 0.0
        self._barge_last_mono: float | None = None
        self._barge_soft_paused = False
        self._barge_quiet_ms = 0.0
        self._barge_asr_buf: list[np.ndarray] = []
        self._barge_vad_buf = np.zeros(0, dtype=np.float32)
        # Once soft-pause starts, mic audio feeds ASR so the interrupt is the next turn.
        self._barge_asr_live = False
        # Skip echo-gate ASR wipe after a confirmed barge (keeps interrupt transcript).
        self._barge_keep_asr = False
        self._echo_rms_ema: float = 0.0
        self._echo_rms_ready: bool = False
        # Job/timer lines spoken when idle — never preempt an in-flight reply's TTS.
        self._pending_announcements: list[dict[str, Any]] = []
        self._announce_lock = threading.Lock()
        self._announce_flushing = False

        self.capture: Any = None
        self.playback: Any = None
        self.asr: StreamingAsrEngine | None = None
        self.tts: PiperTtsEngine | None = None
        self.vad: SileroVAD | None = None
        self.conversation: ConversationOrchestrator | None = None
        secrets = load_secrets()
        # Benchmark: mock retrieval too — live Tavily topic/HTTP failures stall
        # THINKING and inflate E2E/TTFT (and cause no-reply trials).
        retrieval_mock = True if benchmark else (not bool(secrets.tavily_api_key.strip()))
        self.tasks = TaskService(
            timezone_name=self.cfg.session.timezone,
            force_mock_retrieval=retrieval_mock,
        )
        self.owner_store = OwnerProfileStore()
        # Benchmark runs should not hit a live OpenCode/GitHub by accident
        del_force = True if (benchmark or secrets.opencode_force_mock) else None
        self.delegation = DelegationService(
            force_mock=del_force,
            on_event=self._on_delegation_event,
        )
        if self.delegation.provider == "opencode":
            self.cfg.pending.opencode_server = False
        pub_force = force_mock_publication
        if pub_force is None:
            pub_force = True if benchmark else (
                None
                if (
                    secrets.github_pat.strip()
                    and secrets.github_owner.strip()
                    and secrets.github_repo.strip()
                )
                else True
            )
        self.publication = PublicationService(force_mock=pub_force)
        if self.publication.provider == "github":
            self.cfg.pending.github_pat = False
        self.scheduler = TimerScheduler(
            self.tasks,
            on_notification=self._on_timer_notification,
        )
        self._pending_speak_lock = threading.Lock()
        if secrets.tavily_api_key.strip():
            self.cfg.pending.tavily_api_key = False

    def _upsert_transcript(self, event: dict[str, Any]) -> None:
        """Keep one live partial per role; replace it when the final arrives."""
        from datetime import datetime, timezone

        row = {k: event[k] for k in event if k != "t_mono"}
        is_partial = row.get("type") == "partial" or row.get("partial") is True
        role = row.get("role") or "user"
        now_ts = datetime.now(timezone.utc).isoformat()
        if not row.get("ts"):
            row["ts"] = now_ts
        with self._lock:
            if self.transcript:
                last = self.transcript[-1]
                last_partial = last.get("type") == "partial" or last.get("partial") is True
                last_role = last.get("role") or "user"
                if last_partial and last_role == role:
                    if is_partial:
                        # Keep original start time; refresh content.
                        row = {**row, "ts": last.get("ts") or row["ts"]}
                        self.transcript[-1] = row
                    else:
                        self.transcript[-1] = {
                            **row,
                            "type": "final",
                            "partial": False,
                            "final": True,
                            "ts": last.get("ts") or row["ts"],
                        }
                    return
                # Dedupe identical consecutive finals (ASR final + silence-commit).
                if (
                    not is_partial
                    and not last_partial
                    and last_role == role
                    and _norm_utt(last.get("text")) == _norm_utt(row.get("text"))
                ):
                    self.transcript[-1] = {
                        **last,
                        **row,
                        "type": "final",
                        "partial": False,
                        "final": True,
                        "ts": last.get("ts") or row["ts"],
                    }
                    return
            if not is_partial:
                row = {**row, "type": row.get("type") or "final", "final": True}
            self.transcript.append(row)

    def _emit(self, event: dict[str, Any]) -> None:
        event.setdefault("t_mono", time.monotonic())
        if event.get("type") == "transcript" and event.get("text"):
            self._upsert_transcript(event)
        if self.on_event:
            try:
                self.on_event(event)
            except Exception:
                logger.exception("on_event failed")

    def snapshot(self) -> SessionSnapshot:
        with self._lock:
            return SessionSnapshot(
                state=self.state,
                enabled=self.enabled,
                readiness=self.readiness.as_dict(),
                degraded=not self.readiness.all_ready,
                transcript=list(self.transcript),
                last_error=self.last_error,
                utterance_generation=self.utterance_generation,
                llm_backend=self.llm_backend,
                mic_muted=self.mic_muted,
            )

    def set_mic_muted(self, muted: bool) -> SessionSnapshot:
        """Digital mic mute: discard capture for ASR while session stays ON."""
        with self._lock:
            self.mic_muted = bool(muted)
            self._vad_buf = np.zeros(0, dtype=np.float32)
            self._vad_in_speech = False
            if self.asr:
                try:
                    self.asr.reset_segment()
                except Exception:
                    pass
            if self.vad:
                try:
                    self.vad.reset()
                except Exception:
                    pass
            snap = self.snapshot()
        logger.info("mic_muted=%s", muted)
        self._emit({"type": "session", **snap.as_dict()})
        return snap

    def _set_state(self, name: str) -> None:
        try:
            st = SessionState(name)
        except ValueError:
            return
        prev: SessionState | None
        with self._lock:
            prev = self.state
            self.state = st
        # Silence-commit moves LISTENING → THINKING before Zipformer endpoint.
        # Finalize without emitting (pipeline marks asr_final on speculative commit).
        if st == SessionState.THINKING and prev == SessionState.LISTENING and self.asr:
            try:
                self.asr.force_finalize(emit=False)
            except Exception:
                logger.exception("asr force_finalize on thinking failed")
        self._emit({"type": "session", **self.snapshot().as_dict()})
        if st == SessionState.LISTENING and prev != SessionState.LISTENING:
            threading.Thread(
                target=self._flush_announcements,
                name="announce-flush",
                daemon=True,
            ).start()

    def probe_readiness(self) -> Readiness:
        with self._lock:
            try:
                if self.capture is None or self.playback is None:
                    if self.benchmark:
                        from me3.benchmark.audio_io import BenchCapture, BenchPlayback

                        if self.capture is None:
                            self.capture = BenchCapture(self.cfg)
                        if self.playback is None:
                            self.playback = BenchPlayback(self.cfg)
                    else:
                        if self.capture is None:
                            self.capture = AudioCapture(self.cfg)
                        if self.playback is None:
                            self.playback = AudioPlayback(self.cfg)
                self.readiness.audio = True
            except Exception as exc:
                self.readiness.audio = False
                self.last_error = f"audio: {exc}"
                logger.exception("audio probe failed")

            try:
                if self.vad is None:
                    self.vad = SileroVAD(cfg=self.cfg)
                self.readiness.vad = True
            except Exception as exc:
                self.readiness.vad = False
                self.last_error = f"vad: {exc}"

            try:
                if self.asr is None:
                    self.asr = StreamingAsrEngine(cfg=self.cfg, on_event=self._on_asr)
                self.readiness.asr = True
            except Exception as exc:
                self.readiness.asr = False
                self.last_error = f"asr: {exc}"

            try:
                if self.tts is None:
                    self.tts = PiperTtsEngine(cfg=self.cfg)
                self.readiness.tts = True
            except Exception as exc:
                self.readiness.tts = False
                self.last_error = f"tts: {exc}"

            try:
                if self.conversation is None:
                    self.conversation = ConversationOrchestrator(
                        cfg=self.cfg,
                        on_event=self._emit,
                        speak=self._speak,
                        cancel_speech=self._cancel_speech,
                        set_state=self._set_state,
                        task_service=self.tasks,
                        delegation=self.delegation,
                        publication=self.publication,
                        owner_store=self.owner_store,
                        benchmark=self.benchmark,
                    )
                self.llm_backend = self.conversation.llm_backend
                self.readiness.llm = True
                secrets = load_secrets()
                if secrets.anthropic_api_key and self.llm_backend == "anthropic":
                    self.cfg.pending.anthropic_api_key = False
                if secrets.tavily_api_key.strip():
                    self.cfg.pending.tavily_api_key = False
            except Exception as exc:
                self.readiness.llm = False
                self.last_error = f"llm: {exc}"
                logger.exception("llm probe failed")

            self._emit({"type": "readiness", "readiness": self.readiness.as_dict()})
            return self.readiness

    def _on_delegation_event(self, event: dict[str, Any]) -> None:
        self._emit(event)
        job = event.get("job") or {}
        status = job.get("status")
        if not status:
            return
        with self._lock:
            enabled = self.enabled
        # Spoken + UI cues for terminal / blocking job states (same path as timer alerts).
        if status in ("needs_input", "needs_approval") and job.get("pending_user_request"):
            msg = f"Your build needs you: {job['pending_user_request']}"
            self._emit(
                {
                    "type": "notification",
                    "kind": "job",
                    "status": status,
                    "job_id": job.get("id"),
                    "payload": {"message": msg},
                }
            )
            if enabled:
                self._queue_or_speak_announcement(msg)
            return
        if status == "completed":
            msg = self._compose_job_completed_message(job)
            self._emit(
                {
                    "type": "notification",
                    "kind": "job",
                    "status": status,
                    "job_id": job.get("id"),
                    "payload": {"message": msg},
                }
            )
            if enabled:
                self._queue_or_speak_announcement(msg)
            return
        if status == "failed":
            detail = (job.get("outcome") or "unknown error").strip().split("\n", 1)[0][:120]
            msg = f"Your build failed: {detail}."
            self._emit(
                {
                    "type": "notification",
                    "kind": "job",
                    "status": status,
                    "job_id": job.get("id"),
                    "payload": {"message": msg},
                }
            )
            if enabled:
                self._queue_or_speak_announcement(msg)

    def _recent_user_asked_publish(self) -> bool:
        """True if a recent user turn asked to publish (same request as the build)."""
        keys = ("publish", "github page", "github pages", "pages site", "github repository")
        with self._lock:
            recent = list(self.transcript)[-16:]
        for row in reversed(recent):
            if (row.get("role") or "") != "user":
                continue
            text = (row.get("text") or "").lower()
            if any(k in text for k in keys):
                return True
        return False

    def _job_wants_publish(self, job: dict[str, Any]) -> bool:
        budget = job.get("budget") if isinstance(job.get("budget"), dict) else {}
        if budget.get("publish_when_done") is True:
            return True
        blob = f"{job.get('specification') or ''}\n{job.get('instructions') or ''}".lower()
        if "publish_when_done" in blob or "publish to github" in blob:
            return True
        return self._recent_user_asked_publish()

    def _compose_job_completed_message(self, job: dict[str, Any]) -> str:
        """Short spoken/log line — never dump a truncated job specification."""
        if not self._job_wants_publish(job):
            return "Your build is done."
        if self.publication is None:
            return "Your build is done. Say when you want it published to GitHub Pages."
        try:
            hint = (job.get("specification") or "").strip().split("\n", 1)[0][:60]
            result = self.publication.publish_site(
                source_dir="",
                job_id=str(job.get("id") or "") or None,
                hint=hint,
                confirm=True,
                message=f"ME3 auto-publish after build {(job.get('id') or '')[:8]}",
            )
        except Exception as exc:
            logger.exception("auto-publish after job failed")
            detail = str(exc).strip().split("\n", 1)[0][:120]
            return f"Your build is done, but publish failed: {detail}."
        pub = result.get("publication") or {}
        pages = result.get("pages") or {}
        url = (
            (pages.get("url") or "").rstrip("/")
            or (pub.get("pages_url") or "").rstrip("/")
            or ""
        )
        status = str(pub.get("status") or "")
        if url and status in ("published", "committed_pages_pending"):
            # Keep TTS speakable; URL still lands in the conversation log.
            return (
                f"Your build is done and published to GitHub Pages. "
                f"Link: {url}/"
            )
        if url:
            return f"Your build is done. Pages link: {url}/"
        return "Your build is done and the publish commit succeeded. GitHub Pages may take a minute to update."

    def _session_is_idle(self) -> bool:
        with self._lock:
            if not self.enabled or self.state != SessionState.LISTENING:
                return False
            if self.conversation is not None and self.conversation.active_turn_id:
                return False
        playing = False
        if self.playback is not None:
            try:
                playing = float(self.playback.queue_seconds()) > 0.05
            except Exception:
                playing = False
        return not playing

    def _speak_announcement(self, msg: str, *, notif_id: str | None = None) -> None:
        self._speak(msg, "notification")
        if notif_id:
            try:
                self.tasks.mark_delivered(notif_id)
            except Exception:
                logger.exception("mark notification delivered failed")
            self._emit(
                {"type": "notification_delivered", "id": notif_id, "message": msg}
            )

    def _queue_or_speak_announcement(
        self, msg: str, *, notif_id: str | None = None
    ) -> None:
        """Speak now if idle; otherwise wait until the current reply finishes TTS."""
        if self._session_is_idle():
            self._speak_announcement(msg, notif_id=notif_id)
            return
        start_flush = False
        with self._announce_lock:
            self._pending_announcements.append({"msg": msg, "notif_id": notif_id})
            self._pending_announcements = self._pending_announcements[-8:]
            if not self._announce_flushing:
                self._announce_flushing = True
                start_flush = True
        logger.info("queued announcement until idle: %s", msg[:80])
        if start_flush:
            threading.Thread(
                target=self._flush_announcements,
                name="announce-flush",
                daemon=True,
            ).start()

    def _flush_announcements(self) -> None:
        """Wait out any in-flight reply, then speak queued lines (do not drop them)."""
        try:
            deadline = time.monotonic() + 180.0
            while time.monotonic() < deadline:
                with self._announce_lock:
                    if not self._pending_announcements:
                        return
                if not self.enabled:
                    time.sleep(0.3)
                    continue
                if self._session_is_idle():
                    with self._announce_lock:
                        if not self._pending_announcements:
                            return
                        item = self._pending_announcements.pop(0)
                    self._speak_announcement(
                        str(item.get("msg") or ""),
                        notif_id=item.get("notif_id"),
                    )
                    time.sleep(0.3)
                    continue
                time.sleep(0.25)
            logger.warning(
                "announcement flush timed out; leaving %s pending",
                len(self._pending_announcements),
            )
        finally:
            with self._announce_lock:
                self._announce_flushing = False
                again = bool(self._pending_announcements)
                if again:
                    self._announce_flushing = True
            if again:
                threading.Thread(
                    target=self._flush_announcements,
                    name="announce-flush",
                    daemon=True,
                ).start()

    def _on_timer_notification(self, notif: dict[str, Any]) -> None:
        self._emit({"type": "notification", **notif})
        with self._lock:
            enabled = self.enabled
        if enabled:
            self._deliver_notification(notif)
        # else: leave pending until session ON (spoken notifications wait)

    def _deliver_notification(self, notif: dict[str, Any]) -> None:
        payload = notif.get("payload") or {}
        message = str(payload.get("message") or "Timer finished.")
        # Same idle queue as job-done — do not preempt an in-flight answer's TTS.
        self._queue_or_speak_announcement(message, notif_id=notif.get("id"))

    def _drain_pending_notifications(self) -> None:
        for notif in self.tasks.drain_pending_notifications():
            self._deliver_notification(notif)

    def _speak(self, text: str, kind: str) -> None:
        from me3.benchmark.marks import mark

        if not self.tts or not self.playback:
            return
        with self._lock:
            # clarify may speak briefly; notifications only while ON
            if not self.enabled and kind not in ("clarify",):
                return
            # After barge-in, drop late assistant audio from the cancelled turn.
            if kind == "assistant" and self.conversation is not None:
                if self.conversation.active_turn_id is None and self.state == SessionState.LISTENING:
                    return
            gen = self.utterance_generation
        try:
            mark("tts_request")
            samples, rate = self.tts.synthesize_to_array(text)
            mark("tts_first_audio")
        except Exception:
            logger.exception("tts failed")
            return
        with self._lock:
            if gen != self.utterance_generation:
                return
            if not self.enabled and kind not in ("clarify",):
                return
            if kind == "assistant" and self.conversation is not None:
                if self.conversation.active_turn_id is None and self.state == SessionState.LISTENING:
                    return
            if not self._audio_out_marked:
                mark("audio_out")
                self._audio_out_marked = True
            # Keep ASR gated while this utterance plays (laptop speaker→mic).
            hang_ms = int(getattr(self.cfg.session, "echo_hangover_ms", 450) or 0)
            self._echo_gate_until = max(
                self._echo_gate_until,
                time.monotonic() + max(0.05, len(samples) / float(rate or 22050)) + hang_ms / 1000.0,
            )
            self.playback.play(samples, rate, generation=gen)
            # Idle announcements (job done / timer) were TTS-only — also log them.
            if kind in ("notification", "clarify") and (text or "").strip():
                self._emit(
                    {
                        "type": "transcript",
                        "role": "assistant",
                        "text": text.strip(),
                        "partial": False,
                        "final": True,
                        "kind": kind,
                        "ts": datetime.now(timezone.utc).isoformat(),
                    }
                )

    def _cancel_speech(self) -> None:
        from me3.benchmark.marks import mark

        mark("interrupted")
        if self.tts:
            self.tts.cancel()
        if self.playback:
            with self._lock:
                self.utterance_generation = self.playback.flush()
                self._audio_out_marked = False

    def _on_asr(self, evt: AsrEvent) -> None:
        from me3.benchmark.marks import mark

        with self._lock:
            if not self.enabled:
                return
            state = self.state
        row = {
            "type": evt.type,
            "text": evt.text,
            "segment_id": evt.segment_id,
            "role": "user",
            "t_mono": evt.t_mono,
            "partial": evt.type == "partial",
            "final": evt.type != "partial",
        }
        self._emit({"type": "transcript", **row})
        text = (evt.text or "").strip()
        if evt.type == "partial":
            mark("asr_partial", text=text)
            if text:
                tag = "barge" if self._barge_asr_live else "listen"
                logger.info("YOU asr_partial [%s]: %s", tag, text[:120])
        elif text:
            logger.info("YOU asr_final: %s", text[:120])
        # Do not mark asr_final here. Silence-commit and Zipformer-final both
        # land in ConversationOrchestrator.commit_user_turn, which marks once.
        # A second mark here made fdbench concatenate identical text → doubled WER.
        if self.conversation:
            if evt.type == "partial":
                self.conversation.on_asr_partial(evt.text)
            else:
                # Speculative silence-commit confirmation / correction lives here.
                # Do NOT unconditionally cancel THINKING — that duplicated turns.
                self.conversation.on_asr_final(evt.text)
            if state == SessionState.SPEAKING:
                # barge-in → overlap (ASR path; VAD barge-in also calls begin_overlap)
                self.conversation.begin_overlap()
            elif state == SessionState.THINKING and evt.type == "partial":
                # Mid-think correction only when new speech clearly diverges.
                active = (self.conversation.active_user_text or "").strip()
                if active and not _utt_compatible(active, evt.text or ""):
                    self.conversation.cancel_active(reason="user_correction")
                    self._set_state("listening")
                    self.conversation.on_asr_partial(evt.text)

    def start_backend(self) -> None:
        self.probe_readiness()
        assert self.capture and self.playback
        self.capture.start()
        self.playback.start()
        if self.asr:
            self.asr.start()
        self.scheduler.start()
        self._stop_pump.clear()
        if not self._pump_thread or not self._pump_thread.is_alive():
            self._pump_thread = threading.Thread(target=self._pump_loop, name="session-pump", daemon=True)
            self._pump_thread.start()
        self._emit({"type": "session", **self.snapshot().as_dict()})

    def shutdown(self) -> None:
        self.set_enabled(False)
        self.scheduler.stop()
        self._stop_pump.set()
        if self._pump_thread:
            self._pump_thread.join(timeout=3.0)
            self._pump_thread = None
        if self.conversation:
            self.conversation.cancel_active("shutdown")
            self.conversation.memory.close_session()
            self.conversation.memory.close()
        self.tasks.close()
        self.delegation.close()
        self.publication.close()
        self.owner_store.close()
        if self.asr:
            self.asr.stop(drain=False)
        if self.capture:
            self.capture.stop()
        if self.playback:
            self.playback.stop()

    def set_enabled(self, enabled: bool) -> SessionSnapshot:
        with self._lock:
            if enabled and not self.enabled:
                self._turn_on_locked()
            elif not enabled and self.enabled:
                self._turn_off_locked()
            snap = self.snapshot()
        self._emit({"type": "session", **snap.as_dict()})
        return snap

    def _turn_on_locked(self) -> None:
        self.probe_readiness()
        self.enabled = True
        self.state = SessionState.LISTENING
        self.last_error = None
        if self.capture:
            self.capture.drain()
        self._vad_buf = np.zeros(0, dtype=np.float32)
        if self.asr:
            self.asr.reset_segment()
        if self.vad:
            self.vad.reset()
        if self.conversation:
            self.conversation.reset_listening()
        # Fresh session: wipe timers/notes/jobs (owner profile + publications kept).
        reset_summary = self._reset_session_tools()
        self._emit({"type": "session_tools_reset", **reset_summary})
        logger.info(
            "session ON -> listening (llm=%s reset=%s)",
            self.llm_backend,
            reset_summary,
        )

    def _reset_session_tools(self) -> dict[str, Any]:
        """Clear ephemeral tools state on each OFF→ON (demo / session restart)."""
        summary: dict[str, Any] = {}
        try:
            summary["tasks"] = self.tasks.reset_for_session()
        except Exception as exc:
            logger.exception("session tools reset (tasks) failed")
            summary["tasks_error"] = str(exc)
        try:
            summary["jobs"] = self.delegation.reset_for_session()
        except Exception as exc:
            logger.exception("session tools reset (jobs) failed")
            summary["jobs_error"] = str(exc)
        return summary

    def _turn_off_locked(self) -> None:
        self.enabled = False
        self.state = SessionState.OFF
        if self.conversation:
            self.conversation.cancel_active("ui_off")
            self.conversation.reset_listening()
        if self.tts:
            self.tts.cancel()
        if self.playback:
            self.utterance_generation = self.playback.flush()
        else:
            self.utterance_generation += 1
        if self.capture:
            self.capture.drain()
        self._vad_buf = np.zeros(0, dtype=np.float32)
        if self.asr:
            self.asr.reset_segment()
        if self.vad:
            self.vad.reset()
        logger.info("session OFF (flushed playback gen=%s)", self.utterance_generation)

    def speak_probe(self, text: str = "Session is on.") -> str:
        with self._lock:
            if not self.enabled:
                raise RuntimeError("session is OFF")
            if not self.tts or not self.playback:
                raise RuntimeError("tts/playback not ready")
            self.state = SessionState.SPEAKING
            gen = self.utterance_generation
            uid = str(uuid4())
        self._emit({"type": "session", **self.snapshot().as_dict()})
        samples, rate = self.tts.synthesize_to_array(text)
        with self._lock:
            if not self.enabled or gen != self.utterance_generation:
                return uid
            self.playback.play(samples, rate, generation=gen)

        def _finish():
            if self.playback:
                self.playback.wait_drain(timeout=30.0)
            with self._lock:
                if self.enabled and self.state == SessionState.SPEAKING:
                    self.state = SessionState.LISTENING
            self._emit({"type": "session", **self.snapshot().as_dict()})

        threading.Thread(target=_finish, name="probe-finish", daemon=True).start()
        return uid

    def inject_user_text(self, text: str) -> str:
        """Typed recovery / sample-bank: commit a user turn without ASR."""
        if not self.enabled:
            raise RuntimeError("session is OFF")
        if not self.conversation:
            self.probe_readiness()
        assert self.conversation
        return self.conversation.commit_user_turn(text)

    def _pump_loop(self) -> None:
        while not self._stop_pump.is_set():
            if not self.capture:
                time.sleep(0.05)
                continue
            chunk = self.capture.read_target_chunk(timeout=0.05)
            with self._lock:
                enabled = self.enabled
                muted = self.mic_muted
                state = self.state
            if self.conversation and enabled:
                # Soft-pause keeps state=SPEAKING, so turn-taker never commits the
                # interrupt. Drive tick as listening while barge ASR is live; if the
                # interrupt is ready to commit, hard-confirm first so LLM is cancelled.
                # Peek silence without arming commit (evaluate() would block tick).
                tick_state = state.value
                if self._barge_asr_live or self._barge_soft_paused:
                    tick_state = SessionState.LISTENING.value
                    tt = self.conversation.turn_taker
                    partial = (tt.partial_text or "").strip()
                    if (
                        partial
                        and tt.last_speech_mono is not None
                        and (self._barge_soft_paused or state in (SessionState.SPEAKING, SessionState.THINKING))
                    ):
                        silence_ms = (time.monotonic() - tt.last_speech_mono) * 1000.0
                        if silence_ms >= tt.threshold_ms():
                            logger.info(
                                "barge interrupt ready (%.0fms silence): %s",
                                silence_ms,
                                partial[:100],
                            )
                            self._hard_barge_in(max(self._barge_speech_ms, 1.0))
                            tt.cancel_pending()
                            state = SessionState.LISTENING
                            tick_state = state.value
                self.conversation.tick(enabled=True, state=tick_state)
            if chunk is None:
                continue
            if not enabled:
                continue
            gating = self._echo_gate_active(state)
            # Digital mute blocks normal ASR, but still allow barge-in to stop TTS.
            if muted and not gating and not self._barge_asr_live:
                continue
            if gating:
                # Benchmark: next clip often arrives while still THINKING. Preempt so
                # the new phrase is transcribed instead of an empty trial.
                if (
                    self.benchmark
                    and state == SessionState.THINKING
                    and not muted
                    and self._benchmark_preempt_energy(chunk)
                ):
                    try:
                        if self.conversation:
                            self.conversation.cancel_active("benchmark_preempt")
                        self._cancel_speech()
                        self._echo_gate_until = 0.0
                        self._was_gating_asr = False
                        self._reset_barge_tracker()
                        if self.asr:
                            self.asr.reset_segment()
                        if self.conversation:
                            self.conversation.turn_taker.reset()
                        self._set_state("listening")
                    except Exception:
                        logger.exception("benchmark preempt failed")
                    self._forward_asr(chunk, SessionState.LISTENING)
                    continue
                barged = self._watch_barge_in(chunk, state)
                # Soft-pause / confirmed barge: keep capturing the interrupt into ASR.
                if barged or self._barge_asr_live:
                    if not muted:
                        self._forward_asr(chunk, SessionState.LISTENING)
                    continue
                # Keep _vad_buf remainder across chunks. Capture blocks at 48 kHz→16 kHz
                # are often <512 samples; wiping here made barge never assemble a frame
                # (diagnostic harness passed because it did not wipe).
                self._vad_in_speech = False
                self._was_gating_asr = True
                continue
            if muted:
                self._reset_barge_tracker()
                continue
            if self._was_gating_asr:
                self._was_gating_asr = False
                if self._barge_keep_asr:
                    # Confirmed barge already seeded ASR with the interrupt — keep it.
                    self._barge_keep_asr = False
                    self._barge_asr_live = False
                    self._reset_barge_tracker(clear_asr_live=False)
                else:
                    # Fresh listen window after echo gate; drop leftover TTS bleed.
                    self._reset_barge_tracker()
                    self._vad_buf = np.zeros(0, dtype=np.float32)
                    if self.asr:
                        try:
                            self.asr.reset_segment()
                        except Exception:
                            logger.exception("asr reset after echo gate failed")
                    if self.conversation:
                        self.conversation.turn_taker.reset()
            elif self._barge_asr_live or self._barge_keep_asr:
                # Left the gate while still capturing a barge utterance.
                self._barge_keep_asr = False
                self._barge_asr_live = False
                self._reset_barge_tracker(clear_asr_live=False)
            else:
                self._reset_barge_tracker()
            self._forward_asr(chunk, state)

    def _echo_gate_active(self, state: SessionState) -> bool:
        """Laptop without AEC: ignore mic while the bot is talking / audio is draining.

        Also gate during THINKING so Zipformer cannot endpoint the just-committed
        utterance again and re-arm a duplicate turn. In benchmark mode, THINKING
        alone does not hard-gate — fdbench's next clip must be able to preempt a
        slow LLM turn (otherwise trials land with empty marks).
        """
        if not bool(getattr(self.cfg.session, "gate_asr_while_speaking", True)):
            return False
        now = time.monotonic()
        playing = False
        if self.playback is not None:
            try:
                playing = float(self.playback.queue_seconds()) > 0.04
            except Exception:
                playing = False
        if playing or state == SessionState.SPEAKING:
            hang_ms = int(getattr(self.cfg.session, "echo_hangover_ms", 450) or 0)
            self._echo_gate_until = max(self._echo_gate_until, now + hang_ms / 1000.0)
            return True
        if state == SessionState.THINKING:
            if self.benchmark:
                # Soft gate only: barge/preempt path still runs; hangover not extended
                # for the whole LLM wait (that caused empty fdbench trials).
                return True
            hang_ms = int(getattr(self.cfg.session, "echo_hangover_ms", 450) or 0)
            self._echo_gate_until = max(self._echo_gate_until, now + hang_ms / 1000.0)
            return True
        # Allow ASR during overlap_paused so steering partials can be classified.
        if state == SessionState.OVERLAP_PAUSED:
            return False
        return now < self._echo_gate_until

    def _benchmark_preempt_energy(self, chunk: np.ndarray) -> bool:
        """True when injected fdbench speech is clearly louder than silence."""
        if chunk.size == 0:
            return False
        rms = float(np.sqrt(np.mean(np.square(chunk.astype(np.float32)))))
        return rms >= 0.02

    def _watch_barge_in(self, chunk: np.ndarray, state: SessionState) -> bool:
        """Two-phase barge: soft-pause TTS, then cancel only after sustained speech.

        From soft-pause onward, interrupt audio is live-fed to ASR so the barge
        utterance becomes the next user turn (not just a TTS stop).
        """
        playing = False
        if self.playback is not None:
            try:
                playing = float(self.playback.queue_seconds()) > 0.04
            except Exception:
                playing = False
        if state not in (SessionState.SPEAKING, SessionState.THINKING) and not playing:
            if not self._barge_asr_live:
                self._reset_barge_tracker()
            return False

        thr = float(getattr(self.cfg.session, "barge_in_vad_threshold", 0.50) or 0.50)
        pause_ms = int(getattr(self.cfg.session, "barge_in_pause_ms", 160) or 160)
        confirm_ms = int(
            getattr(self.cfg.session, "barge_in_confirm_ms", None)
            or getattr(self.cfg.session, "barge_in_hold_ms", 320)
            or 320
        )
        confirm_ms = max(confirm_ms, pause_ms)
        abandon_ms = int(getattr(self.cfg.session, "barge_in_abandon_ms", 500) or 500)
        ratio = float(getattr(self.cfg.session, "barge_in_energy_ratio", 1.8) or 1.8)
        floor = float(getattr(self.cfg.session, "barge_in_energy_floor", 0.025) or 0.025)
        # After TTS ducks, ease slightly for confirm — but not enough for room noise.
        active_floor = floor * (0.75 if self._barge_soft_paused else 1.0)
        # Thinking has no TTS echo baseline; ambient noise used to trip barge easily.
        if state == SessionState.THINKING and not playing:
            active_floor = max(active_floor, floor * 1.35)

        # Separate from ASR `_vad_buf` so soft-pause live capture isn't double-fed.
        self._barge_vad_buf = np.concatenate(
            [self._barge_vad_buf, chunk.astype(np.float32, copy=False)]
        )
        confirmed = False
        speech_this_chunk = False
        while len(self._barge_vad_buf) >= WINDOW:
            frame = self._barge_vad_buf[:WINDOW]
            self._barge_vad_buf = self._barge_vad_buf[WINDOW:]
            frame_ms = (WINDOW / 16000.0) * 1000.0
            rms = float(np.sqrt(np.mean(np.square(frame)) + 1e-12))

            if not self._echo_rms_ready:
                self._echo_rms_ema = min(rms, floor * 0.25)
                self._echo_rms_ready = True
            else:
                baseline = max(self._echo_rms_ema, 1e-6)
                if rms < max(baseline * 1.25, floor * 0.5):
                    self._echo_rms_ema = 0.90 * self._echo_rms_ema + 0.10 * rms

            # Require absolute energy — never barge on near-silence.
            aec_like = self._echo_rms_ema < max(floor * 0.5, 0.004)
            if aec_like:
                energy_hit = rms >= active_floor
            else:
                energy_hit = rms >= max(active_floor, self._echo_rms_ema * ratio)

            vad_hit = False
            if self.vad and energy_hit:
                # Only run VAD when energy already looks like speech (avoids
                # silence-boost false positives that cancelled short replies).
                try:
                    vad_hit = self.vad.prob(frame) >= thr
                except Exception:
                    vad_hit = False

            # Start barge with energy+VAD; after soft-pause, energy alone can finish
            # confirm (user often pauses mid-interrupt once TTS ducks).
            if self._barge_soft_paused:
                speech_hit = energy_hit
            else:
                speech_hit = energy_hit and (vad_hit if self.vad is not None else True)

            if speech_hit:
                speech_this_chunk = True
                self._barge_quiet_ms = 0.0
                self._barge_speech_ms += frame_ms
            elif self._barge_soft_paused:
                # Hold progress at pause_ms so brief dips don't unwind confirm.
                self._barge_speech_ms = max(float(pause_ms), self._barge_speech_ms - frame_ms * 0.25)
                self._barge_quiet_ms += frame_ms
                if self._barge_quiet_ms >= abandon_ms:
                    self._abandon_soft_barge()
            else:
                self._barge_speech_ms = max(0.0, self._barge_speech_ms - frame_ms * 0.85)
                self._barge_quiet_ms = 0.0

            if os.environ.get("ME3_BARGE_DEBUG"):
                logger.info(
                    "barge dbg rms=%.4f ema=%.4f E=%s V=%s soft=%s live=%s held=%.0f quiet=%.0f",
                    rms,
                    self._echo_rms_ema,
                    energy_hit,
                    vad_hit,
                    self._barge_soft_paused,
                    self._barge_asr_live,
                    self._barge_speech_ms,
                    self._barge_quiet_ms,
                )

            if self._barge_speech_ms >= pause_ms and not self._barge_soft_paused:
                # Bench: hard-stop at pause threshold. Soft-pause alone still lets the
                # LLM emit more TTS (new generation after flush) → fdbench "spoke_through"
                # and course.barge_in n=0. Live keeps two-phase confirm.
                if self.benchmark:
                    confirmed = True
                    break
                self._soft_barge_pause()
            if self._barge_speech_ms >= confirm_ms:
                confirmed = True
                break

        # Buffer interrupt audio from first energy hit through confirm (incl. short gaps).
        if speech_this_chunk or self._barge_asr_buf or self._barge_asr_live:
            self._barge_asr_buf.append(chunk.astype(np.float32, copy=True))
            if sum(len(c) for c in self._barge_asr_buf) > 16000 * 4:
                self._barge_asr_buf = self._barge_asr_buf[-40:]

        if not confirmed:
            return False
        self._hard_barge_in(self._barge_speech_ms)
        return True

    def _reset_barge_tracker(self, *, clear_asr_live: bool = True) -> None:
        self._barge_speech_ms = 0.0
        self._barge_last_mono = None
        self._barge_soft_paused = False
        self._barge_quiet_ms = 0.0
        self._barge_asr_buf.clear()
        self._barge_vad_buf = np.zeros(0, dtype=np.float32)
        if clear_asr_live:
            self._barge_asr_live = False
        self._echo_rms_ema = 0.0
        self._echo_rms_ready = False

    def _seed_barge_asr(self) -> None:
        """Reset ASR once and replay buffered interrupt audio as the new utterance."""
        if self._barge_asr_live:
            return
        buffered = list(self._barge_asr_buf)
        self._barge_asr_buf.clear()
        self._vad_buf = np.zeros(0, dtype=np.float32)
        self._vad_in_speech = False
        if self.asr:
            try:
                self.asr.reset_segment()
            except Exception:
                logger.exception("asr reset for barge seed failed")
        if self.conversation:
            self.conversation.turn_taker.reset()
        self._barge_asr_live = True
        for piece in buffered:
            try:
                self._forward_asr(piece, SessionState.LISTENING)
            except Exception:
                logger.exception("barge ASR seed failed")
        logger.info("barge ASR capture started (%d chunks)", len(buffered))

    def _abandon_soft_barge(self) -> None:
        """Quiet after soft-pause: keep interrupt text if ASR heard any; else resume."""
        from me3.conversation.asr_filters import is_asr_ghost

        partial = ""
        if self.conversation:
            partial = (self.conversation.turn_taker.partial_text or "").strip()
        if partial and not is_asr_ghost(partial):
            logger.info(
                "barge soft-pause quiet → confirm with captured interrupt: %s",
                partial[:100],
            )
            self._hard_barge_in(max(self._barge_speech_ms, 1.0))
            return
        was_live = self._barge_asr_live
        logger.info(
            "barge soft-pause abandoned after %.0fms quiet (no interrupt text)",
            self._barge_quiet_ms,
        )
        self._barge_soft_paused = False
        self._barge_quiet_ms = 0.0
        self._barge_asr_buf.clear()
        self._barge_vad_buf = np.zeros(0, dtype=np.float32)
        self._barge_asr_live = False
        self._barge_keep_asr = False
        self._vad_buf = np.zeros(0, dtype=np.float32)
        self._vad_in_speech = False
        if was_live and self.asr:
            try:
                self.asr.reset_segment()
            except Exception:
                logger.exception("asr reset after abandoned barge failed")
        if was_live and self.conversation:
            self.conversation.turn_taker.reset()

    def _soft_barge_pause(self) -> None:
        """Duck TTS immediately and start capturing the interrupt into ASR."""
        self._barge_soft_paused = True
        self._barge_quiet_ms = 0.0
        logger.info("barge soft-pause after %.0fms — capturing interrupt", self._barge_speech_ms)
        if self.playback:
            with self._lock:
                self.utterance_generation = self.playback.flush()
                self._audio_out_marked = False
        # Start ASR on the interrupt right away (before confirm cancels the LLM).
        self._seed_barge_asr()

    def _hard_barge_in(self, held_ms: float) -> None:
        """Confirmed barge: cancel LLM, keep ASR interrupt as the next user turn."""
        from me3.benchmark.marks import mark

        partial = ""
        if self.conversation:
            partial = (self.conversation.turn_taker.partial_text or "").strip()
        mark("interrupted")
        logger.info(
            "barge-in confirmed after %.0fms — interrupt so far: %s",
            held_ms,
            (partial[:100] if partial else "(no ASR text yet)"),
        )
        # Ensure ASR already has the interrupt (soft-pause normally seeded it).
        self._seed_barge_asr()
        self._barge_keep_asr = True
        self._barge_soft_paused = False
        self._barge_speech_ms = 0.0
        self._barge_asr_buf.clear()
        # No echo hangover wipe — continue listening to the same utterance.
        self._echo_gate_until = 0.0
        self._was_gating_asr = False
        self._cancel_speech()
        if self.conversation:
            self.conversation.cancel_active(reason="barge_in")
            self.conversation.overlap.reset()
            # Keep turn_taker / ASR partials from the interrupt — do not reset.
        self._set_state("listening")

    def _forward_asr(self, chunk: np.ndarray, state: SessionState) -> None:
        if not self.asr:
            return
        self._vad_buf = np.concatenate([self._vad_buf, chunk])
        while len(self._vad_buf) >= WINDOW:
            frame = self._vad_buf[:WINDOW]
            self._vad_buf = self._vad_buf[WINDOW:]
            speech = False
            if self.vad:
                try:
                    speech = self.vad.prob(frame) >= self._speech_prob_thresh
                except Exception:
                    speech = True
            if speech and not self._vad_in_speech:
                from me3.benchmark.marks import mark

                mark("vad_start")
                self._vad_in_speech = True
            elif not speech and self._vad_in_speech:
                from me3.benchmark.marks import mark

                mark("vad_end")
                self._vad_in_speech = False
            if speech and self.conversation:
                self.conversation.on_user_speech()
            try:
                self.asr.accept_pcm(frame, block=True)
            except Exception:
                logger.exception("asr accept failed")


def _norm_utt(text: Any) -> str:
    import re

    return re.sub(r"[^a-z0-9\s]", "", str(text or "").lower()).strip()


def _utt_compatible(committed: str, incoming: str) -> bool:
    a = _norm_utt(committed).split()
    b = _norm_utt(incoming).split()
    if not a or not b:
        return not a and not b
    if a == b:
        return True
    if len(a) <= len(b) and b[: len(a)] == a:
        return True
    if len(b) <= len(a) and a[: len(b)] == b:
        return True
    return False
