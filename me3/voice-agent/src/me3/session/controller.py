"""Session controller Phase 3: Off/Listening/Thinking/Speaking/OverlapPaused."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
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

        self.capture: Any = None
        self.playback: Any = None
        self.asr: StreamingAsrEngine | None = None
        self.tts: PiperTtsEngine | None = None
        self.vad: SileroVAD | None = None
        self.conversation: ConversationOrchestrator | None = None
        secrets = load_secrets()
        self.tasks = TaskService(
            timezone_name=self.cfg.session.timezone,
            force_mock_retrieval=not bool(secrets.tavily_api_key.strip()),
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
        row = {k: event[k] for k in event if k != "t_mono"}
        is_partial = row.get("type") == "partial" or row.get("partial") is True
        role = row.get("role") or "user"
        with self._lock:
            if self.transcript:
                last = self.transcript[-1]
                last_partial = last.get("type") == "partial" or last.get("partial") is True
                last_role = last.get("role") or "user"
                if last_partial and last_role == role:
                    if is_partial:
                        self.transcript[-1] = row
                    else:
                        self.transcript[-1] = {**row, "type": "final", "partial": False, "final": True}
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
        with self._lock:
            self.state = st
        self._emit({"type": "session", **self.snapshot().as_dict()})

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
        if status in ("needs_input", "needs_approval") and job.get("pending_user_request"):
            with self._lock:
                enabled = self.enabled
            if enabled:
                self._speak(
                    f"Job needs you: {job['pending_user_request']}",
                    "notification",
                )

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
        self._speak(message, "notification")
        self.tasks.mark_delivered(notif["id"])
        self._emit({"type": "notification_delivered", "id": notif["id"], "message": message})

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
            gen = self.utterance_generation
            if kind == "assistant" and self.state not in (
                SessionState.SPEAKING,
                SessionState.THINKING,
                SessionState.LISTENING,
            ):
                # allow first sentence to move into speaking via set_state
                pass
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
            if not self._audio_out_marked:
                mark("audio_out")
                self._audio_out_marked = True
            self.playback.play(samples, rate, generation=gen)

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
        if evt.type == "partial":
            mark("asr_partial", text=evt.text or "")
        else:
            mark("asr_final", text=evt.text or "")
        if self.conversation:
            if evt.type == "partial":
                self.conversation.on_asr_partial(evt.text)
            else:
                self.conversation.on_asr_final(evt.text)
            if state == SessionState.SPEAKING:
                # barge-in → overlap
                self.conversation.begin_overlap()
            elif state == SessionState.THINKING:
                # user correction while thinking — cancel generation
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
        # Drain notifications that fired while OFF (timers keep running)
        threading.Thread(
            target=self._drain_pending_notifications,
            name="notif-drain",
            daemon=True,
        ).start()
        logger.info("session ON -> listening (llm=%s)", self.llm_backend)

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
                self.conversation.tick(enabled=True, state=state.value)
            if chunk is None:
                continue
            if not enabled:
                continue
            # Digital mute: keep capture/device path alive, do not feed ASR/VAD
            if muted:
                continue
            self._forward_asr(chunk, state)

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
            # Always feed ASR while ON so finals/partials update (session gate already applied)
            try:
                self.asr.accept_pcm(frame, block=True)
            except Exception:
                logger.exception("asr accept failed")
