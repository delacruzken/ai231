"""Phase 8: trial reset + bench audio I/O + marks no-op without fdbench."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from me3.benchmark.audio_io import BenchCapture, BenchPlayback
from me3.benchmark.marks import mark, me3_model_card
from me3.benchmark.reset import reset_controller_for_trial
from me3.session.controller import SessionController, SessionState


def test_marks_noop_without_fdbench():
    mark("vad_end")  # must not raise
    card = me3_model_card()
    assert "ASR" in card and "LLM" in card


def test_mic_mute_keeps_session_enabled(monkeypatch):
    monkeypatch.setenv("OPENCODE_FORCE_MOCK", "true")
    ctrl = SessionController(benchmark=True)
    ctrl.probe_readiness = MagicMock(return_value=ctrl.readiness)  # type: ignore
    ctrl.enabled = True
    ctrl.state = SessionState.LISTENING
    snap = ctrl.set_mic_muted(True)
    assert snap.mic_muted is True
    assert snap.enabled is True
    snap2 = ctrl.set_mic_muted(False)
    assert snap2.mic_muted is False
    ctrl.shutdown()


def test_bench_capture_playback_roundtrip():
    got: list[tuple[int, int]] = []

    def on_audio(samples, rate):
        got.append((len(samples), rate))

    cap = BenchCapture()
    play = BenchPlayback(on_audio=on_audio)
    cap.start()
    play.start()
    pcm = (np.zeros(1600, dtype=np.float32) * 0).astype(np.float32)
    # push int16 silence
    raw = (np.zeros(512, dtype="<i2")).tobytes()
    cap.push_pcm16(raw)
    chunk = cap.read_target_chunk(timeout=0.2)
    assert chunk is not None and chunk.dtype == np.float32
    gen = play.flush()
    play.play(pcm, 22050, generation=gen)
    assert got and got[0][1] == 16000
    cap.stop()
    play.stop()


def test_trial_reset(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ME3_UI_TOKEN", "t")
    monkeypatch.setenv("OPENCODE_FORCE_MOCK", "true")
    ctrl = SessionController(benchmark=True)
    # Avoid real model load in readiness
    ctrl.probe_readiness = MagicMock(return_value=ctrl.readiness)  # type: ignore
    ctrl.readiness.audio = True
    ctrl.readiness.vad = True
    ctrl.readiness.asr = True
    ctrl.readiness.tts = True
    ctrl.readiness.llm = True
    ctrl.capture = BenchCapture()
    ctrl.playback = BenchPlayback()
    ctrl.capture.start()
    ctrl.playback.start()
    ctrl.enabled = True
    ctrl.state = SessionState.LISTENING
    ctrl.transcript.append({"type": "final", "text": "hi", "role": "user"})
    # seed a timer
    ctrl.tasks.handle("timer.create", label="bench", duration_seconds=60)
    summary = reset_controller_for_trial(ctrl)
    assert summary.get("ok") is True
    assert ctrl.enabled is True
    assert ctrl.transcript == []
    assert ctrl.tasks.store.list_timers(status="active") == []
    ctrl.shutdown()
