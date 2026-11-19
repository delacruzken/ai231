"""Barge-in energy baseline / AEC-path unit checks (no mic)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from me3.config import AppConfig, SessionConfig
from me3.session.controller import SessionController, SessionState
from me3.speech.vad import WINDOW


def _ctrl(**session_kw) -> SessionController:
    cfg = AppConfig(session=SessionConfig(**session_kw))
    ctrl = SessionController(cfg=cfg)
    ctrl.state = SessionState.SPEAKING
    ctrl.enabled = True
    ctrl.playback = MagicMock()
    ctrl.playback.queue_seconds.return_value = 1.0
    ctrl.playback.flush.return_value = 2
    ctrl.vad = None  # energy-only path
    ctrl.conversation = MagicMock()
    ctrl.tts = MagicMock()
    return ctrl


def test_barge_in_fires_on_soft_speech_after_quiet_echo():
    ctrl = _ctrl(
        barge_in_hold_ms=64,  # ~2 frames at 512/16k
        barge_in_energy_floor=0.005,
        barge_in_energy_ratio=1.55,
    )
    quiet = (np.random.randn(WINDOW).astype(np.float32) * 0.0008).astype(np.float32)
    speech = (np.random.randn(WINDOW).astype(np.float32) * 0.02).astype(np.float32)
    # Seed baseline with quiet echo/AEC residual.
    assert ctrl._watch_barge_in(quiet, SessionState.SPEAKING) is False
    assert ctrl._echo_rms_ready is True
    # Sustained soft speech should hard-stop.
    fired = False
    for _ in range(4):
        if ctrl._watch_barge_in(speech, SessionState.SPEAKING):
            fired = True
            break
    assert fired
    ctrl.playback.flush.assert_called()
    ctrl.conversation.cancel_active.assert_called()


def test_barge_baseline_not_locked_to_first_loud_frame():
    ctrl = _ctrl(barge_in_hold_ms=64, barge_in_energy_floor=0.005, barge_in_energy_ratio=1.55)
    loud = (np.random.randn(WINDOW).astype(np.float32) * 0.05).astype(np.float32)
    ctrl._watch_barge_in(loud, SessionState.SPEAKING)
    # Seed must stay at/below floor so ratio remains beatable.
    assert ctrl._echo_rms_ema <= 0.005 + 1e-6
