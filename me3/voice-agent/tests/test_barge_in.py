"""Barge-in energy / confirm-path unit checks (no mic)."""

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
    ctrl.asr = MagicMock()
    return ctrl


def _feed(ctrl: SessionController, frame: np.ndarray, n: int) -> bool:
    for _ in range(n):
        if ctrl._watch_barge_in(frame, SessionState.SPEAKING):
            return True
    return False


def test_noise_below_floor_does_not_barge():
    ctrl = _ctrl(
        barge_in_pause_ms=64,
        barge_in_confirm_ms=96,
        barge_in_energy_floor=0.012,
        barge_in_energy_ratio=1.8,
    )
    noise = (np.random.randn(WINDOW).astype(np.float32) * 0.0004).astype(np.float32)
    assert _feed(ctrl, noise, 12) is False
    ctrl.conversation.cancel_active.assert_not_called()
    assert ctrl._barge_asr_live is False


def test_soft_pause_seeds_asr_then_confirm_keeps_interrupt():
    ctrl = _ctrl(
        barge_in_pause_ms=64,
        barge_in_confirm_ms=128,
        barge_in_energy_floor=0.012,
        barge_in_energy_ratio=1.8,
    )
    quiet = (np.random.randn(WINDOW).astype(np.float32) * 0.0005).astype(np.float32)
    speech = (np.random.randn(WINDOW).astype(np.float32) * 0.04).astype(np.float32)
    assert ctrl._watch_barge_in(quiet, SessionState.SPEAKING) is False

    # ~2 frames → soft pause + ASR seed of interrupt audio
    assert ctrl._watch_barge_in(speech, SessionState.SPEAKING) is False
    assert ctrl._watch_barge_in(speech, SessionState.SPEAKING) is False
    assert ctrl._barge_soft_paused is True
    assert ctrl._barge_asr_live is True
    ctrl.playback.flush.assert_called()
    ctrl.asr.reset_segment.assert_called()
    ctrl.conversation.cancel_active.assert_not_called()

    # More sustained speech → confirm + cancel, keep ASR (no turn_taker wipe)
    assert _feed(ctrl, speech, 4) is True
    ctrl.conversation.cancel_active.assert_called()
    assert ctrl._barge_keep_asr is True
    assert ctrl._barge_asr_live is True
    assert ctrl._echo_gate_until == 0.0
    ctrl.conversation.turn_taker.reset.assert_called()  # only from soft-pause seed
    assert ctrl.state == SessionState.LISTENING


def test_echo_gate_exit_keeps_barge_asr():
    ctrl = _ctrl(
        barge_in_pause_ms=64,
        barge_in_confirm_ms=96,
        barge_in_energy_floor=0.012,
    )
    speech = (np.random.randn(WINDOW).astype(np.float32) * 0.05).astype(np.float32)
    assert _feed(ctrl, speech, 8) is True
    ctrl.asr.reset_segment.reset_mock()
    ctrl.conversation.turn_taker.reset.reset_mock()

    ctrl._was_gating_asr = True
    ctrl.state = SessionState.LISTENING
    ctrl.playback.queue_seconds.return_value = 0.0
    # Simulate first post-gate pump path
    assert ctrl._barge_keep_asr is True
    ctrl._was_gating_asr = True
    ctrl._barge_keep_asr = True
    ctrl._barge_asr_live = True
    # Mimic pump exit branch
    ctrl._was_gating_asr = False
    ctrl._barge_keep_asr = False
    ctrl._barge_asr_live = False
    ctrl._reset_barge_tracker(clear_asr_live=False)
    ctrl.asr.reset_segment.assert_not_called()


def test_sub_window_chunks_accumulate_then_confirm():
    ctrl = _ctrl(
        barge_in_pause_ms=64,
        barge_in_confirm_ms=96,
        barge_in_energy_floor=0.012,
        barge_in_energy_ratio=1.8,
    )
    half = WINDOW // 2
    speech = (np.random.randn(half).astype(np.float32) * 0.05).astype(np.float32)
    assert ctrl._watch_barge_in(speech, SessionState.SPEAKING) is False
    assert len(ctrl._barge_vad_buf) == half
    fired = _feed(ctrl, speech, 8)
    assert fired is True
    assert ctrl._barge_keep_asr is True
