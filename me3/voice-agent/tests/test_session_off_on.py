"""OFF/ON stale-speech gate (uses real engines when models present; skips audio HW)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from me3.session.controller import SessionController, SessionState

ROOT = Path(__file__).resolve().parents[1]
HAS_MODELS = (ROOT / "models" / "MANIFEST.json").is_file()


@pytest.mark.skipif(not HAS_MODELS, reason="speech models not fetched")
def test_off_flushes_and_blocks_stale_replay():
    ctrl = SessionController()
    # Avoid opening real mic/speaker in unit test — inject fakes after probe pieces
    ctrl.probe_readiness()
    assert ctrl.tts is not None

    fake_play = MagicMock()
    fake_play.flush.return_value = 7
    fake_play.play.return_value = True
    fake_play.wait_drain.return_value = True
    fake_play.queue_seconds.return_value = 0.0
    ctrl.playback = fake_play

    fake_cap = MagicMock()
    fake_cap.drain = MagicMock()
    ctrl.capture = fake_cap

    # Don't start real pump/devices
    ctrl.enabled = False
    ctrl.state = SessionState.OFF

    snap = ctrl.set_enabled(True)
    assert snap.enabled is True
    assert snap.state == SessionState.LISTENING

    # Simulate speaking path enqueue then OFF before finish
    ctrl.state = SessionState.SPEAKING
    gen_before = ctrl.utterance_generation
    samples = np.random.randn(22050).astype(np.float32) * 0.1
    # Direct play as probe would
    ctrl.playback.play(samples, 22050, generation=gen_before)

    snap_off = ctrl.set_enabled(False)
    assert snap_off.enabled is False
    assert snap_off.state == SessionState.OFF
    fake_play.flush.assert_called()
    assert ctrl.utterance_generation == 7  # from flush return

    # Controller generation advanced on OFF; set_enabled(True) must not auto-replay
    assert ctrl.utterance_generation != gen_before
    play_calls_after_off = fake_play.play.call_count

    snap_on = ctrl.set_enabled(True)
    assert snap_on.enabled is True
    assert fake_play.play.call_count == play_calls_after_off

