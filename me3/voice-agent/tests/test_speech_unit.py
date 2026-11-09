"""Unit tests that do not require model weights (except optional integration)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models" / "MANIFEST.json"
HAS_MODELS = MODELS.is_file()


def test_vad_window_constant():
    from me3.speech.vad import WINDOW

    assert WINDOW == 512


def test_asr_queue_bound_without_engine():
    # Bounded queue behavior is covered via StreamingAsrEngine.queue_max default
    from me3.config import load_config

    cfg = load_config(ROOT / "config" / "settings.example.yaml")
    assert cfg.speech.asr_queue_max == 64
    assert cfg.speech.tts_queue_max == 32


@pytest.mark.skipif(not HAS_MODELS, reason="speech models not fetched")
def test_vad_prob_on_silence():
    from me3.speech.vad import SileroVAD, WINDOW

    vad = SileroVAD()
    silence = np.zeros(WINDOW, dtype=np.float32)
    p = vad.prob(silence)
    assert 0.0 <= p <= 1.0
    assert p < 0.5


@pytest.mark.skipif(not HAS_MODELS, reason="speech models not fetched")
def test_tts_synthesize_nonzero():
    from me3.speech.tts_engine import PiperTtsEngine

    eng = PiperTtsEngine()
    audio, rate = eng.synthesize_to_array("Hello.")
    assert rate == 22050
    assert len(audio) > 1000
