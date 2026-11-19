"""AGC + ASR config unit tests (no live mic)."""

from __future__ import annotations

import numpy as np

from me3.config import AppConfig
from me3.speech.agc import apply_agc


def test_agc_boosts_quiet_speech():
    quiet = (np.random.randn(1600).astype(np.float32) * 0.01).astype(np.float32)
    out, level = apply_agc(quiet, target_rms=0.10, max_gain=12.0, min_rms=0.001)
    assert level > 0.001
    out_rms = float(np.sqrt(np.mean(np.square(out))))
    quiet_rms = float(np.sqrt(np.mean(np.square(quiet))))
    assert out_rms > quiet_rms * 2.0
    assert float(np.max(np.abs(out))) <= 0.95 + 1e-5


def test_agc_skips_near_silence():
    silence = np.zeros(1600, dtype=np.float32)
    out, level = apply_agc(silence, target_rms=0.10, max_gain=12.0, min_rms=0.001)
    assert level < 0.001
    assert np.allclose(out, silence)


def test_asr_defaults_use_kroko_and_beam():
    cfg = AppConfig()
    assert "kroko" in cfg.models.asr
    assert cfg.models.asr_model_type == ""
    assert cfg.speech.asr_decoding_method == "modified_beam_search"
    assert cfg.speech.asr_max_active_paths == 4
    assert cfg.audio.agc_enabled is True
