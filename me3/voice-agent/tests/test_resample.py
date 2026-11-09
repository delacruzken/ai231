"""Resample boundary unit tests (no hardware)."""

from __future__ import annotations

import numpy as np

from me3.speech.resample import resample_linear, to_mono


def test_resample_length_16k_to_48k():
    src = np.zeros(16000, dtype=np.float32)
    src[::100] = 0.5
    out = resample_linear(src, 16000, 48000)
    assert abs(len(out) - 48000) <= 1


def test_resample_identity():
    src = np.linspace(-1, 1, 1000, dtype=np.float32)
    out = resample_linear(src, 16000, 16000)
    assert np.allclose(src, out)


def test_to_mono_stereo():
    stereo = np.stack([np.ones(8), np.zeros(8)], axis=1).astype(np.float32)
    mono = to_mono(stereo)
    assert mono.shape == (8,)
    assert np.allclose(mono, 0.5)
