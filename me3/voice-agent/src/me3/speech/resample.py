"""Documented resample boundaries for capture (→16 kHz) and playback (Piper → device)."""

from __future__ import annotations

import numpy as np


def resample_linear(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Mono float32 linear interpolation. Length ≈ len(audio) * dst_rate / src_rate."""
    if src_rate <= 0 or dst_rate <= 0:
        raise ValueError("sample rates must be positive")
    if audio.ndim != 1:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    else:
        audio = np.asarray(audio, dtype=np.float32)
    if src_rate == dst_rate or len(audio) == 0:
        return audio.copy()
    n_dst = max(1, int(round(len(audio) * float(dst_rate) / float(src_rate))))
    x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_dst, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def to_mono(frames: np.ndarray) -> np.ndarray:
    """Collapse multi-channel float frames to mono."""
    arr = np.asarray(frames, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        return arr.mean(axis=1).astype(np.float32)
    raise ValueError(f"unexpected audio shape {arr.shape}")
