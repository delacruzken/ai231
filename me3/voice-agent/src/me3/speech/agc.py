"""Soft automatic gain control for ASR input.

Boosts quiet mic levels toward a target RMS without hard-clipping peaks.
Keeps silence near-zero (does not amplify noise floors below min_rms).
"""

from __future__ import annotations

import numpy as np


def apply_agc(
    samples: np.ndarray,
    *,
    target_rms: float = 0.10,
    max_gain: float = 12.0,
    min_rms: float = 0.001,
    ema: float | None = None,
    ema_alpha: float = 0.15,
) -> tuple[np.ndarray, float]:
    """Return (normalized float32 mono, updated ema rms used for gain).

    ``ema`` is an optional previous RMS estimate for smooth gain across chunks.
    """
    if samples.dtype != np.float32:
        samples = samples.astype(np.float32, copy=False)
    if samples.size == 0:
        return samples, float(ema or 0.0)

    chunk_rms = float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))
    if ema is None:
        level = chunk_rms
    else:
        level = (1.0 - ema_alpha) * float(ema) + ema_alpha * chunk_rms

    if level < min_rms:
        return samples, level

    gain = min(max_gain, target_rms / max(level, 1e-8))
    if abs(gain - 1.0) < 1e-3:
        return samples, level

    out = samples * np.float32(gain)
    # Soft peak limit — avoid hard wrap if a sudden loud frame arrives.
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0.95:
        out = out * np.float32(0.95 / peak)
    return out, level
