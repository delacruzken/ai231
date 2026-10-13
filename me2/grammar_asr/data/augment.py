"""Light SpecAugment on log-mel features (time + frequency masks)."""
from __future__ import annotations

from typing import Optional, Union

import numpy as np

PRESETS = {
    "none": None,
    "light": dict(
        n_time_masks=1, n_freq_masks=1, max_time_width=20, max_freq_width=5, p=0.35,
    ),
    "heavy": dict(
        n_time_masks=2, n_freq_masks=2, max_time_width=40, max_freq_width=8, p=0.8,
    ),
}


def resolve_aug(augment: Union[bool, str, None]) -> Optional[dict]:
    if augment is True or str(augment).lower() in {"1", "true", "yes", "heavy"}:
        return dict(PRESETS["heavy"])
    if str(augment).lower() == "light":
        return dict(PRESETS["light"])
    return None


def spec_augment(
    feat: np.ndarray,
    *,
    n_time_masks: int = 2,
    n_freq_masks: int = 2,
    max_time_width: int = 40,
    max_freq_width: int = 8,
    p: float = 0.8,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply SpecAugment-style masks. feat shape (T, F)."""
    if p <= 0 or (rng := rng or np.random.default_rng()).random() > p:
        return feat
    out = feat.copy()
    t, f = out.shape
    for _ in range(n_time_masks):
        w = int(rng.integers(1, max(2, min(max_time_width, t // 2) + 1)))
        t0 = int(rng.integers(0, max(1, t - w + 1)))
        out[t0:t0 + w, :] = 0.0
    for _ in range(n_freq_masks):
        w = int(rng.integers(1, max(2, min(max_freq_width, f // 2) + 1)))
        f0 = int(rng.integers(0, max(1, f - w + 1)))
        out[:, f0:f0 + w] = 0.0
    return out
