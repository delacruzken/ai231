"""Model factory for config-driven experiments."""
from __future__ import annotations

from typing import Any, Dict

from .ctc_low import ResCTCEncoderLow
from .kiwi_crnn import KiwiCRNN
from .hybrid import HybridVCM
from .wake import MicroWakeNet


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def build_model(name: str, **kwargs) -> Any:
    name = name.lower().strip()
    if name in {"ctc", "ctc_low", "low"}:
        return ResCTCEncoderLow(
            hidden=int(kwargs.get("hidden", 192)),
            num_blocks=int(kwargs.get("num_blocks", 5)),
            dropout=float(kwargs.get("dropout", 0.1)),
        )
    if name in {"kiwi", "kiwi_crnn", "crnn"}:
        return KiwiCRNN(
            gru_hidden=int(kwargs.get("gru_hidden", 96)),
            gru_layers=int(kwargs.get("gru_layers", 2)),
            dropout=float(kwargs.get("dropout", 0.15)),
        )
    if name in {"hybrid"}:
        return HybridVCM(
            hidden=int(kwargs.get("hidden", 128)),
            num_blocks=int(kwargs.get("num_blocks", 4)),
            dropout=float(kwargs.get("dropout", 0.15)),
        )
    if name in {"wake", "microwakenet"}:
        return MicroWakeNet(dropout=float(kwargs.get("dropout", 0.1)))
    raise ValueError(f"Unknown model: {name}")


def model_flops_estimate(name: str, seconds: float = 3.0, n_mels: int = 40) -> Dict:
    """Rough MAC estimate for reporting (not exact FLOPs)."""
    import torch
    model = build_model(name)
    T = int(seconds * 100)  # 10 ms hop
    x = torch.randn(1, T, n_mels)
    # Count params as a cheap proxy; exact FLOPs optional.
    return {
        "params": count_params(model),
        "approx_input_frames": T,
        "n_mels": n_mels,
    }
