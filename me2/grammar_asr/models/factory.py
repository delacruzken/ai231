"""Model factory for config-driven experiments."""
from __future__ import annotations

from typing import Any, Dict

from .ctc_low import ResCTCEncoderLow
from .intent_crnn import IntentCRNN
from .intent_vcm import build_conformer, build_matchbox
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
    if name in {"intent", "intent_crnn", "crnn"}:
        return IntentCRNN(
            gru_hidden=int(kwargs.get("gru_hidden", 96)),
            gru_layers=int(kwargs.get("gru_layers", 2)),
            dropout=float(kwargs.get("dropout", 0.15)),
        )
    if name in {"matchbox", "intent_matchbox"}:
        return build_matchbox(
            width=int(kwargs.get("width", 104)),
            blocks=int(kwargs.get("blocks", 4)),
            repeat=int(kwargs.get("repeat", 2)),
            n_mels=int(kwargs.get("n_mels", 40)),
            dropout=float(kwargs.get("dropout", 0.1)),
            n_heads=int(kwargs.get("n_heads", 4)),
        )
    if name in {"conformer", "intent_conformer", "tiny_conformer"}:
        return build_conformer(
            d=int(kwargs.get("d", 64)),
            layers=int(kwargs.get("layers", 4)),
            heads=int(kwargs.get("heads", 4)),
            ff_mult=int(kwargs.get("ff_mult", 2)),
            n_mels=int(kwargs.get("n_mels", 40)),
            stride=int(kwargs.get("stride", 4)),
            dropout=float(kwargs.get("dropout", 0.1)),
            n_heads=int(kwargs.get("n_heads", 4)),
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
    _ = x  # keep signature parity with older helper
    return {
        "params": count_params(model),
        "approx_input_frames": T,
        "n_mels": n_mels,
    }


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
