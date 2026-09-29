"""
Tiny 1-D CNN + CTC encoder for character-level ASR.

Architecture (PocketSphinx-inspired, grammar-constrained at decode time):
  log-mel (T, 40)
    -> Conv1d blocks (stride 2) x 4   -> temporal downsampling 16x
    -> Linear projection to 28 classes (27 chars + CTC blank)

Kept small (~1-3M params) so it runs real-time on an RPi 5 and exports to a
sub-1MB ONNX int8 model. Temporal downsampling 16x means a 2 s utterance
(~195 frames) becomes ~12 steps — very cheap to decode.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .grammar import NUM_CLASSES  # 28 (27 chars + blank)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 5, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, stride=stride,
                              padding=kernel // 2)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        # x: (B, C, T)
        return F.dropout(F.relu(self.bn(self.conv(x))), p=0.1, training=self.training)


class CTCEncoder(nn.Module):
    def __init__(self, n_mels: int = 40, hidden: int = 128,
                 num_layers: int = 4, num_classes: int = NUM_CLASSES,
                 downsample: int = 2):
        super().__init__()
        layers = []
        ch = n_mels
        for i in range(num_layers):
            out_ch = hidden
            stride = downsample if i < num_layers - 1 else 1
            layers.append(ConvBlock(ch, out_ch, kernel=5, stride=stride))
            ch = out_ch
        self.features = nn.Sequential(*layers)
        self.proj = nn.Linear(hidden, num_classes)
        self.hidden = hidden
        self.downsample_factor = downsample ** (num_layers - 1)

    def forward(self, feat: torch.Tensor):
        """feat: (B, T, n_mels) -> logits (B, T', num_classes)."""
        x = feat.transpose(1, 2)                 # (B, n_mels, T)
        x = self.features(x)                     # (B, hidden, T')
        x = x.transpose(1, 2)                    # (B, T', hidden)
        logits = self.proj(x)                    # (B, T', num_classes)
        return logits

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def ctc_loss(logits: torch.Tensor, targets: torch.Tensor,
             input_lengths: torch.Tensor, target_lengths: torch.Tensor,
             blank: int = NUM_CLASSES - 1) -> torch.Tensor:
    """Standard CTC loss. logits (B,T',C) log-probs, targets (B,S) concatenated."""
    log_probs = F.log_softmax(logits, dim=-1)
    # torch expects (T', B, C)
    log_probs = log_probs.permute(1, 0, 2)
    return F.ctc_loss(
        log_probs, targets, input_lengths, target_lengths,
        blank=blank, zero_infinity=True,
    )


def greedy_decode(logits: torch.Tensor, blank: int = NUM_CLASSES - 1) -> list:
    """Greedy CTC decode: argmax per frame, collapse repeats + blanks."""
    import numpy as np
    idx = logits.argmax(dim=-1).cpu().numpy()  # (B, T')
    out = []
    for row in idx:
        chars = []
        prev = None
        for c in row:
            if c == blank:
                prev = blank
                continue
            if c != prev:
                chars.append(c)
            prev = c
        out.append(chars)
    return out


if __name__ == "__main__":
    m = CTCEncoder()
    print(f"Params: {m.count_params():,} ({m.count_params()*4/1e6:.2f} MB fp32)")
    x = torch.randn(2, 195, 40)
    y = m(x)
    print(f"in {tuple(x.shape)} -> out {tuple(y.shape)} (T'={y.shape[1]})")
    # quick loss smoke test (pad targets to equal length; input_lengths = T')
    Tp = y.shape[1]
    tgt = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 0, 0]])
    il = torch.full((2,), Tp); tl = torch.tensor([5, 3])
    loss = ctc_loss(y, tgt, il, tl)
    print(f"CTC loss (random init): {loss.item():.3f}")
