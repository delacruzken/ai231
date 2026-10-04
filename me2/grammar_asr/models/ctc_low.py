"""Residual 1-D CNN + CTC encoder with LOW (2×) temporal downsampling."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..schema import NUM_VARIATIONS  # noqa: F401 — keep schema import side-effect free
from ..text import NUM_CLASSES


class ResBlock(nn.Module):
    def __init__(self, ch: int, kernel: int = 5, stride: int = 1, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(ch, ch, kernel, stride=stride, padding=kernel // 2)
        self.bn1 = nn.BatchNorm1d(ch)
        self.conv2 = nn.Conv1d(ch, ch, kernel, stride=1, padding=kernel // 2)
        self.bn2 = nn.BatchNorm1d(ch)
        self.stride = stride
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        if self.stride > 1 and x.shape[2] != h.shape[2]:
            x = x[:, :, :h.shape[2]]
        return self.drop(F.relu(h + x))


class ResCTCEncoderLow(nn.Module):
    """Production CTC baseline (~1.93M params at hidden=192, blocks=5)."""

    def __init__(self, n_mels: int = 40, hidden: int = 192,
                 num_blocks: int = 5, num_classes: int = NUM_CLASSES,
                 dropout: float = 0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_mels, hidden, 5, stride=2, padding=2),
            nn.BatchNorm1d(hidden), nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[
            ResBlock(hidden, kernel=5, stride=1, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )
        self.hidden = hidden
        self.model_kind = "ctc"

    def forward(self, feat: torch.Tensor):
        """feat (B, T, n_mels) -> CTC logits (B, T', C)."""
        x = feat.transpose(1, 2)
        x = self.stem(x)
        x = self.blocks(x)
        x = x.transpose(1, 2)
        return {"ctc_logits": self.head(x)}

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def post_len_low(T: int) -> int:
    return max(1, (T - 5) // 2 + 1)
