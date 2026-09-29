"""
Residual 1-D CNN + CTC encoder with LOW temporal downsampling.

Key design choice: only the stem downsamples (stride 2). All residual blocks
use stride 1, so T' = floor(T/2). For a 2 s utterance (T≈195) that gives
T'≈98 frames — comfortably above the longest command (46 chars), so NO
grammar word is structurally impossible under the CTC constraint T' >= L.

(The earlier 16×-downsample variant made long commands unalignable, which is
why the decoder collapsed to short words like "time"/"stop".)

Still compact (~1.5M params) -> real-time on RPi 5, small ONNX export.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .grammar import NUM_CLASSES


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
    def __init__(self, n_mels: int = 40, hidden: int = 192,
                 num_blocks: int = 5, num_classes: int = NUM_CLASSES,
                 dropout: float = 0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_mels, hidden, 5, stride=2, padding=2),
            nn.BatchNorm1d(hidden), nn.ReLU(),
        )
        blocks = []
        for i in range(num_blocks):
            # all stride 1 -> preserve temporal resolution
            blocks.append(ResBlock(hidden, kernel=5, stride=1, dropout=dropout))
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )
        self.hidden = hidden

    def forward(self, feat: torch.Tensor):
        """feat: (B, T, n_mels) -> logits (B, T', num_classes), T' ~= T//2."""
        x = feat.transpose(1, 2)
        x = self.stem(x)
        x = self.blocks(x)
        x = x.transpose(1, 2)
        return self.head(x)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def post_len_low(T: int) -> int:
    """Frames after the low-downsample CNN (only stem stride 2)."""
    return max(1, (T - 5) // 2 + 1)


if __name__ == "__main__":
    m = ResCTCEncoderLow()
    print(f"Params: {m.count_params():,} ({m.count_params()*4/1e6:.2f} MB fp32)")
    for T in [195, 200, 100, 60, 46]:
        x = torch.randn(2, T, 40)
        y = m(x)
        print(f"  T_in={T:4d} -> T_out={y.shape[1]}  (formula={post_len_low(T)})")
    print(f"\nLongest command = 46 chars. Need T' >= 46.")
    print(f"T=90 (0.56s) -> T'={post_len_low(90)}  {'OK' if post_len_low(90)>=46 else 'TOO SHORT'}")
