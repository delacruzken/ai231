"""
Deeper residual 1-D CNN + CTC encoder (better capacity than the 4-layer base).

Adds residual connections (skip when in/out channels match) and a final
linear->BN->ReLU->Linear head. Still compact (~1-3M params) so it stays
real-time on an RPi 5 and exports to a small ONNX model.
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
        # x: (B, C, T)
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        if self.stride > 1 and x.shape[2] != h.shape[2]:
            # crop residual to match conv-strided length
            x = x[:, :, :h.shape[2]]
        return self.drop(F.relu(h + x))


class ResCTCEncoder(nn.Module):
    def __init__(self, n_mels: int = 40, hidden: int = 192,
                 num_blocks: int = 4, num_classes: int = NUM_CLASSES,
                 dropout: float = 0.1):
        super().__init__()
        # initial conv: n_mels -> hidden, stride 2
        self.stem = nn.Sequential(
            nn.Conv1d(n_mels, hidden, 5, stride=2, padding=2),
            nn.BatchNorm1d(hidden), nn.ReLU(),
        )
        blocks = []
        for i in range(num_blocks):
            stride = 2 if i < num_blocks - 1 else 1
            blocks.append(ResBlock(hidden, kernel=5, stride=stride, dropout=dropout))
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )
        self.hidden = hidden
        self.downsample_factor = 2 ** num_blocks  # stem(2) * (num_blocks-1) stride-2

    def forward(self, feat: torch.Tensor):
        """feat: (B, T, n_mels) -> logits (B, T', num_classes)."""
        x = feat.transpose(1, 2)                 # (B, n_mels, T)
        x = self.stem(x)
        x = self.blocks(x)
        x = x.transpose(1, 2)                    # (B, T', hidden)
        return self.head(x)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def post_len_res(T: int, num_blocks: int = 4) -> int:
    """Approx frames after the residual CNN (stem stride2 + (L-1) stride-2)."""
    t = T
    t = (t - 5) // 2 + 1          # stem
    for _ in range(num_blocks - 1):
        t = (t - 5) // 2 + 1
    return max(1, t)


if __name__ == "__main__":
    m = ResCTCEncoder()
    print(f"Params: {m.count_params():,} ({m.count_params()*4/1e6:.2f} MB fp32)")
    for T in [195, 200, 100, 50]:
        x = torch.randn(1, T, 40)
        y = m(x)
        print(f"  T_in={T:4d} -> T_out={y.shape[1]}  (formula={post_len_res(T)})")
