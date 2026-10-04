"""Tiny from-scratch wake-word detector (MicroWakeNet-style)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MicroDSConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.pw(self.dw(x))))


class MicroWakeNet(nn.Module):
    """Binary wake / not-wake classifier on log-mel (B, T, 40)."""

    def __init__(self, n_mels: int = 40, channels=(8, 16, 32), dropout: float = 0.1):
        super().__init__()
        self.model_kind = "wake"
        layers = []
        in_ch = 1
        for i, ch in enumerate(channels):
            stride = 2 if i < 2 else 1
            layers.append(MicroDSConv(in_ch, ch, stride=stride))
            in_ch = ch
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(channels[-1], 2)

    def forward(self, feat: torch.Tensor):
        # feat: (B, T, n_mels) -> (B, 1, n_mels, T)
        x = feat.transpose(1, 2).unsqueeze(1)
        x = self.features(x)
        x = self.pool(x).flatten(1)
        logits = self.head(self.drop(x))
        return {"wake_logits": logits}

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
