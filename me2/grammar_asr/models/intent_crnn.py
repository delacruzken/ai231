"""Depthwise-CRNN with intent + slot heads (from-scratch).

Compact DS-conv blocks, BiGRU, attention pooling, 20-way intent/OOS head,
and six 3-way slot heads. Sized for on-device intent classification.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..schema import NUM_INTENT_OOS, SLOT_INTENTS, SLOT_VALUES


class DSConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dropout: float = 0.1):
        super().__init__()
        self.depthwise = nn.Conv1d(in_ch, in_ch, 5, stride=stride, padding=2,
                                  groups=in_ch)
        self.pointwise = nn.Conv1d(in_ch, out_ch, 1)
        self.bn = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Identity() if in_ch == out_ch and stride == 1 else nn.Conv1d(
            in_ch, out_ch, 1, stride=stride)

    def forward(self, x):
        h = self.pointwise(self.depthwise(x))
        h = self.drop(F.relu(self.bn(h)))
        return h + self.proj(x)


class AttentionPool(nn.Module):
    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.scale = (dim // n_heads) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.cls, std=0.02)

    def forward(self, x, mask=None):
        # x: (B, T, D)
        B, T, D = x.shape
        cls = self.cls.expand(B, -1, -1)
        q = self.q(cls).view(B, 1, self.n_heads, D // self.n_heads).transpose(1, 2)
        kv = self.kv(x).view(B, T, 2, self.n_heads, D // self.n_heads)
        k, v = kv.unbind(2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            # -1e9 overflows float16 under AMP; use the dtype's finite floor.
            attn = attn.masked_fill(
                ~mask[:, None, None, :], torch.finfo(attn.dtype).min
            )
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, 1, D)
        return self.proj(out).squeeze(1)


class IntentCRNN(nn.Module):
    def __init__(self, n_mels: int = 40, conv_channels=(64, 96, 128, 128),
                 gru_hidden: int = 96, gru_layers: int = 2,
                 n_heads: int = 4, dropout: float = 0.15):
        super().__init__()
        self.model_kind = "intent"
        chs = [n_mels, *conv_channels]
        blocks = []
        for i in range(len(conv_channels)):
            # Mild temporal reduction: strides 2,1,2,1 -> 4× total (still OK for pooling)
            stride = 2 if i in (0, 2) else 1
            blocks.append(DSConvBlock(chs[i], chs[i + 1], stride=stride,
                                      dropout=dropout))
        self.conv = nn.Sequential(*blocks)
        self.gru = nn.GRU(
            input_size=conv_channels[-1],
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        dim = gru_hidden * 2
        self.pool = AttentionPool(dim, n_heads=n_heads)
        self.drop = nn.Dropout(dropout)
        self.intent_head = nn.Linear(dim, NUM_INTENT_OOS)
        self.slot_heads = nn.ModuleDict({
            intent: nn.Linear(dim, len(SLOT_VALUES[intent]))
            for intent in SLOT_INTENTS
        })

    def encode(self, feat: torch.Tensor, lengths=None):
        x = feat.transpose(1, 2)          # (B, C, T)
        x = self.conv(x)
        x = x.transpose(1, 2)             # (B, T', C)
        x, _ = self.gru(x)
        # Optional length mask after 4× downsample
        mask = None
        if lengths is not None:
            t_out = x.shape[1]
            # approximate frame lengths
            fl = torch.clamp((lengths.float() / 4.0).ceil().long(), max=t_out)
            mask = (torch.arange(t_out, device=feat.device)[None, :] < fl[:, None])
        return self.pool(x, mask=mask)

    def forward(self, feat: torch.Tensor, lengths=None):
        h = self.drop(self.encode(feat, lengths=lengths))
        slot_logits = {k: head(h) for k, head in self.slot_heads.items()}
        return {
            "intent_logits": self.intent_head(h),
            "slot_logits": slot_logits,
            "embedding": h,
        }

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
