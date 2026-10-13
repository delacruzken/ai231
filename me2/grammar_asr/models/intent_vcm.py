"""MatchboxNet + TinyConformer intent classifiers (adapted from airimonda/vcm-me2).

Same interface as IntentCRNN: log-mel (B, T, n_mels) in → intent/slot logits.
Sized for ~300k params (tier M) on 40-band mels used by this package.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..schema import NUM_INTENT_OOS, SLOT_INTENTS, SLOT_VALUES
from .intent_crnn import AttentionPool


# ---------------------------------------------------------------- MatchboxNet
class MBSub(nn.Module):
    def __init__(self, cin, cout, k, relu=True):
        super().__init__()
        self.dw = nn.Conv1d(cin, cin, k, 1, k // 2, groups=cin, bias=False)
        self.pw = nn.Conv1d(cin, cout, 1, bias=False)
        self.bn = nn.BatchNorm1d(cout)
        self.relu = relu

    def forward(self, x):
        y = self.bn(self.pw(self.dw(x)))
        return F.relu(y) if self.relu else y


class MBBlock(nn.Module):
    def __init__(self, cin, cout, k, repeat):
        super().__init__()
        subs = []
        for i in range(repeat):
            subs.append(MBSub(cin if i == 0 else cout, cout, k, relu=i < repeat - 1))
        self.subs = nn.Sequential(*subs)
        self.res = nn.Sequential(
            nn.Conv1d(cin, cout, 1, bias=False), nn.BatchNorm1d(cout)
        )

    def forward(self, x):
        return F.relu(self.subs(x) + self.res(x))


class MatchboxBackbone(nn.Module):
    def __init__(self, width=104, blocks=4, repeat=2, n_mels=40):
        super().__init__()
        self.pro = nn.Sequential(
            nn.Conv1d(n_mels, width, 11, 2, 5, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(),
        )
        ks = [13, 15, 17, 19, 21]
        layers = []
        cin = width
        for i in range(blocks):
            layers.append(MBBlock(cin, width, ks[i % len(ks)], repeat))
            cin = width
        self.blocks = nn.Sequential(*layers)
        self.epi = nn.Sequential(
            nn.Conv1d(width, width, 29, 1, 28, dilation=2, groups=width, bias=False),
            nn.Conv1d(width, width * 2, 1, bias=False),
            nn.BatchNorm1d(width * 2),
            nn.ReLU(),
        )
        self.out_dim = width * 2

    def forward(self, x):  # (B, C, T) → (B, d, T')
        return self.epi(self.blocks(self.pro(x)))


# ---------------------------------------------------------------- Tiny Conformer
class MHSA(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)

    def forward(self, x):  # (B, T, d)
        b, t, d = x.shape
        q, k, v = self.qkv(x).reshape(b, t, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        att = torch.softmax(
            torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dk), dim=-1
        )
        return self.o(torch.matmul(att, v).transpose(1, 2).reshape(b, t, d))


class ConformerBlock(nn.Module):
    def __init__(self, d, heads, ff_mult=2, k=9):
        super().__init__()
        self.ff1 = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d * ff_mult), nn.SiLU(),
            nn.Linear(d * ff_mult, d),
        )
        self.ln_att = nn.LayerNorm(d)
        self.att = MHSA(d, heads)
        self.ln_conv = nn.LayerNorm(d)
        self.pw1 = nn.Conv1d(d, 2 * d, 1)
        self.dw = nn.Conv1d(d, d, k, 1, k // 2, groups=d)
        self.bn = nn.BatchNorm1d(d)
        self.pw2 = nn.Conv1d(d, d, 1)
        self.ff2 = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d * ff_mult), nn.SiLU(),
            nn.Linear(d * ff_mult, d),
        )
        self.ln_out = nn.LayerNorm(d)

    def forward(self, x):  # (B, T, d)
        x = x + 0.5 * self.ff1(x)
        x = x + self.att(self.ln_att(x))
        c = self.ln_conv(x).transpose(1, 2)
        c = F.glu(self.pw1(c), dim=1)
        c = self.pw2(F.silu(self.bn(self.dw(c))))
        x = x + c.transpose(1, 2)
        x = x + 0.5 * self.ff2(x)
        return self.ln_out(x)


class ConformerBackbone(nn.Module):
    def __init__(self, d=64, layers=4, heads=4, ff_mult=2, n_mels=40, stride=4):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv1d(n_mels, d, 2 * stride, stride, stride // 2, bias=False),
            nn.BatchNorm1d(d),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            *[ConformerBlock(d, heads, ff_mult) for _ in range(layers)]
        )
        self.out_dim = d

    def forward(self, x):  # (B, C, T) → (B, d, T')
        x = self.embed(x).transpose(1, 2)
        return self.blocks(x).transpose(1, 2)


# ---------------------------------------------------------------- Intent wrapper
class IntentVCM(nn.Module):
    """Backbone + attention pool + intent/slot heads (IntentCRNN-compatible API)."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        n_heads: int = 4,
        dropout: float = 0.1,
        arch_name: str = "vcm",
    ):
        super().__init__()
        self.model_kind = "intent"
        self.arch_name = arch_name
        self.backbone = backbone
        dim = backbone.out_dim
        self.pool = AttentionPool(dim, n_heads=min(n_heads, max(1, dim // 16)))
        self.drop = nn.Dropout(dropout)
        self.intent_head = nn.Linear(dim, NUM_INTENT_OOS)
        self.slot_heads = nn.ModuleDict({
            intent: nn.Linear(dim, len(SLOT_VALUES[intent]))
            for intent in SLOT_INTENTS
        })

    def encode(self, feat: torch.Tensor, lengths=None):
        # feat: (B, T, C) → backbone wants (B, C, T)
        x = feat.transpose(1, 2)
        x = self.backbone(x)          # (B, d, T')
        x = x.transpose(1, 2)         # (B, T', d)
        mask = None
        if lengths is not None:
            t_out = x.shape[1]
            # Approximate: Matchbox stride~2, Conformer stride~4
            ratio = max(1.0, float(feat.shape[1]) / max(t_out, 1))
            fl = torch.clamp((lengths.float() / ratio).ceil().long(), max=t_out)
            mask = torch.arange(t_out, device=feat.device)[None, :] < fl[:, None]
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


def build_matchbox(
    *,
    width: int = 104,
    blocks: int = 4,
    repeat: int = 2,
    n_mels: int = 40,
    dropout: float = 0.1,
    n_heads: int = 4,
) -> IntentVCM:
    return IntentVCM(
        MatchboxBackbone(width=width, blocks=blocks, repeat=repeat, n_mels=n_mels),
        n_heads=n_heads,
        dropout=dropout,
        arch_name="matchbox",
    )


def build_conformer(
    *,
    d: int = 64,
    layers: int = 4,
    heads: int = 4,
    ff_mult: int = 2,
    n_mels: int = 40,
    stride: int = 4,
    dropout: float = 0.1,
    n_heads: int = 4,
) -> IntentVCM:
    return IntentVCM(
        ConformerBackbone(
            d=d, layers=layers, heads=heads, ff_mult=ff_mult,
            n_mels=n_mels, stride=stride,
        ),
        n_heads=n_heads,
        dropout=dropout,
        arch_name="conformer",
    )
