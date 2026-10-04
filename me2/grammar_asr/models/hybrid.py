"""Hybrid compact encoder: CTC + intent/OOS + slot heads.

Shares a low-downsample residual trunk with the CTC baseline, then branches:
  - character CTC head (grammar decoding)
  - pooled intent/OOS head
  - six slot heads
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..schema import NUM_INTENT_OOS, SLOT_INTENTS, SLOT_VALUES
from ..text import NUM_CLASSES
from .ctc_low import ResBlock


class HybridVCM(nn.Module):
    def __init__(self, n_mels: int = 40, hidden: int = 128,
                 num_blocks: int = 4, dropout: float = 0.15):
        super().__init__()
        self.model_kind = "hybrid"
        self.hidden = hidden
        self.stem = nn.Sequential(
            nn.Conv1d(n_mels, hidden, 5, stride=2, padding=2),
            nn.BatchNorm1d(hidden), nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[
            ResBlock(hidden, kernel=5, stride=1, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.ctc_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, NUM_CLASSES),
        )
        self.attn = nn.Linear(hidden, 1)
        self.intent_head = nn.Linear(hidden, NUM_INTENT_OOS)
        self.slot_heads = nn.ModuleDict({
            intent: nn.Linear(hidden, len(SLOT_VALUES[intent]))
            for intent in SLOT_INTENTS
        })
        self.drop = nn.Dropout(dropout)

    def forward(self, feat: torch.Tensor, lengths=None):
        x = feat.transpose(1, 2)
        x = self.stem(x)
        x = self.blocks(x)
        x = x.transpose(1, 2)                 # (B, T', H)
        ctc_logits = self.ctc_head(x)

        # Attention pool over frames
        scores = self.attn(x).squeeze(-1)     # (B, T')
        if lengths is not None:
            t_out = x.shape[1]
            fl = torch.clamp((lengths.float() / 2.0).ceil().long(), max=t_out)
            mask = torch.arange(t_out, device=x.device)[None, :] < fl[:, None]
            scores = scores.masked_fill(~mask, -1e9)
        w = torch.softmax(scores, dim=-1)
        pooled = torch.einsum("bt,bth->bh", w, x)
        pooled = self.drop(pooled)
        return {
            "ctc_logits": ctc_logits,
            "intent_logits": self.intent_head(pooled),
            "slot_logits": {k: head(pooled) for k, head in self.slot_heads.items()},
            "embedding": pooled,
        }

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
