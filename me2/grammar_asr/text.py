"""Character alphabet + CTC helpers shared by CTC/hybrid models."""
from __future__ import annotations

from typing import Dict, List

from .schema import normalize_text

BLANK = "<blk>"
SPACE = " "
CHARS = [chr(c) for c in range(ord("a"), ord("z") + 1)]
ALPHABET = CHARS + [SPACE]
NUM_CLASSES = len(ALPHABET) + 1  # 28

char_to_idx: Dict[str, int] = {c: i for i, c in enumerate(ALPHABET)}
char_to_idx[BLANK] = NUM_CLASSES - 1
idx_to_char: List[str] = ALPHABET + [BLANK]


def encode_transcript(text: str) -> List[int]:
    norm = normalize_text(text)
    return [char_to_idx[c] for c in norm if c in char_to_idx]


def greedy_decode_ids(logits) -> List[List[int]]:
    """Greedy CTC collapse. logits: (B, T, C) tensor."""
    import torch
    ids = logits.argmax(dim=-1)  # (B, T)
    blank = NUM_CLASSES - 1
    out = []
    for seq in ids.tolist():
        prev = None
        chars = []
        for i in seq:
            if i == blank:
                prev = None
                continue
            if i != prev:
                chars.append(i)
            prev = i
        out.append(chars)
    return out
