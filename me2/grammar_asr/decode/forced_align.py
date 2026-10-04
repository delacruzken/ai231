"""CTC forced-alignment scoring over the grammar."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from ..text import char_to_idx, NUM_CLASSES
from .grammar_trie import GrammarTrie


@dataclass
class WordResult:
    intent: str
    transcript: str
    normalized: str
    variation_id: int
    slot: str
    score: float


def forced_align_score(log_probs: np.ndarray, word_chars: List[int],
                       blank_idx: int, length_norm: bool = True,
                       char_reward: float = 0.15) -> float:
    T, _ = log_probs.shape
    L = len(word_chars)
    NEG = -1e9
    best = [NEG] * (L + 1)
    best[0] = 0.0
    for t in range(T):
        lp = log_probs[t]
        new_best = best[:]
        for l in range(L):
            if best[l] <= NEG / 2:
                continue
            cand = best[l] + lp[word_chars[l]]
            if cand > new_best[l + 1]:
                new_best[l + 1] = cand
        best = new_best
    total = best[L]
    if total <= NEG / 2:
        return NEG
    if length_norm:
        total = total / L + char_reward * L
    return total


def score_all_words(log_probs: np.ndarray, trie: GrammarTrie,
                    length_norm: bool = True,
                    char_reward: float = 0.15) -> List[WordResult]:
    blank_idx = NUM_CLASSES - 1
    results: List[WordResult] = []
    for v in trie.variations:
        chars = [char_to_idx[c] for c in v.normalized]
        s = forced_align_score(log_probs, chars, blank_idx,
                               length_norm=length_norm, char_reward=char_reward)
        results.append(WordResult(
            intent=v.intent,
            transcript=v.phrase,
            normalized=v.normalized,
            variation_id=v.variation_id,
            slot=v.slot_value,
            score=float(s),
        ))
    results.sort(key=lambda r: r.score, reverse=True)
    return results
