"""Pure-NumPy grammar decoder for the RPi5 (no torch import).

Contains only what inference needs: WordResult, forced_align_score,
score_all_words. Identical math to decode_grammar.py, minus the torch
training helpers. On the Pi, infer_rpi.py uses THIS module so torch is
never imported.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

try:
    from .grammar import NUM_CLASSES, char_to_idx, GrammarTrie
except ImportError:                      # flat-script mode (RPi)
    from grammar import NUM_CLASSES, char_to_idx, GrammarTrie


@dataclass
class WordResult:
    intent: str
    transcript: str
    score: float
    aligned_text: str


def forced_align_score(log_probs: np.ndarray, word_chars: List[int],
                       blank_idx: int, frame_penalty: float = 0.1,
                       length_norm: bool = True,
                       char_reward: float = 0.05) -> Tuple[float, List[int]]:
    T, C = log_probs.shape
    L = len(word_chars)
    NEG = -1e9
    best = [NEG] * (L + 1)
    best[0] = 0.0
    align_frame = [-1] * L
    for t in range(T):
        lp = log_probs[t]
        new_best = best[:]
        for l in range(L):
            if best[l] <= NEG / 2:
                continue
            c = word_chars[l]
            cand = best[l] + lp[c]
            if cand > new_best[l + 1]:
                new_best[l + 1] = cand
                if align_frame[l] == -1:
                    align_frame[l] = t
        best = new_best
    total = best[L]
    if total <= NEG / 2:
        return NEG, align_frame
    if length_norm:
        total = total / L + char_reward * L
    else:
        total -= frame_penalty * max(0, T - L)
    return total, align_frame


def score_all_words(log_probs: np.ndarray, trie: GrammarTrie,
                    frame_penalty: float = 0.1, length_norm: bool = True,
                    char_reward: float = 0.05) -> List[WordResult]:
    results: List[WordResult] = []
    for transcript in trie.all_transcripts():
        chars = [char_to_idx[c] for c in transcript]
        s, _ = forced_align_score(log_probs, chars, NUM_CLASSES - 1,
                                  frame_penalty, length_norm=length_norm,
                                  char_reward=char_reward)
        node = trie.root
        for c in transcript:
            node = node.children[c]
        intent = next(iter(node.intents))
        results.append(WordResult(intent, transcript, s, transcript))
    results.sort(key=lambda r: r.score, reverse=True)
    return results
