"""
Grammar-constrained CTC decoder — forced-alignment (Viterbi) formulation.

Why not a naive beam? With a fixed grammar, the right move is to score EVERY
grammar word against the acoustic posteriors and pick the best. A naive beam
that carries a "blank-stay" hypothesis compounds its score each frame, so short
commands always win — degenerate. Forced alignment avoids this: each candidate
word is aligned to the frames where it actually fits, scored by its total
log-prob, and the highest-scoring word wins.

This is the PocketSphinx recipe: acoustic model (CTC posteriors) + grammar
(trie of known words) -> Viterbi/forced-align each word -> argmax.

  score(word) = sum over its characters of the best-aligned frame log-prob,
                with a small penalty for unused frames (so a short word can't
                trivially beat a long one).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .asr_model import CTCEncoder, greedy_decode
from .asr_model_res import ResCTCEncoder
from .asr_model_low import ResCTCEncoderLow
from .features import load_wav, log_mel, normalize_feat, load_manifest
from .grammar import (GrammarTrie, build_grammar, normalize, char_to_idx,
                      idx_to_char, NUM_CLASSES, BLANK)


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
    """
    Score one word against per-frame posteriors via CTC forced alignment.

    log_probs: (T, C) log-prob per frame.
    word_chars: list of character indices (no blanks).
    Returns (total_score, chosen_frame_per_char).

    DP: align L characters to T frames, monotonic, each char may emit over a
    run of frames (CTC allows repeats). We take the max over alignments.

    length_norm: divide by L so a long word isn't penalized just for having
    more characters (fixes the short-word bias).
    """
    T, C = log_probs.shape
    L = len(word_chars)
    NEG = -1e9

    best = [NEG] * (L + 1)
    best[0] = 0.0
    align_frame = [-1] * L

    for t in range(T):
        lp = log_probs[t]
        new_best = best[:]  # blank / no-advance
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
        # mean log-prob per char + a small reward for longer (more specific)
        # commands, so a 4-char word can't outscores a 12-char one on ties.
        total = total / L + char_reward * L
    else:
        total -= frame_penalty * max(0, T - L)
    return total, align_frame


def score_all_words(log_probs: np.ndarray, trie: GrammarTrie,
                    frame_penalty: float = 0.1, length_norm: bool = True,
                    char_reward: float = 0.05) -> List[WordResult]:
    """Score every grammar terminal word; return sorted best-first."""
    blank_idx = NUM_CLASSES - 1
    results: List[WordResult] = []
    for transcript in trie.all_transcripts():
        chars = [char_to_idx[c] for c in transcript]
        s, _ = forced_align_score(log_probs, chars, blank_idx, frame_penalty,
                                  length_norm=length_norm,
                                  char_reward=char_reward)
        # find an intent for this transcript
        node = trie.root
        for c in transcript:
            node = node.children[c]
        intent = next(iter(node.intents))
        results.append(WordResult(intent, transcript, s, transcript))
    results.sort(key=lambda r: r.score, reverse=True)
    return results


def make_model(model_type: str = "low", **kwargs):
    if model_type == "base":
        return CTCEncoder(**{k: v for k, v in kwargs.items()
                             if k in ("hidden", "num_layers")})
    if model_type == "res":
        return ResCTCEncoder(**{k: v for k, v in kwargs.items()
                                if k in ("hidden", "num_blocks")})
    return ResCTCEncoderLow(**{k: v for k, v in kwargs.items()
                               if k in ("hidden", "num_blocks")})


def decode_audio(model, audio_path, trie: GrammarTrie,
                 device: str = "cpu", mode: str = "grammar",
                 frame_penalty: float = 0.1, char_reward: float = 0.05):
    """Full pipeline: WAV -> features -> logits -> decode."""
    from pathlib import Path
    audio = load_wav(Path(audio_path))
    feat = normalize_feat(log_mel(audio))
    x = torch.from_numpy(feat).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(x).squeeze(0)          # (T', C)
    log_probs = torch.log_softmax(logits, dim=-1).cpu().numpy()

    if mode == "greedy":
        ids = greedy_decode(logits.unsqueeze(0))[0]
        text = "".join(idx_to_char[i] for i in ids if i != NUM_CLASSES - 1)
        node = trie.root
        for c in text:
            if c in node.children:
                node = node.children[c]
            else:
                node = None
                break
        intent = next(iter(node.intents)) if (node and node.is_terminal) else None
        return {"intent": intent, "text": text, "score": None}

    else:  # grammar (forced alignment)
        results = score_all_words(log_probs, trie, frame_penalty,
                                  length_norm=True, char_reward=char_reward)
        if results:
            r = results[0]
            return {"intent": r.intent, "text": r.transcript, "score": r.score}
        return {"intent": None, "text": "", "score": -1e9}


def evaluate_command_accuracy(model, samples, trie: GrammarTrie,
                              device: str = "cpu", mode: str = "grammar",
                              frame_penalty: float = 0.1,
                              char_reward: float = 0.05,
                              max_n: Optional[int] = None):
    """End-to-end command (intent) accuracy."""
    from collections import defaultdict
    correct = 0
    total = 0
    per_intent = defaultdict(lambda: [0, 0])
    errors = []
    for i, s in enumerate(samples):
        if max_n and i >= max_n:
            break
        res = decode_audio(model, s.path, trie, device, mode, frame_penalty,
                           char_reward)
        pred = res["intent"]
        ref = s.intent
        per_intent[ref][1] += 1
        if pred == ref:
            correct += 1
            per_intent[ref][0] += 1
        else:
            errors.append({"ref": ref, "pred": pred, "text": res["text"],
                           "file": s.path.name})
        total += 1
    acc = correct / max(total, 1)
    return {
        "accuracy": acc, "correct": correct, "total": total,
        "per_intent": {k: {"correct": v[0], "total": v[1],
                            "acc": v[0] / max(v[1], 1)}
                        for k, v in sorted(per_intent.items())},
        "errors": errors[:50],
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--condition", type=str, default=None)
    ap.add_argument("--mode", type=str, default="grammar",
                    choices=["grammar", "greedy"])
    ap.add_argument("--frame-penalty", type=float, default=0.1)
    ap.add_argument("--char-reward", type=float, default=0.05)
    ap.add_argument("--max-n", type=int, default=None)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--model", type=str, default="low",
                    choices=["base", "res", "low"])
    args = ap.parse_args()

    device = torch.device(args.device)
    model = make_model(args.model).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    trie = build_grammar()
    samples = load_manifest(split=args.split, condition=args.condition)
    print(f"Evaluating {len(samples)} samples ({args.split}/{args.condition}) "
          f"mode={args.mode} frame_penalty={args.frame_penalty} "
          f"char_reward={args.char_reward}")
    res = evaluate_command_accuracy(model, samples, trie, device,
                                    args.mode, args.frame_penalty,
                                    args.char_reward, args.max_n)
    print(f"\nCOMMAND ACCURACY: {res['accuracy']*100:.2f}% "
          f"({res['correct']}/{res['total']})")
    print("\nPer-intent:")
    for intent, st in res["per_intent"].items():
        bar = "#" * int(st["acc"] * 20)
        print(f"  {intent:20s} {st['correct']:3d}/{st['total']:<3d} "
              f"{st['acc']*100:5.1f}% {bar}")
    if res["errors"]:
        print(f"\nSample errors ({min(len(res['errors']),10)} shown):")
        for e in res["errors"][:10]:
            print(f"  ref={e['ref']:18s} pred={str(e['pred']):18s} "
                  f"text={e['text']!r} file={e['file']}")
