"""Typed semantic prediction + calibrated OOS rejection."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..schema import (
    OUT_OF_SCOPE, INTENT_TO_ID, ID_TO_INTENT, SLOT_INTENTS, SLOT_VALUES,
    SLOT_VALUE_TO_ID, VARIATIONS, lookup_variation,
)
from ..text import NUM_CLASSES
from .forced_align import score_all_words
from .grammar_trie import build_grammar, GrammarTrie


@dataclass
class SemanticPrediction:
    intent: str
    slot: str
    variation: str
    variation_id: int
    rejected: bool
    score: float
    margin: float
    confidence: float
    source: str  # "grammar" | "intent_head" | "hybrid"

    def to_log_dict(self, infer_ms: float, audio_ms: float) -> dict:
        return {
            "intent": self.intent if not self.rejected else OUT_OF_SCOPE,
            "slot": "" if self.rejected else self.slot,
            "variation": OUT_OF_SCOPE if self.rejected else (
                self.variation or self.intent
            ),
            "variation_id": -1 if self.rejected else self.variation_id,
            "infer_ms": round(float(infer_ms), 1),
            "audio_ms": round(float(audio_ms)),
            "score": round(float(self.score), 4),
            "margin": round(float(self.margin), 4),
            "confidence": round(float(self.confidence), 4),
            "rejected": bool(self.rejected),
            "source": self.source,
        }


class RejectGate:
    """Score/margin/confidence thresholds for OOS rejection."""

    def __init__(self, min_score: float = -1e9, min_margin: float = 0.0,
                 min_confidence: float = 0.0,
                 intent_thresholds: Optional[Dict[str, float]] = None):
        self.min_score = min_score
        self.min_margin = min_margin
        self.min_confidence = min_confidence
        self.intent_thresholds = intent_thresholds or {}

    def should_reject(self, pred: SemanticPrediction) -> bool:
        if pred.intent == OUT_OF_SCOPE:
            return True
        thr = self.intent_thresholds.get(pred.intent, self.min_confidence)
        if pred.confidence < thr:
            return True
        if pred.score < self.min_score:
            return True
        if pred.margin < self.min_margin:
            return True
        return False

    def apply(self, pred: SemanticPrediction) -> SemanticPrediction:
        if self.should_reject(pred):
            return SemanticPrediction(
                intent=OUT_OF_SCOPE, slot="", variation=OUT_OF_SCOPE,
                variation_id=-1, rejected=True, score=pred.score,
                margin=pred.margin, confidence=pred.confidence,
                source=pred.source,
            )
        return pred

    def to_dict(self) -> dict:
        return {
            "min_score": self.min_score,
            "min_margin": self.min_margin,
            "min_confidence": self.min_confidence,
            "intent_thresholds": self.intent_thresholds,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RejectGate":
        return cls(
            min_score=float(d.get("min_score", -1e9)),
            min_margin=float(d.get("min_margin", 0.0)),
            min_confidence=float(d.get("min_confidence", 0.0)),
            intent_thresholds=dict(d.get("intent_thresholds") or {}),
        )

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> "RejectGate":
        return cls.from_dict(json.loads(path.read_text()))


def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def decode_from_ctc_logits(ctc_logits, trie: Optional[GrammarTrie] = None,
                           char_reward: float = 0.15) -> SemanticPrediction:
    if isinstance(ctc_logits, torch.Tensor):
        lp = torch.log_softmax(ctc_logits, dim=-1).detach().cpu().numpy()
    else:
        x = ctc_logits - ctc_logits.max(axis=-1, keepdims=True)
        lp = x - np.log(np.exp(x).sum(axis=-1, keepdims=True))
    if lp.ndim == 3:
        lp = lp[0]
    trie = trie or build_grammar()
    ranked = score_all_words(lp, trie, char_reward=char_reward)
    if not ranked:
        return SemanticPrediction(OUT_OF_SCOPE, "", OUT_OF_SCOPE, -1, True,
                                  -1e9, 0.0, 0.0, "grammar")
    best, second = ranked[0], ranked[1] if len(ranked) > 1 else None
    margin = best.score - (second.score if second else best.score - 1.0)
    # Map score to a pseudo-confidence via top-2 softmax over scores.
    scores = np.array([r.score for r in ranked[:8]], dtype=np.float64)
    conf = float(_softmax_np(scores)[0])
    return SemanticPrediction(
        intent=best.intent, slot=best.slot, variation=best.transcript,
        variation_id=best.variation_id, rejected=False,
        score=best.score, margin=margin, confidence=conf, source="grammar",
    )


def decode_from_intent_slot_heads(intent_logits, slot_logits: dict) -> SemanticPrediction:
    if isinstance(intent_logits, torch.Tensor):
        intent_logits = intent_logits.detach().cpu().numpy()
    if intent_logits.ndim == 2:
        intent_logits = intent_logits[0]
    probs = _softmax_np(intent_logits.astype(np.float64))
    intent_id = int(probs.argmax())
    conf = float(probs[intent_id])
    # margin over top-2 intent probs
    order = probs.argsort()[::-1]
    margin = float(probs[order[0]] - probs[order[1]] if len(order) > 1 else probs[order[0]])
    intent = ID_TO_INTENT[intent_id]
    if intent == OUT_OF_SCOPE:
        return SemanticPrediction(OUT_OF_SCOPE, "", OUT_OF_SCOPE, -1, True,
                                  conf, margin, conf, "intent_head")
    slot = ""
    if intent in SLOT_INTENTS:
        logits = slot_logits[intent]
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()
        if logits.ndim == 2:
            logits = logits[0]
        sid = int(np.argmax(logits))
        slot = SLOT_VALUES[intent][sid]
    # Pick a canonical variation phrase for this (intent, slot)
    phrase = ""
    vid = -1
    for v in VARIATIONS:
        if v.intent == intent and v.slot_value == slot:
            phrase = v.phrase
            vid = v.variation_id
            break
    return SemanticPrediction(
        intent=intent, slot=slot, variation=phrase, variation_id=vid,
        rejected=False, score=conf, margin=margin, confidence=conf,
        source="intent_head",
    )


def decode_model_output(outputs: dict, trie: Optional[GrammarTrie] = None,
                        char_reward: float = 0.15,
                        prefer: str = "auto") -> SemanticPrediction:
    """Decode a model forward() dict into a SemanticPrediction."""
    has_ctc = "ctc_logits" in outputs
    has_intent = "intent_logits" in outputs

    if prefer == "grammar" and has_ctc:
        return decode_from_ctc_logits(outputs["ctc_logits"], trie, char_reward)
    if prefer == "intent_head" and has_intent:
        return decode_from_intent_slot_heads(outputs["intent_logits"],
                                            outputs.get("slot_logits") or {})
    if has_intent and not has_ctc:
        return decode_from_intent_slot_heads(outputs["intent_logits"],
                                            outputs.get("slot_logits") or {})
    if has_ctc and not has_intent:
        return decode_from_ctc_logits(outputs["ctc_logits"], trie, char_reward)

    # Hybrid: take intent head if confident OOS or strong, else grammar.
    intent_pred = decode_from_intent_slot_heads(outputs["intent_logits"],
                                                outputs.get("slot_logits") or {})
    gram_pred = decode_from_ctc_logits(outputs["ctc_logits"], trie, char_reward)
    if intent_pred.intent == OUT_OF_SCOPE and intent_pred.confidence >= 0.5:
        intent_pred.source = "hybrid"
        return intent_pred
    # Prefer agreement
    if intent_pred.intent == gram_pred.intent and intent_pred.slot == gram_pred.slot:
        gram_pred.confidence = max(gram_pred.confidence, intent_pred.confidence)
        gram_pred.source = "hybrid"
        return gram_pred
    # Otherwise prefer higher confidence channel
    if intent_pred.confidence >= gram_pred.confidence:
        intent_pred.source = "hybrid"
        return intent_pred
    gram_pred.source = "hybrid"
    return gram_pred


def calibrate_reject_gate(
    predictions: Sequence[SemanticPrediction],
    labels_oos: Sequence[bool],
    target_far: float = 0.10,
) -> RejectGate:
    """Sweep confidence threshold on a calibration set (e.g. val + tune_oos)."""
    confs = sorted({round(p.confidence, 4) for p in predictions})
    best = RejectGate(min_confidence=0.0)
    best_score = -1e9
    for thr in confs or [0.0]:
        gate = RejectGate(min_confidence=thr, min_margin=0.0)
        fa = fr = 0
        n_oos = n_in = 0
        for pred, is_oos in zip(predictions, labels_oos):
            out = gate.apply(pred)
            if is_oos:
                n_oos += 1
                if not out.rejected:
                    fa += 1
            else:
                n_in += 1
                if out.rejected:
                    fr += 1
        far = fa / max(n_oos, 1)
        frr = fr / max(n_in, 1)
        # maximize in-scope keep under FAR ceiling
        score = (1.0 - frr) - (0.0 if far <= target_far else 10.0 * (far - target_far))
        if score > best_score:
            best_score = score
            best = gate
    return best
