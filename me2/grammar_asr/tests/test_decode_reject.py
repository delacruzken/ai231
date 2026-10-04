import numpy as np
import torch

from grammar_asr.decode import (
    build_grammar, decode_model_output, RejectGate, calibrate_reject_gate,
)
from grammar_asr.decode.semantic import SemanticPrediction
from grammar_asr.schema import OUT_OF_SCOPE, NUM_INTENT_OOS, VARIATIONS


def test_grammar_has_93():
    trie = build_grammar()
    assert trie.num_words == 93
    assert "make a call" in trie.all_transcripts()
    assert "pause song" in trie.all_transcripts()


def test_intent_head_decode_oos():
    logits = torch.zeros(1, NUM_INTENT_OOS)
    logits[0, -1] = 5.0  # OOS
    pred = decode_model_output({"intent_logits": logits, "slot_logits": {}})
    assert pred.intent == OUT_OF_SCOPE
    assert pred.rejected


def test_reject_gate():
    gate = RejectGate(min_confidence=0.8)
    pred = SemanticPrediction("TIMER", "30 seconds", "Timer 30 seconds", 0,
                              False, 0.5, 0.1, 0.5, "intent_head")
    out = gate.apply(pred)
    assert out.rejected and out.intent == OUT_OF_SCOPE


def test_calibrate_prefers_far_ceiling():
    preds = []
    labels = []
    # in-scope high conf
    for _ in range(20):
        preds.append(SemanticPrediction("TIME", "", "Time", 0, False, 0.9, 0.5, 0.9, "x"))
        labels.append(False)
    # oos medium conf
    for _ in range(20):
        preds.append(SemanticPrediction("TIME", "", "Time", 0, False, 0.4, 0.1, 0.4, "x"))
        labels.append(True)
    gate = calibrate_reject_gate(preds, labels, target_far=0.1)
    assert gate.min_confidence >= 0.4


def test_log_dict_fields():
    pred = SemanticPrediction(VARIATIONS[0].intent, VARIATIONS[0].slot_value,
                              VARIATIONS[0].phrase, VARIATIONS[0].variation_id,
                              False, 1.0, 0.2, 0.8, "grammar")
    d = pred.to_log_dict(12.3, 1500)
    assert {"intent", "slot", "infer_ms", "audio_ms"} <= set(d)
