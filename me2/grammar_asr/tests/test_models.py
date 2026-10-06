import torch

from grammar_asr.models import build_model, count_params
from grammar_asr.schema import NUM_INTENT_OOS, SLOT_INTENTS


def test_intent_forward_and_size():
    m = build_model("intent", gru_hidden=96, gru_layers=2)
    x = torch.randn(2, 300, 40)
    out = m(x)
    assert out["intent_logits"].shape == (2, NUM_INTENT_OOS)
    for intent in SLOT_INTENTS:
        assert intent in out["slot_logits"]
    # Compact on-device band (~0.2–0.4M); allow some slack
    n = count_params(m)
    assert 100_000 < n < 600_000, n


def test_ctc_low_temporal():
    m = build_model("ctc_low", hidden=64, num_blocks=2)
    x = torch.randn(2, 200, 40)
    out = m(x)
    logits = out["ctc_logits"]
    assert logits.shape[0] == 2
    assert logits.shape[1] >= 90  # ~T/2
    assert logits.shape[2] == 28


def test_hybrid_forward():
    m = build_model("hybrid", hidden=64, num_blocks=2)
    out = m(torch.randn(2, 200, 40))
    assert "ctc_logits" in out and "intent_logits" in out
    assert out["intent_logits"].shape[-1] == NUM_INTENT_OOS


def test_wake_tiny():
    m = build_model("wake")
    out = m(torch.randn(2, 100, 40))
    assert out["wake_logits"].shape == (2, 2)
    assert count_params(m) < 50_000
