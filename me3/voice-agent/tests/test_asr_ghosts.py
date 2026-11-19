"""ASR ghost / filler rejection."""

from __future__ import annotations

import time

from me3.conversation.asr_filters import is_asr_ghost
from me3.conversation.turn_taker import TurnTaker, TurnTakerConfig


def test_ghost_tokens():
    for t in ("so", "Now", "okay.", "Thank you", "one", "OK", "um"):
        assert is_asr_ghost(t), t
    for t in ("Who is the mayor?", "Set a timer for five minutes", "Thank you for the page"):
        assert not is_asr_ghost(t), t


def test_turn_taker_drops_ghost_instead_of_commit():
    tt = TurnTaker(TurnTakerConfig(silence_normal_ms=200, silence_incomplete_ms=400))
    tt.on_partial("so")
    now = time.monotonic()
    tt.last_speech_mono = now - 2.0
    assert tt.evaluate(now=now).action == "none"
    assert tt.partial_text == ""
