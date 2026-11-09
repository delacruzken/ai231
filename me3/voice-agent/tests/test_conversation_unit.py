"""Phase 3 unit tests (no mic; mock LLM)."""

from __future__ import annotations

import time

from me3.conversation.acks import AckManager
from me3.conversation.overlap import OverlapClassifier
from me3.conversation.turn_taker import TurnTaker, TurnTakerConfig


def test_turn_taker_incomplete_needs_longer_silence():
    tt = TurnTaker(TurnTakerConfig(silence_normal_ms=600, silence_incomplete_ms=1200))
    tt.on_partial("I want to")
    now = time.monotonic()
    tt.last_speech_mono = now - 0.7
    assert tt.evaluate(now=now).action == "none"
    tt.last_speech_mono = now - 1.3
    assert tt.evaluate(now=now).action == "commit"


def test_turn_taker_complete_600():
    tt = TurnTaker(TurnTakerConfig())
    tt.on_partial("What is the weather today?")
    now = time.monotonic()
    tt.last_speech_mono = now - 0.65
    assert tt.evaluate(now=now).action == "commit"


def test_overlap_steering_and_backchannel():
    oc = OverlapClassifier()
    oc.begin()
    oc.on_partial("okay")
    assert oc.evaluate(speech_ms=150).action == "resume"
    oc.begin()
    oc.on_partial("wait stop that")
    assert oc.evaluate(speech_ms=400).action == "yield"


def test_ack_delay_and_suppress():
    ack = AckManager()
    ack.on_operation_started()
    ack.state.pending_since = time.monotonic() - 0.8
    assert ack.should_ack() is not None
    ack.on_operation_started()
    ack.on_result_ready()
    assert ack.should_ack() is None
