"""Phase 3 unit tests (no mic; mock LLM)."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from me3.conversation.acks import AckManager
from me3.conversation.overlap import OverlapClassifier
from me3.conversation.pipeline import ConversationOrchestrator, _asr_text_compatible
from me3.conversation.turn_taker import TurnTaker, TurnTakerConfig


def test_turn_taker_incomplete_needs_longer_silence():
    tt = TurnTaker(TurnTakerConfig(silence_normal_ms=600, silence_incomplete_ms=1200))
    tt.on_partial("I want to")
    now = time.monotonic()
    tt.last_speech_mono = now - 0.7
    assert tt.evaluate(now=now).action == "none"
    tt.last_speech_mono = now - 1.3
    assert tt.evaluate(now=now).action == "commit"


def test_turn_taker_complete_question_fast():
    tt = TurnTaker(TurnTakerConfig(silence_normal_ms=400, silence_incomplete_ms=900))
    tt.on_partial("What is the weather today?")
    now = time.monotonic()
    tt.last_speech_mono = now - 0.45
    assert tt.evaluate(now=now).action == "commit"


def test_turn_taker_partial_growth_resets_silence_not_same_text():
    tt = TurnTaker(TurnTakerConfig(silence_normal_ms=400))
    tt.on_partial("What time")
    first = tt.last_speech_mono
    time.sleep(0.05)
    tt.on_partial("What time")  # unchanged — should not reset silence
    assert tt.last_speech_mono == first
    tt.on_partial("What time is it")  # growth — resets
    assert tt.last_speech_mono is not None and tt.last_speech_mono >= first


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


def test_asr_text_compatible_prefix_and_mismatch():
    assert _asr_text_compatible("open the garage", "open the garage door please")
    assert _asr_text_compatible("Turn off all smart plugs.", "TURN OFF ALL SMART PLUGS")
    assert not _asr_text_compatible("while sheds on", "should i consider freelancing")


def test_silence_commit_marks_llm_request_speculative(monkeypatch):
    monkeypatch.setenv("OPENCODE_FORCE_MOCK", "true")
    orch = ConversationOrchestrator(force_mock=True)
    orch.set_state = MagicMock()
    marks: list[tuple[str, dict]] = []

    def capture(stage: str, **fields):
        marks.append((stage, fields))

    with patch("me3.benchmark.marks.mark", side_effect=capture):
        tid = orch.commit_user_turn("Open the garage door please", speculative=True)
        assert tid
        # worker is async; give mock LLM a moment
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not any(s == "llm_request" for s, _ in marks):
            time.sleep(0.02)
    req = next(((s, f) for s, f in marks if s == "llm_request"), None)
    assert req is not None
    assert req[1].get("speculative") is True
    orch.cancel_active(reason="test_done")


def test_asr_final_mismatch_restarts_confirmed_turn(monkeypatch):
    monkeypatch.setenv("OPENCODE_FORCE_MOCK", "true")
    orch = ConversationOrchestrator(force_mock=True)
    orch.set_state = MagicMock()
    # Simulate an in-flight speculative commit without racing the mock LLM worker.
    orch.active_turn_id = "turn-spec"
    orch.active_user_text = "while sheds on"
    orch.active_llm_speculative = True
    orch.cancel_active = MagicMock(wraps=orch.cancel_active)
    orch.commit_user_turn = MagicMock(return_value="turn-final")

    orch.on_asr_final("Should I consider freelancing?")

    orch.cancel_active.assert_called_once_with(reason="asr_final_correction")
    orch.commit_user_turn.assert_called_once_with(
        "Should I consider freelancing?", speculative=False
    )


def test_asr_final_compatible_keeps_speculative_turn(monkeypatch):
    monkeypatch.setenv("OPENCODE_FORCE_MOCK", "true")
    orch = ConversationOrchestrator(force_mock=True)
    orch.active_turn_id = "turn-spec"
    orch.active_user_text = "open the garage"
    orch.active_llm_speculative = True
    orch.turn_taker.on_partial("open the garage")
    orch.cancel_active = MagicMock()
    orch.commit_user_turn = MagicMock()

    orch.on_asr_final("Open the garage door please")

    orch.cancel_active.assert_not_called()
    orch.commit_user_turn.assert_not_called()
    assert orch.active_llm_speculative is False
    assert orch.turn_taker.partial_text == ""


def test_stale_asr_final_after_turn_does_not_rearm(monkeypatch):
    """Late Zipformer final must not seed turn_taker after the answer finished."""
    monkeypatch.setenv("OPENCODE_FORCE_MOCK", "true")
    orch = ConversationOrchestrator(force_mock=True)
    orch._last_commit_norm = "hello what time is it now"
    orch._last_commit_mono = time.monotonic()
    orch.active_turn_id = None
    orch.active_llm_speculative = False
    orch.commit_user_turn = MagicMock()

    orch.on_asr_final("HELLO WHAT TIME IS IT NOW")

    orch.commit_user_turn.assert_not_called()
    assert orch.turn_taker.partial_text == ""


def test_duplicate_commit_window_skips_repeat(monkeypatch):
    monkeypatch.setenv("OPENCODE_FORCE_MOCK", "true")
    orch = ConversationOrchestrator(force_mock=True)
    orch.set_state = MagicMock()
    orch._last_commit_norm = "hello what time is it now"
    orch._last_commit_mono = time.monotonic() - 8.0  # beyond old 4s window, inside 12s
    orch.active_turn_id = None

    tid = orch.commit_user_turn("HELLO WHAT TIME IS IT NOW", speculative=True)
    assert tid == ""
    assert orch.active_turn_id is None


def test_norm_utt_helpers():
    from me3.session.controller import _norm_utt, _utt_compatible

    assert _norm_utt("HELLO, Ken!") == "hello ken"
    assert _utt_compatible("what time is it", "WHAT TIME IS IT NOW")
    assert not _utt_compatible("hello", "where is the holiest place")
