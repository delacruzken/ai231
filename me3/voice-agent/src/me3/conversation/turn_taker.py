"""Adaptive end-of-turn from last speech audio + transcript heuristics."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

INCOMPLETE_TAIL = re.compile(
    r"(?:\b(?:and|or|but|if|when|because|so|to|with|for|the|a|an|my|your)\s*$|[,\-–—…]\s*$)",
    re.IGNORECASE,
)


@dataclass
class TurnTakerConfig:
    silence_normal_ms: int = 600
    silence_incomplete_ms: int = 1200
    silence_clarification_ms: int = 2000


@dataclass
class TurnDecision:
    action: str  # none | commit | clarify
    silence_ms: float
    threshold_ms: int
    incomplete: bool


class TurnTaker:
    """Clock starts from last detected speech audio, not stacked timers."""

    def __init__(self, cfg: TurnTakerConfig | None = None):
        self.cfg = cfg or TurnTakerConfig()
        self.last_speech_mono: float | None = None
        self.partial_text = ""
        self._commit_armed = False
        self._clarify_sent = False

    def reset(self) -> None:
        self.last_speech_mono = None
        self.partial_text = ""
        self._commit_armed = False
        self._clarify_sent = False

    def on_speech(self, mono: float | None = None) -> None:
        self.last_speech_mono = mono if mono is not None else time.monotonic()
        self._commit_armed = False
        self._clarify_sent = False

    def on_partial(self, text: str) -> None:
        """Update transcript text without resetting the silence clock.

        Silence is driven by VAD ``on_speech``. Resetting on every ASR partial
        made endpointing wait until partials stopped trickling (often slow, and
        worse with echo). Only treat *growing* text as continued speech.
        """
        t = (text or "").strip()
        prev = self.partial_text
        if t and len(t) > len(prev):
            self.on_speech()
        self.partial_text = t

    def is_incomplete(self, text: str | None = None) -> bool:
        t = (text if text is not None else self.partial_text).strip()
        if not t:
            return True
        # Clear questions / exclamations are complete even if short.
        if t.endswith(("?", "!")):
            return False
        if INCOMPLETE_TAIL.search(t):
            return True
        # very short fragments
        if len(t.split()) < 2:
            return True
        return False

    def threshold_ms(self, text: str | None = None) -> int:
        if self.is_incomplete(text):
            return self.cfg.silence_incomplete_ms
        return self.cfg.silence_normal_ms

    def evaluate(self, now: float | None = None) -> TurnDecision:
        now = now if now is not None else time.monotonic()
        if self.last_speech_mono is None:
            return TurnDecision("none", 0.0, self.cfg.silence_normal_ms, True)
        silence_ms = (now - self.last_speech_mono) * 1000.0
        incomplete = self.is_incomplete()
        thr = self.cfg.silence_incomplete_ms if incomplete else self.cfg.silence_normal_ms
        if silence_ms >= self.cfg.silence_clarification_ms and incomplete and not self._clarify_sent:
            self._clarify_sent = True
            return TurnDecision("clarify", silence_ms, self.cfg.silence_clarification_ms, incomplete)
        if silence_ms >= thr and self.partial_text and not self._commit_armed:
            self._commit_armed = True
            return TurnDecision("commit", silence_ms, thr, incomplete)
        return TurnDecision("none", silence_ms, thr, incomplete)

    def cancel_pending(self) -> None:
        """Resumed speech before commit — cancel pending end-of-turn."""
        self._commit_armed = False
        self._clarify_sent = False
