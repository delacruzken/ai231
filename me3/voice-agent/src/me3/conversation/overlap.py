"""Overlap pause classification: resume backchannel vs yield steering."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

BACKCHANNEL_ONLY = re.compile(
    r"^\s*(?:yeah|yep|yup|uh[\s-]?huh|mm+|mhm|okay|ok|right|sure|got it|uh|um|hmm+)\s*[.!]?\s*$",
    re.IGNORECASE,
)
STEERING = re.compile(
    r"\b(?:stop|wait|cancel|no\b|don't|dont|actually|instead|hold on|never ?mind|shut up)\b",
    re.IGNORECASE,
)


@dataclass
class OverlapConfig:
    yield_ms: int = 900
    min_speech_ms_for_steering: int = 350


@dataclass
class OverlapDecision:
    action: str  # wait | resume | yield
    reason: str


class OverlapClassifier:
    def __init__(self, cfg: OverlapConfig | None = None):
        self.cfg = cfg or OverlapConfig()
        self.started_mono: float | None = None
        self.partial = ""

    def begin(self, mono: float | None = None) -> None:
        self.started_mono = mono if mono is not None else time.monotonic()
        self.partial = ""

    def reset(self) -> None:
        self.started_mono = None
        self.partial = ""

    def on_partial(self, text: str) -> None:
        self.partial = (text or "").strip()

    def evaluate(self, now: float | None = None, speech_ms: float = 0.0) -> OverlapDecision:
        now = now if now is not None else time.monotonic()
        if self.started_mono is None:
            return OverlapDecision("wait", "not_in_overlap")
        elapsed_ms = (now - self.started_mono) * 1000.0
        text = self.partial
        if text and STEERING.search(text):
            return OverlapDecision("yield", "steering_phrase")
        if text and BACKCHANNEL_ONLY.match(text) and speech_ms < self.cfg.min_speech_ms_for_steering * 2:
            return OverlapDecision("resume", "backchannel_like")
        if speech_ms >= self.cfg.min_speech_ms_for_steering and text and len(text.split()) >= 3:
            return OverlapDecision("yield", "contentful_partial")
        if elapsed_ms >= self.cfg.yield_ms:
            if text and BACKCHANNEL_ONLY.match(text):
                return OverlapDecision("resume", "timeout_backchannel")
            return OverlapDecision("yield", "timeout_unresolved")
        return OverlapDecision("wait", "collecting")
