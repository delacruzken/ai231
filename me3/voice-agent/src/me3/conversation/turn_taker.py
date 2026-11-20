"""Adaptive end-of-turn from last speech audio + transcript heuristics."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from me3.conversation.asr_filters import is_asr_ghost

# Function / determiner / preposition tails — speaker is mid-clause.
INCOMPLETE_TAIL = re.compile(
    r"(?:\b(?:and|or|but|if|when|while|because|so|to|with|for|of|from|about|"
    r"the|a|an|my|your|our|their|his|her|its|this|that|these|those|"
    r"is|are|was|were|be|been|being|do|does|did|can|could|would|will|shall|may|might|"
    r"who|what|when|where|why|how|which|whose)\s*$|[,\-–—…:/]\s*$)",
    re.IGNORECASE,
)

# Open questions / prompts that usually continue unless ASR emitted terminal punctuation.
OPEN_STEM = re.compile(
    r"^(?:please\s+)?(?:who|what|when|where|why|how|which|whose|do|does|did|is|are|"
    r"can|could|would|will|tell\s+me|i\s+(?:want|would\s+like|need)\s+to\s+know)\b",
    re.IGNORECASE,
)

# Truncated / initialism fragments: "Pre", "Pres", "G-C", "G C"
TRUNCATED_TAIL = re.compile(
    r"(?:\b[A-Z][a-z]{0,3}\s*$|\b[A-Za-z](?:-[A-Za-z]{0,3})+\s*$|\b[A-Z](?:\s+[A-Z]){1,3}\s*$)"
)

# Short final tokens that are complete words (not cut-off syllables).
_COMPLETE_SHORT_TAILS = frozenset(
    {"am", "pm", "ok", "no", "yes", "hi", "me", "us", "it", "id", "ai", "tv"}
)


@dataclass
class TurnTakerConfig:
    silence_normal_ms: int = 700
    silence_incomplete_ms: int = 1800
    silence_clarification_ms: int = 2600


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
        if t.endswith("."):
            # Kroko often emits periods; treat as complete unless it looks truncated.
            body = t[:-1].rstrip()
            if not TRUNCATED_TAIL.search(body) and not INCOMPLETE_TAIL.search(body):
                return False
            return True
        # No terminal punctuation: prefer waiting. Kroko often omits periods, and
        # silence-commit at the normal gate was cutting mid-phrase on the Pi bank.
        if INCOMPLETE_TAIL.search(t) or TRUNCATED_TAIL.search(t) or OPEN_STEM.search(t):
            return True
        last = re.sub(r"[^A-Za-z0-9']+", "", t.split()[-1])
        if 1 <= len(last) <= 2 and last.lower() not in _COMPLETE_SHORT_TAILS:
            return True
        if len(t.split()) < 3:
            return True
        return True

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
            if is_asr_ghost(self.partial_text):
                # Drop silence/noise ghosts — do not clarify or commit.
                self.reset()
                return TurnDecision("none", silence_ms, thr, True)
            self._commit_armed = True
            return TurnDecision("commit", silence_ms, thr, incomplete)
        return TurnDecision("none", silence_ms, thr, incomplete)

    def cancel_pending(self) -> None:
        """Resumed speech before commit — cancel pending end-of-turn."""
        self._commit_armed = False
        self._clarify_sent = False
