"""Reject ASR ghost / filler commits that are not real user turns."""

from __future__ import annotations

import re

# Lone tokens Kroko often invents from silence / room noise.
_ASR_GHOST = re.compile(
    r"^(?:"
    r"ok(?:ay)?|so|now|one|um+|uh+|ah+|oh|hmm+|mm+|mhm|"
    r"yes|yeah|yep|no|nope|right|well|and|the|a|an|to|for|"
    r"please|thanks?|thank\s+you|hey|hi|hello|"
    r"huh|what|sorry"
    r")[\s\.\!\,]*$",
    re.IGNORECASE,
)


def is_asr_ghost(text: str) -> bool:
    """True for stray one-word (or filler) transcripts that should not start a turn."""
    t = (text or "").strip()
    if not t:
        return True
    if _ASR_GHOST.match(t):
        return True
    # Extremely short non-question fragments with no substance.
    words = re.findall(r"[A-Za-z0-9']+", t)
    if len(words) == 1 and len(words[0]) <= 3 and not t.endswith("?"):
        return True
    return False
