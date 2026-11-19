"""When to force live web_search instead of trusting model memory.

Policy: external / world facts go stale. Default to requiring web evidence for
informational asks about the world; only skip for session-local tools, the
runtime clock, owner profile, chitchat, and clear action commands.
"""

from __future__ import annotations

import re

# Session-local or non-world intents — answer without forced web_search.
_SESSION_LOCAL = re.compile(
    r"(?:"
    r"^\s*(?:hi|hello|hey|thanks|thank\s+you|bye|goodbye|ok|okay|yes|no|sure|stop|"
    r"cancel|never\s*mind|nm)\b|"
    r"\b(?:what\s+time\s+is\s+it|what'?s\s+the\s+time|what\s+day\s+is\s+it|"
    r"what(?:'?s|\s+is)\s+today(?:'?s)?\s+date)\b|"
    r"\b(?:set|start|create|cancel|stop|list|show)\b.{0,40}\b(?:timer|reminder|alarm)s?\b|"
    r"\b(?:add|save|take|find|search|list|show|delete)\b.{0,40}\bnotes?\b|"
    r"\b(?:publish|deploy)\b.{0,40}\b(?:site|page|pages|github)\b|"
    r"\b(?:jobs?_|job\b|opencode|coding\s+job|build|create|make|submit).{0,60}"
    r"\b(?:site|page|landing|website|app)\b|"
    r"\b(?:jobs?_status|job\s+status|list\s+jobs|my\s+jobs)\b|"
    r"\b(?:who\s+am\s+i|what(?:'?s|\s+is)\s+my\s+name|my\s+profile)\b|"
    r"\b(?:what\s+can\s+you\s+do|your\s+capabilities|help)\b|"
    r"\b(?:unmute|mute)\b.{0,20}\bmic"
    r")",
    re.IGNORECASE | re.DOTALL,
)

# Looks like an ask for information (not just an imperative tool action).
_INFO_ASK = re.compile(
    r"(?:"
    r"^\s*(?:who|what|when|where|why|how|which|whose)\b|"
    r"\b(?:who|what|when|where|why|how|which)\b.+\?|"
    r"\b(?:tell\s+me|look\s+up|search(?:\s+for)?|find\s+out|do\s+you\s+know|"
    r"can\s+you\s+(?:tell|check|look)|what(?:'?s|\s+is|\s+are)\b|"
    r"who(?:'?s|\s+is|\s+are)\b|is\s+it\s+true|latest|current|today|"
    r"right\s+now|as\s+of|update\s+me|news\s+about)\b"
    r")",
    re.IGNORECASE,
)


def needs_live_web_evidence(text: str) -> bool:
    """True when this turn must call web_search before stating world facts."""
    t = (text or "").strip()
    if not t:
        return False
    if _SESSION_LOCAL.search(t):
        return False
    if _INFO_ASK.search(t):
        return True
    # Bare factual prompts without "?" still need evidence if they name the world.
    if re.search(
        r"\b(president|minister|election|weather|forecast|stock|price|score|"
        r"champion|winner|ceo|released|announced)\b",
        t,
        re.I,
    ):
        return True
    return False
