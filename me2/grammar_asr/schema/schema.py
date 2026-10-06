"""Canonical ME2 Option-B semantic schema.

Source of truth: schema/variations.csv (class Option B sheet mirrored from
the course gold Hugging Face dataset).

Counts:
  - 19 intents + OUT_OF_SCOPE
  - 93 phrase/slot combinations (variations)
  - 6 slotted intents × 3 values
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCHEMA_DIR = Path(__file__).resolve().parent
VARIATIONS_CSV = SCHEMA_DIR / "variations.csv"

OUT_OF_SCOPE = "OUT_OF_SCOPE"

FIXED_INTENTS = [
    "PLAY_MUSIC", "WEATHER", "TIME", "LIGHT_ON", "LIGHT_OFF",
    "PAUSE", "STOP", "NEXT", "VOLUME_UP", "VOLUME_DOWN",
    "CALL", "MESSAGE", "LIST_REMINDERS",
]
SLOT_INTENTS = [
    "TIMER", "ALARM", "TEMPERATURE", "BRIGHTNESS", "COLOR", "CREATE_REMINDER",
]
INTENTS = FIXED_INTENTS + SLOT_INTENTS
INTENT_TO_ID = {name: i for i, name in enumerate(INTENTS)}
INTENT_TO_ID[OUT_OF_SCOPE] = len(INTENTS)  # 19
ID_TO_INTENT = {i: name for name, i in INTENT_TO_ID.items()}
NUM_INTENTS = len(INTENTS)                 # 19
NUM_INTENT_OOS = NUM_INTENTS + 1           # 20 (incl. OOS)

SLOT_TYPES = {
    "TIMER": "duration",
    "ALARM": "time_of_day",
    "TEMPERATURE": "degrees",
    "BRIGHTNESS": "percent",
    "COLOR": "color",
    "CREATE_REMINDER": "reminder_item",
}

# Intent-name aliases accepted by vcm-benchmark / classmates.
_INTENT_ALIASES = {
    "SET_TIMER": "TIMER",
    "SET_ALARM": "ALARM",
    "SET_TEMPERATURE": "TEMPERATURE",
    "SET_BRIGHTNESS": "BRIGHTNESS",
    "SET_COLOR": "COLOR",
    "CREATE_REMINDER": "CREATE_REMINDER",
    "LIGHTS_ON": "LIGHT_ON",
    "LIGHTS_OFF": "LIGHT_OFF",
    "TURN_ON_LIGHTS": "LIGHT_ON",
    "TURN_OFF_LIGHTS": "LIGHT_OFF",
    "GET_WEATHER": "WEATHER",
    "GET_TIME": "TIME",
    "PLAY": "PLAY_MUSIC",
    "UNKNOWN": OUT_OF_SCOPE,
    "NONE": OUT_OF_SCOPE,
    "REJECT": OUT_OF_SCOPE,
    "OOS": OUT_OF_SCOPE,
}


_DIGITS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}
_TEENS = {
    "10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen",
    "14": "fourteen", "15": "fifteen", "16": "sixteen", "17": "seventeen",
    "18": "eighteen", "19": "nineteen",
}
_TENS = {
    "20": "twenty", "30": "thirty", "40": "forty", "50": "fifty",
    "60": "sixty", "70": "seventy", "80": "eighty", "90": "ninety",
}


def _num_to_words(n: int) -> str:
    if n == 0:
        return "zero"
    parts = []
    if n >= 100:
        parts.append(_DIGITS[str(n // 100)] + " hundred")
        n %= 100
    if n >= 20:
        t, o = divmod(n, 10)
        word = _TENS[str(t * 10)]
        if o:
            word += " " + _DIGITS[str(o)]
        parts.append(word)
    elif 10 <= n < 20:
        parts.append(_TEENS[str(n)])
    elif 0 < n < 10:
        parts.append(_DIGITS[str(n)])
    return " ".join(p for p in parts if p)


def normalize_text(text: str) -> str:
    """Lowercase, spell digits, strip punctuation, collapse whitespace."""
    t = (text or "").lower()
    out = []
    i = 0
    while i < len(t):
        ch = t[i]
        if ch.isdigit():
            j = i
            while j < len(t) and t[j].isdigit():
                j += 1
            out.append(" " + _num_to_words(int(t[i:j])))
            i = j
        elif ch.isalpha():
            out.append(ch)
            i += 1
        elif ch == " ":
            out.append(" ")
            i += 1
        else:
            i += 1
    collapsed = []
    prev_space = False
    for ch in out:
        if ch == " ":
            if not prev_space and collapsed:
                collapsed.append(" ")
            prev_space = True
        else:
            collapsed.append(ch)
            prev_space = False
    return "".join(collapsed).strip()


def normalize_slot(slot: Optional[str]) -> str:
    """Normalize slot values for exact-match comparison."""
    if not slot:
        return ""
    s = slot.strip()
    # Canonical alarm times use "6:00 AM"; accept "6 AM" / "6:00AM".
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"(\d)\s*:\s*00\s*(AM|PM)", r"\1:00 \2", s, flags=re.I)
    s = re.sub(r"^(\d)\s*(AM|PM)$", r"\1:00 \2", s, flags=re.I)
    # Color / reminder casing
    if s.lower() in {"red", "blue", "green"}:
        return s.capitalize()
    if s.lower() in {"drink water", "study", "exercise"}:
        return {"drink water": "Drink water", "study": "Study",
                "exercise": "Exercise"}[s.lower()]
    return s


def alias_intent(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    key = str(name).strip().upper().replace("-", "_").replace(" ", "_")
    if key in INTENT_TO_ID:
        return key
    return _INTENT_ALIASES.get(key, key if key in INTENT_TO_ID else None)


@dataclass(frozen=True)
class Variation:
    variation_id: int
    intent: str
    phrase_index: int
    slot_value: str
    phrase: str
    normalized: str

    @property
    def has_slot(self) -> bool:
        return bool(self.slot_value)

    def to_dict(self) -> dict:
        return asdict(self)


def load_variations_csv(path: Optional[Path] = None) -> List[Variation]:
    path = path or VARIATIONS_CSV
    rows: List[Variation] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            intent = row["label"].strip()
            phrase = row["phrase"].strip()
            slot = normalize_slot(row.get("value") or "")
            rows.append(Variation(
                variation_id=i,
                intent=intent,
                phrase_index=int(row["variation"]),
                slot_value=slot,
                phrase=phrase,
                normalized=normalize_text(phrase),
            ))
    if len(rows) != 93:
        raise ValueError(f"Expected 93 variations, got {len(rows)} from {path}")
    intents_seen = sorted({v.intent for v in rows})
    if intents_seen != sorted(INTENTS):
        raise ValueError(f"Intent set mismatch: {intents_seen}")
    return rows


VARIATIONS: List[Variation] = load_variations_csv()
NUM_VARIATIONS = len(VARIATIONS)
VARIATION_BY_PHRASE: Dict[str, Variation] = {v.phrase: v for v in VARIATIONS}
VARIATION_BY_NORM: Dict[str, Variation] = {v.normalized: v for v in VARIATIONS}
PHRASE_TO_VARIATION_ID: Dict[str, int] = {
    v.phrase: v.variation_id for v in VARIATIONS
}

# Per-slot value vocabularies (3 values each).
SLOT_VALUES: Dict[str, List[str]] = {}
for v in VARIATIONS:
    if v.slot_value:
        SLOT_VALUES.setdefault(v.intent, [])
        if v.slot_value not in SLOT_VALUES[v.intent]:
            SLOT_VALUES[v.intent].append(v.slot_value)
SLOT_VALUE_TO_ID: Dict[str, Dict[str, int]] = {
    intent: {val: i for i, val in enumerate(vals)}
    for intent, vals in SLOT_VALUES.items()
}


def lookup_variation(phrase_or_norm: str) -> Optional[Variation]:
    if phrase_or_norm in VARIATION_BY_PHRASE:
        return VARIATION_BY_PHRASE[phrase_or_norm]
    norm = normalize_text(phrase_or_norm)
    return VARIATION_BY_NORM.get(norm)


def variation_to_prediction(v: Optional[Variation], rejected: bool = False) -> dict:
    if rejected or v is None:
        return {
            "intent": OUT_OF_SCOPE,
            "slot": "",
            "variation": OUT_OF_SCOPE,
            "variation_id": -1,
            "rejected": True,
        }
    return {
        "intent": v.intent,
        "slot": v.slot_value,
        "variation": v.phrase,
        "variation_id": v.variation_id,
        "rejected": False,
    }


def schema_fingerprint() -> dict:
    raw = VARIATIONS_CSV.read_bytes()
    return {
        "variations_csv_sha256": hashlib.sha256(raw).hexdigest(),
        "num_intents": NUM_INTENTS,
        "num_intent_oos": NUM_INTENT_OOS,
        "num_variations": NUM_VARIATIONS,
        "slot_intents": SLOT_INTENTS,
        "slot_values": SLOT_VALUES,
        "phrases": [v.phrase for v in VARIATIONS],
    }


def export_grammar_commands_json(out: Path) -> Path:
    """Write deployable grammar list from the canonical schema."""
    data = [
        {"transcript": v.phrase, "intent": v.intent, "slot": v.slot_value,
         "variation_id": v.variation_id}
        for v in VARIATIONS
    ]
    out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    fp = schema_fingerprint()
    print(json.dumps({
        "num_intents": fp["num_intents"],
        "num_variations": fp["num_variations"],
        "sha256": fp["variations_csv_sha256"],
        "slot_values": fp["slot_values"],
        "sample": VARIATIONS[0].to_dict(),
        "call_phrases": [v.phrase for v in VARIATIONS if v.intent == "CALL"],
        "pause_phrases": [v.phrase for v in VARIATIONS if v.intent == "PAUSE"],
    }, indent=2))
