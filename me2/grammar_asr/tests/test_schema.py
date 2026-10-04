from grammar_asr.schema import (
    VARIATIONS, NUM_VARIATIONS, NUM_INTENTS, NUM_INTENT_OOS,
    FIXED_INTENTS, SLOT_INTENTS, lookup_variation, normalize_text,
    schema_fingerprint, OUT_OF_SCOPE,
)


def test_counts():
    assert NUM_VARIATIONS == 93
    assert NUM_INTENTS == 19
    assert NUM_INTENT_OOS == 20
    assert len(FIXED_INTENTS) == 13
    assert len(SLOT_INTENTS) == 6
    assert len(VARIATIONS) == 93


def test_canonical_phrases():
    assert lookup_variation("Make a call") is not None
    assert lookup_variation("Pause song") is not None
    # Obsolete Option-B wording must not be canonical
    assert lookup_variation("Place a call") is None
    assert lookup_variation("Pause for now") is None


def test_alarm_normalization():
    v = lookup_variation("Set an alarm for 6:00 AM")
    assert v is not None
    assert v.intent == "ALARM"
    assert v.slot_value == "6:00 AM"
    assert normalize_text("What's the weather?") == "whats the weather"


def test_fingerprint_stable():
    fp = schema_fingerprint()
    assert len(fp["variations_csv_sha256"]) == 64
    assert OUT_OF_SCOPE == "OUT_OF_SCOPE"
