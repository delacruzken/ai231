# Patch: 2026-11-19_turn-endpoint-patience

| Field | Value |
|---|---|
| ID | `2026-11-19_turn-endpoint-patience` |
| Date | 2026-11-19 |
| Phase | 2 / conversation |
| Verdict | `improved` (unit); live M1A retest pending |
| Related decisions | #11 |

## Challenge

Live M1A turns cut mid-phrase (`What is the time to`, `Who is the Pre`, `Do you know the G-C`) because silence commit was ~400 ms and incomplete heuristics missed truncated / open-question tails; sherpa endpoint defaults also finalized early.

## Change

- Silence defaults → 700 / 1500 / 2400 ms (normal / incomplete / clarify)
- Incomplete heuristic: open stems without `?`, truncated tails (`Pre`, `G-C`), richer function-word endings
- Sherpa endpoint trailing silence → rule1 2.8 s, rule2 1.8 s (configurable)

## Hypothesis

Waiting longer on unfinished-looking partials lets ASR catch the rest of the utterance before speculative commit.

## How measured

Unit: `tests/test_conversation_unit.py` truncated/open-stem cases. Live retest after server restart.

## Before / after

- **Before:** commit at ~0.4 s on “Who is the Pre”
- **After (unit):** same text held until ≥1.5 s silence

## Follow-up

Keep. If replies feel sluggish on short clear questions with terminal `?`, trim `silence_normal_ms` only.
