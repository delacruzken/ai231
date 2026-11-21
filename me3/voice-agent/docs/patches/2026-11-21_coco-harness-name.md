# Patch: 2026-11-21_coco-harness-name

| Field | Value |
|---|---|
| ID | `2026-11-21_coco-harness-name` |
| Date | 2026-11-21 |
| Phase | docs / deck |
| Verdict | `neutral` (naming) |
| Related decisions | #9, #23 |

## Challenge

Deck/template wording “harness” was easy to confuse with the class name `SessionController` or with classmates’ Pipecat-as-harness story.

## Change

- Product-name the live duplex runtime **Coco** (persona remains **Kaki**).
- Update deck draft, topology, narrative, README, Gap 2 line, fdbench model-card harness field.
- Keep Python class `SessionController` (no mass rename).

## How measured

Docs/deck consistency only — no latency claim.
