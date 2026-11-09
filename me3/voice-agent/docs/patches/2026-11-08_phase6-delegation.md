# Patch: 2026-11-08_phase6-delegation

| Field | Value |
|---|---|
| ID | `2026-11-08_phase6-delegation` |
| Date | 2026-11-08 |
| Phase | 6 |
| Verdict | `improved` |
| Related decisions | #14, #16, #17 |

## Challenge

Need a coding worker that does not block voice: one active job, progress/needs-input/approval UI, and cleanup — without requiring a cloud “OpenCode API key.”

## Change

- `me3.delegation/`: job registry, MockWorker, OpenCode HTTP client, DelegationService, sample bank
- MCP `jobs.submit/status/update/cancel/result`
- LLM tools + MockLLM job routing; persona updated
- Session wires delegation events; UI job cards with approve/cancel
- Env: `OPENCODE_URL`, `OPENCODE_SERVER_USERNAME`/`PASSWORD`, `OPENCODE_FORCE_MOCK` (no cloud key)

## Hypothesis

Local `opencode serve` (optional basic auth) plus MockWorker fallback proves queue/update/cleanup gates; voice path stays on the session pump while the worker runs off-thread.

## How measured

- Host: laptop (macOS); OpenCode not running → MockWorker
- `pytest` — 38 passed
- `python -m me3.delegation.sample_bank` → structural_ok; opencode probe connection refused (expected)

## Before / after

- **Before:** Delegation MCP unsupported stub
- **After:** one-job enforce, needs_input→respond→complete, cancel+cleanup, UI/API job controls; OpenCode adapter ready when server healthy
- **Not claimed:** live OpenCode end-to-end coding success; GitHub Pages publish

## Keys / setup

- No OpenCode cloud API key
- Optional: run `opencode serve --port 4096`; set `OPENCODE_SERVER_PASSWORD` if the server requires basic auth
- OpenCode itself still needs its own LLM provider credentials (often Anthropic) in the OpenCode process env

## Follow-up

Phase 7 publication (GitHub PAT). Point `OPENCODE_URL` at a live serve instance for a live-worker patch.

### OpenCode v2 API note

OpenCode 2.x serves a web UI at `/` (HTML). JSON APIs live under `/api/*` (e.g. `/api/session`). There is no `/global/health` on v2 — ME3 probes `GET /api/session` with basic auth instead. Client updated accordingly.
