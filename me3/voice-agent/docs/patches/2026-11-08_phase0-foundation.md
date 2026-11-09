# Patch: 2026-11-08_phase0-foundation

| Field | Value |
|---|---|
| ID | `2026-11-08_phase0-foundation` |
| Date | 2026-11-08 |
| Phase | 0 |
| Verdict | `neutral` |
| Related decisions | #14, #17, #20, #21 |

## Challenge

No application repository existed under `me3/` — only the architecture plan and the cloned benchmark checkout. Development needed a reproducible foundation before speech or device work.

## Change

- Renamed `me3/ai231-me3-full-duplex-benchmarking` → `me3/fd-benchmark` (pin unchanged: `57eaddbc188ce9a19b8fd002b575f703426956f4`)
- Created `me3/voice-agent/` Phase 0 skeleton: config schema, SQLite migration, four MCP stubs, app entry stub, systemd examples, patch-notes system
- Documented patch-notes practice in architecture §10

## Hypothesis

A locked skeleton with explicit pending settings and an engineering log will keep later phases auditable and prevent secret/config drift.

## How measured

Layout and smoke import checks on the development laptop; `git rev-parse HEAD` on `fd-benchmark`; pytest smoke suite.

## Before / after

- **Before:** `me3/` contained only the plan + long-named benchmark clone
- **After:** `me3/voice-agent/` importable; migrations apply; MCP stubs return `unsupported`; patch index seeded
- **Evidence (laptop, warm):** `pytest` → **6 passed**; `python -m me3.app.main` exits 0; `fd-benchmark` HEAD = `57eaddbc188ce9a19b8fd002b575f703426956f4`; `pipecat-ai==1.15.0` installed successfully into `.venv`

## Follow-up

Keep. Next: Phase 1 local speech feasibility (Zipformer, Piper, Silero) with measured CPU/RAM under concurrent load.
