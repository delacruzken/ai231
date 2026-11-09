# Patch: 2026-11-08_phase3-conversation-haiku

| Field | Value |
|---|---|
| ID | `2026-11-08_phase3-conversation-haiku` |
| Date | 2026-11-08 |
| Phase | 3 |
| Verdict | `improved` |
| Related decisions | #9, #10, #11, #12, #13, #17, #18 |

## Challenge

Phase 2 had duplex session ON/OFF without turn-taking, LLM replies, or overlap handling. Needed adaptive commits, Thinking/OverlapPaused, cancel paths, and a Haiku (or MockLLM) conversation loop validated on a sample bank.

## Change

- Package `me3.conversation/`: `pipeline`, `llm` (Anthropic streaming + MockLLM), `persona`, `turn_taker`, `overlap`, `acks`, `memory`, `sample_bank`
- Session states: Off / Listening / **Thinking** / Speaking / **OverlapPaused**; UI shows status + user/assistant transcript
- Adaptive silence 600 / 1200 / ~2000 ms from last speech audio; incomplete-transcript heuristics; cancel pending commit on resumed speech
- Overlap pause → classify → resume (backchannel/false trigger) or yield (steering/timeout); event-driven ack (~700 ms); `backchannel_enabled` default false, cooldown ≥10 s
- SQLite turn logging with thread-safe connection (`check_same_thread=False` + RLock)
- Settings: `models.llm: claude-haiku-4-5`; clear `pending.anthropic_api_key` when key present
- **Orchestration choice:** local `me3.conversation` + `SessionController`; `pipecat-ai==1.15.0` remains pinned but Phase 3 does **not** route PCM through Pipecat transports (custom-transport friction avoided; pin kept without silent abandon)

## Hypothesis

In-process turn heuristics + Haiku streaming into existing ASR/TTS/session stack is enough to pass pause/correction/interrupt sample-bank gates before tools (Phase 4).

## How measured

- Host: laptop (macOS)
- Structural: `python -m me3.conversation.sample_bank` (MockLLM)
- Live: same runner with `ANTHROPIC_API_KEY` set
- Unit: `pytest` (21 passed)
- Evidence: `artifacts/conversation_phase3/20261108T045818Z/report.json`

## Before / after

- **Before:** no LLM path; states Off/Listening/Speaking only
- **After:**
  - Structural sample bank all cases OK (pause 600/1200, cancel-on-resume, overlap resume/yield, ack suppress, mock cancel through Thinking→Speaking→Listening)
  - Live Haiku OK — model id **`claude-haiku-4-5`**, reply preview `hello from haiku`
  - MockLLM path green without relying on live quality for structural cancel
- **Not claimed:** tool execution; M1A full-duplex quality; Taglish; Pi soak

## Follow-up

Phase 4 timers/notes tools. Keep Pipecat pin until a transport integration has measured benefit. M1A still Phase 2 follow-up.
