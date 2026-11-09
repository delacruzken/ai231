# Patch: 2026-11-08_phase4-reliable-tools

| Field | Value |
|---|---|
| ID | `2026-11-08_phase4-reliable-tools` |
| Date | 2026-11-08 |
| Phase | 4 |
| Verdict | `improved` |
| Related decisions | #14, #15, #17 |

## Challenge

Phase 3 conversation had no consequential tools. Needed timers/notes with idempotent retries, wall-clock deadlines that survive restart, and spoken notifications that wait while the session is OFF.

## Change

- `me3.tasks/`: `store`, `service`, `scheduler`, `tools`, `sample_bank`
- Migration `002_phase4_tools.sql` — `tool_idempotency`, notes idempotency, notification source link
- Tasks MCP: timer.* and notes.* live; web.* still Phase 5 unsupported
- Conversation LLM tool use (Anthropic tool rounds + MockLLM intent routing)
- Session scheduler: fire due timers → pending notifications; speak when ON; drain on ON
- API `/api/timers`, `/api/notes`, `/api/notifications`; UI timers/notes panel

## Hypothesis

SQLite as source of truth plus idempotency keys prevents duplicate creates on retry; wall-clock deadlines enable restart reconcile without relying on in-memory timers alone.

## How measured

- Host: laptop (macOS)
- `pytest` — 28 passed (idempotency, fire-once, scheduler, mock tool path, MCP)
- `python -m me3.tasks.sample_bank` → `artifacts/tools_phase4/20261108T052007Z/report.json`

## Before / after

- **Before:** Tasks MCP returned unsupported; persona said no timers/notes
- **After:**
  - Idempotent timer/note create (same key → same row)
  - Past-deadline timers fire once on reconcile; short timeout fires
  - MockLLM note tool persists state
  - Notifications remain pending until delivered (OFF-safe)
- **Not claimed:** web search/crawl; OpenCode; live Haiku tool-quality soak; M1A

## Follow-up

Phase 5 Tavily retrieval. Optional: exercise live Haiku tool_use on a scripted timer create when convenient.
