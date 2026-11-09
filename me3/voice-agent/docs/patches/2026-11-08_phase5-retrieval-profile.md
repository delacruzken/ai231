# Patch: 2026-11-08_phase5-retrieval-profile

| Field | Value |
|---|---|
| ID | `2026-11-08_phase5-retrieval-profile` |
| Date | 2026-11-08 |
| Phase | 5 |
| Verdict | `improved` |
| Related decisions | #14, #17, #18 |

## Challenge

Needed grounded web research with citations, ambiguity handling for Philippine Senate-style questions, and an approved owner public profile that is never biometric identity.

## Change

- `me3.retrieval/`: evidence envelopes, Tavily client (search/extract/crawl), MockRetrieval, sample bank
- `me3.profile/`: approved owner public profile store + persona injection
- Tasks MCP `web.search` / `web.fetch` / `web.crawl` live; LLM tool schemas + MockLLM research routing
- API `GET/PUT /api/profile`; clear `pending.tavily_api_key` / `owner_public_profile` when provisioned
- Persona: cite evidence; clarify nationwide vs local vote leaders when ambiguous

## Hypothesis

Bounded Tavily calls returning URL/title/content envelopes plus explicit ambiguity cues keep answers source-grounded without inventing identity or treating snippets as authority.

## How measured

- Host: laptop (macOS)
- `pytest` — 33 passed
- `python -m me3.retrieval.sample_bank` → `artifacts/retrieval_phase5/` (structural mock + live Tavily)

## Before / after

- **Before:** web.* unsupported; no owner profile injection
- **After:**
  - Structural: mock search/fetch/crawl, profile inject, Senate ambiguity clarify cue — all OK
  - Live Tavily: `live_ok` with real docs.tavily.com / tavily-python results
- **Not claimed:** OpenCode delegation; GitHub publication; M1A; exhaustive election accuracy

## Follow-up

Phase 6 OpenCode delegation. Keep crawl bounds tight (depth≤2, limit≤20) for budget.

### Auth fix (same day)

Live sample testing showed `web.search` / `web.fetch` OK with JSON `api_key`, but `/crawl` returned HTTP 401 until `Authorization: Bearer` was also sent. Client updated; crawl sample (docs.tavily.com, limit=3) OK after fix.
