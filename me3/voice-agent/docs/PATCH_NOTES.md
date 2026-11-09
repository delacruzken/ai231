# ME3 patch notes

Living engineering log for challenges, changes, and measured outcomes.  
One file per change under [`patches/`](patches/). Use [`patches/_TEMPLATE.md`](patches/_TEMPLATE.md).

## Process

1. Open a patch entry when starting a non-trivial experiment or fix (not every typo).
2. Fill **Before / after** and **Verdict** before closing the entry.
3. Cite benchmark or soak evidence when a change claims improvement; silence alone is not evidence.
4. Reverts get their own entry — do not delete the original.
5. Phase exit gates may reference patch IDs as supporting evidence.

## Verdicts

`improved` | `regressed` | `neutral` | `inconclusive` | `blocked`

## Index (newest first)

| ID | Phase | Verdict | Summary |
|---|---|---|---|
| [2026-11-09_phase8-benchmark](patches/2026-11-09_phase8-benchmark.md) | 8 | improved | fdbench direct-feed bridge + marks + trial reset; Pipecat 1.15 transport shuttle |
| [2026-11-08_phase7-publication](patches/2026-11-08_phase7-publication.md) | 7 | improved | GitHub Pages publish adapter; sanitize secrets; confirm gate; mock + live PAT path |
| [2026-11-08_phase6-delegation](patches/2026-11-08_phase6-delegation.md) | 6 | improved | One-job OpenCode/mock queue; needs_input UI; cancel/cleanup; no cloud OpenCode key |
| [2026-11-08_phase5-retrieval-profile](patches/2026-11-08_phase5-retrieval-profile.md) | 5 | improved | Tavily evidence envelopes + mock fallback; owner profile; Senate ambiguity cue |
| [2026-11-08_phase4-reliable-tools](patches/2026-11-08_phase4-reliable-tools.md) | 4 | improved | Timers/notes idempotency; restart fire-once; OFF-queued notifications; MCP + LLM tools |
| [2026-11-08_phase3-conversation-haiku](patches/2026-11-08_phase3-conversation-haiku.md) | 3 | improved | Adaptive turns + OverlapPaused; MockLLM + live `claude-haiku-4-5`; local orchestration (Pipecat pin kept) |
| [2026-11-08_phase2-device-session-laptop](patches/2026-11-08_phase2-device-session-laptop.md) | 2 | improved | Laptop duplex + OFF flush/stale reject; session UI; M1A follow-up |
| [2026-11-08_phase1-speech-feasibility](patches/2026-11-08_phase1-speech-feasibility.md) | 1 | improved | Zipformer+Piper+Silero; ASR RTF≈0.076 under TTS on laptop |
| [2026-11-08_phase0-foundation](patches/2026-11-08_phase0-foundation.md) | 0 | neutral | Repo foundation, fd-benchmark rename, MCP/SQLite stubs |
