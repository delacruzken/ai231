# ME3 decision register (compact)

Source of truth for rationale: [`../ME3_Architecture_and_Development_Plan.md`](../../ME3_Architecture_and_Development_Plan.md) §3.  
When a choice is revisited, open a patch note and reference the decision ID here.

| # | Choice | Status |
|---|---|---|
| 1 | Local pretrained speech; cloud LLM | agreed |
| 2 | sherpa-onnx streaming Zipformer en 2023-02-21 INT8 | verified laptop (Phase 1); Pi pending |
| 3 | English first | agreed |
| 4 | Piper en_US-lessac-medium via piper-tts 1.8.0 (same ONNX; not sherpa OfflineTts on v2) | verified laptop (Phase 1); Pi pending |
| 5 | EMEET M1A I/O | pending hardware — laptop default devices used in Phase 2; hint swap later |
| 6 | M1A AEC first; no software AEC yet | agreed (laptop echo expected / not mitigated) |
| 7 | Silero ONNX VAD (in-process; vendored from fd-benchmark) | verified laptop (Phase 1) |
| 8 | UI session ON/OFF toggle | verified laptop (Phase 2) |
| 9 | Pipecat 1.15.0 pin | verified laptop Phase 8 — transport shuttle for fdbench only; speech still local SessionController |
| 10 | Claude Haiku API | verified laptop — live model id `claude-haiku-4-5` (settings unchanged) |
| 11 | Adaptive silence + transcript heuristics | verified sample bank (600/1200/2000 + incomplete heuristic) |
| 12 | Pause overlap, classify, resume or yield | verified sample bank |
| 13 | Event-driven acks; sparse backchannels | verified unit/sample bank; backchannels flag default off |
| 14 | Four MCP services | agreed — Tasks + Delegation tools live |
| 15 | Timers + persistent notes; to-dos later | verified Phase 4 (idempotency + restart/timeout sample bank) |
| 16 | OpenCode, one active job | verified mock queue; live OpenCode when `opencode serve` healthy |
| 17 | SQLite state | verified — sessions/turns/timers/notes/notifications; thread-safe |
| 18 | Concise friendly task persona | verified — `me3.conversation.persona` |
| 19 | Bounded recovery; no second provider initially | agreed |
| 20 | Native systemd | agreed |
| 21 | Colleague fd-benchmark + supplementary tests | Phase 8 entry + marks wired; full scored runs follow |
| 22 | Demo/deck planning deferred | agreed |

Additional: Tavily retrieval verified laptop (Phase 5, mock + live); approved owner public profile store; owner GitHub via PAT + Pages `docs/` publish (Phase 7); minimal visual UI; patch-notes practice (see `PATCH_NOTES.md`).
