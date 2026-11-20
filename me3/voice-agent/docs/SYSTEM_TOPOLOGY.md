# ME3 system topology (deck §1 source of truth)

**Design lock (2026-11-20):** Gap 1 **Option B — Hybrid**; Gap 2 **Option A — Honest harness label**.  
Local ASR/TTS (Gap 4 closed). Do not draw Deepgram/Cartesia-style cloud speech.

## On device (Raspberry Pi 5 — target; laptop for now)

```
USB mic/speaker (M1A when available)
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│  ME3 session runtime (custom)                             │
│  SessionController · ConversationOrchestrator · UI :8780  │
│                                                           │
│   Silero VAD (in-process)                                 │
│        │                                                  │
│        ▼                                                  │
│   StreamingAsrEngine  ←── same code as me3.mcp.asr        │
│   (Kroko Zipformer, local)                                │
│        │                                                  │
│        ▼                                                  │
│   LLM client (Haiku) ──tools──► Task / Delegation /       │
│        │                        Publication services      │
│        │                        (MCP-shaped tool APIs)    │
│        ▼                                                  │
│   PiperTtsEngine  ←── same code as me3.mcp.tts            │
│        │                                                  │
│        ▼                                                  │
│   Playback (+ echo gate / barge-in)                       │
└───────────────────────────────────────────────────────────┘
        │
        ▼
USB speaker

Barge-in: user speech (energy+VAD) → soft-pause TTS → confirm →
          cancel LLM + keep interrupt as next ASR turn
```

### Optional speech MCP servers (same engines)

```bash
python -m me3.mcp.asr.server   # loopback WS PCM → Zipformer
python -m me3.mcp.tts.server   # loopback WS ← Piper
```

Live `--serve` and fdbench use **in-process** engines for latency. MCP/WS servers expose the **same** ASR/TTS code for interop and “we built the servers” evidence. Diagram wording: **“local ASR/TTS engines (also packaged as MCP/WS services)”** — not “every PCM chunk crosses MCP in the demo path.”

## Cloud

| Service | Role |
|---|---|
| Anthropic Claude Haiku | Conversation + tool calling |
| Tavily | Web search / extract (via Tasks) |
| GitHub | Pages publish (via Publication; PAT on device) |

Raw microphone audio is **not** sent to cloud speech APIs.

## Harness / Pipecat (Gap 2)

- **Live duplex:** ME3 custom session runtime (not a full Pipecat speech pipeline).
- **fd-benchmark:** Pipecat 1.15 transport shuttle → same `SessionController` (`me3.benchmark.bridge`).
- Deck line: *“Custom ME3 duplex runtime; Pipecat used as the fdbench audio transport adapter.”*

## MCP services (deck §4 rows)

| Component | Server / surface | Model / notes | Where |
|---|---|---|---|
| ASR | `me3.mcp.asr` (+ in-process engine) | Kroko Zipformer streaming | On device |
| TTS | `me3.mcp.tts` (+ in-process engine) | Piper `en_US-lessac-medium` | On device |
| VAD | in-process | Silero ONNX | On device |
| Tasks | `me3.mcp.tasks` | timers, notes, web_* | On device → Tavily |
| Delegation | `me3.mcp.delegation` | OpenCode / mock one-job | On device |
| Publication | service + tools | GitHub Pages | On device → GitHub |

## Benchmarks (Gap 3)

| Tier | Rig | Use |
|---|---|---|
| Laptop / direct | macOS, fdbench `--connect direct` | Method + regression (labeled) |
| Pi / direct or remote | RPi monotonic + resources | Deck Results machine line |
| Pi / M1A or cables | Physical audio | Barge-in + AEC acceptance |
