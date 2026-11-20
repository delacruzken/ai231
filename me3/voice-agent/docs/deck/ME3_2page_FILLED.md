# ME3 — Machine Exercise 2-page deck (FILLED DRAFT)

**Student:** 2015-02211 · **Agent:** Kaki · **Target:** Raspberry Pi 5  
**Repo:** `delacruzken/ai231` · `me3/voice-agent`  
**Honesty label:** Numbers below are **laptop / fdbench direct-feed** (`2026-11-20_09-04-05`) unless marked. Re-measure on Pi before calling them final Results.

Use this to compare against classmate decks (e.g. Pipecat + cloud ASR/TTS). Gray = still weak vs template.

---

# PAGE 1 — System, prompt, context

## 1 · SYSTEM DIAGRAM

```
                    ┌────────────── ON DEVICE (Pi 5 target / laptop today) ──────────────┐
 USB mic ──► Silero VAD (in-process)
                │
                ▼
         Local ASR engine ◄──── same code as me3.mcp.asr (WS server optional)
         (Kroko Zipformer)         streaming partials → silence-commit final
                │
                ▼
 ┌── ME3 CUSTOM HARNESS ─────────────────────────────────────────────────────────────┐
 │  SessionController + ConversationOrchestrator + UI :8780                          │
 │  turn-taking · echo gate · two-phase barge-in · tool dispatch                     │
 │         │                                                                         │
 │         ├──► Tasks MCP-shaped     (timers, notes, web_search/crawl → Tavily)      │
 │         ├──► Delegation           (jobs_* → OpenCode or mock)                     │
 │         └──► Publication          (publish_* → GitHub Pages)                      │
 │         │                                                                         │
 │         ▼                                                                         │
 │  Local TTS engine ◄──── same code as me3.mcp.tts (WS server optional)             │
 │  (Piper en_US-lessac-medium) · cancel on barge-in                                 │
 └───────────────────────────────────────────────────────────────────────────────────┘
                │
                ▼
            USB speaker

 Barge-in: speech (energy+VAD) → soft-pause TTS → confirm → cancel LLM + listen
           (interrupt audio kept as next user turn)

 fd-benchmark only: Pipecat 1.15 transport shuttle → same SessionController
```

**Cloud boundary (outside the box):** Anthropic Claude Haiku · Tavily · GitHub.  
**Not in cloud:** mic PCM / ASR / TTS (local models).

| Template expectation | ME3 (honest) | vs classmate (typical) |
|---|---|---|
| Harness e.g. Pipecat | **Custom ME3 harness**; Pipecat = **bench adapter** | Pipecat **is** the live agent |
| ASR/TTS MCP on path | Engines **in-process**; MCP/WS = **same code, optional process** | `voice-listen` / `voice-speak` MCP → **cloud** APIs |
| Barge-in stops TTS+LLM | Yes (two-phase confirm) | Yes (often VAD → hard stop) |

**One-liner flow:** `voice in → VAD → ASR → LLM(+tools) → TTS → voice out`  
**Barge-in:** `speech → stop TTS + cancel LLM → listen`

---

## 2 · PROMPT — SYSTEM PROMPT (v1) excerpt

*Repo: `docs/PROMPT_v1.md` · code: `me3.conversation.persona` (`PROMPT_VERSION = "v1"`)*

```
You are Kaki, a capable voice assistant for the session owner.
Keep every reply very short: prefer one sentence, two only if essential.
Never use markdown — the reply is read aloud by text-to-speech.
If the user speaks while you are talking, stop and listen (barge-in).
Confirm before costly or irreversible actions (e.g. publish).
Use tools for world facts (web_search), timers/notes, coding jobs, and publish.
Never invent credentials or private data.
```

---

## 3 · CONTEXT ENGINEERING

| GOES IN | MANAGED |
|---|---|
| System prompt v1 + persona | Short spoken turns; no markdown/URLs aloud |
| ASR partials → silence-commit / finals | `max_context_turns` = 12; duplicate-commit window |
| VAD / turn / barge-in state | idle → listen → think → speak |
| Tool results (timers, notes, web, jobs, publish) | World facts **must** `web_search` before answering |
| Owner public profile (approved fields only) | On barge-in: flush TTS; drop stale assistant audio |
| Runtime clock (Asia/Manila) | Log `messages` / `chars` on `llm_request` marks |

---

# PAGE 2 — MCP, evaluation, demo / results / challenges

## 4 · MCP SERVICES + model serving

| Component | MCP server / transport | Tools | Model | Served where / notes |
|---|---|---|---|---|
| **ASR** | `me3.mcp.asr` / loopback WS (+ **in-process** live path) | session + streaming PCM → partial/final | Kroko Zipformer `…-2025-08-06` | **On device**; final at **silence-commit** (not raw VAD end) |
| **TTS** | `me3.mcp.tts` / loopback WS (+ **in-process** live path) | speak / stream / cancel | Piper `en_US-lessac-medium` | **On device**; cancel on barge-in |
| **LLM** | Harness HTTP client (not an MCP server) | tool / function calling | `claude-haiku-4-5` | **Cloud** Anthropic; streaming tokens |
| **VAD** | in-process | speech frames @ 16 kHz | Silero ONNX | On device |
| **Tasks** | `me3.mcp.tasks` | `timer.*` `notes.*` `web.*` | n/a → Tavily | On device; cloud for search API |
| **Delegation** | `me3.mcp.delegation` | `jobs.submit/status/…` | OpenCode or mock | On device worker |
| **Publication** | publish tools / service | `publish.validate/site` | GitHub Pages | On device → GitHub PAT |

**Gray vs classmates:** They often put Deepgram/Cartesia in the Model column. We put **local** models and keep speech off the cloud.

---

## 5 · EVALUATION & BENCHMARK

**Method:** one monotonic clock · marks: VAD end → ASR final → LLM 1st token → TTS 1st audio → speaker · E2E = audio_out − vad_end · 10 phrases × 3 runs · JSON per run (`stage_latency.json`).

| Metric | Definition | Target | Mean | p95 | n |
|---|---|---|---|---|---|
| E2E latency | end of speech → audio out | &lt; 4.00 s | **2.80 s** | **3.81 s** | 20 |
| ASR final | end of speech → transcript | &lt; 0.80 s | **1.96 s** | **2.95 s** | 20 |
| LLM TTFT | transcript → first token | &lt; 1.50 s | **1.40 s** | **2.39 s** | 6 |
| TTS TTFB | text → first audio chunk | &lt; 0.60 s | **0.09 s** | **0.24 s** | 19 |
| Barge-in | speech → playback stopped | &lt; 0.80 s | — | — | **0** |
| WER | ASR accuracy on test set | &lt; 15 % | **5.7 %** | — | 20 |

**Run card (laptop — not final deck Results):**  
Darwin 27.0.0 · 16 GB · direct-feed · 2026-11-20 · Kroko + Haiku + Piper · run `2026-11-20_09-04-05`  
**Pi Results:** *pending* (Gap 3).

| Status | Metric |
|---|---|
| Met | E2E, TTS TTFB, WER |
| Over | ASR final (by design: silence patience), LLM TTFT p95 |
| Empty | Barge-in (stage-latency bank has no cut-in trials) |

---

## DEMO (deck 2) — suggested 2–3 min script

1. Session ON → short greeting (no feature laundry list).  
2. Natural Q&A (e.g. current fact → must call `web_search`).  
3. Timer or note (live task tool).  
4. Barge-in mid-answer → agent stops, answers the interrupt.  
5. Optional: coding job / publish preview (mock OpenCode if 401).  
6. Mention recovery: tool/API slow → short spoken status, no invented success.

---

## RESULTS (deck 2) — fill from table above

- Paste `latency_chart.svg` from the run folder (laptop for now).  
- **Replace machine line with Pi model/RAM/cooling when Gap 3 closes.**  
- Do **not** label Darwin numbers as “Raspberry Pi Results.”

---

## CHALLENGES & REFERENCES

| Challenge | Fix | Measurement |
|---|---|---|
| Course WER empty (`asr_final` n=0) | Emit `asr_final` on silence-commit before speculative `llm_request` | Laptop bank WER **5.7%** n=20 (`2026-11-20_09-04-05`) |
| Mid-phrase cutoffs | Longer incomplete silence + ghost filters | Live M1A/laptop listen + turn sample bank |
| False barge-in on noise | Energy floor + soft-pause then confirm | `ME3_BARGE_DEBUG` logs; live interrupt capture |
| Echo on laptop without AEC | Gate ASR while speaking + hangover | Duplex soak; M1A AEC still pending on Pi |
| Harness ≠ Pipecat live | Honest diagram: custom runtime; Pipecat = fdbench shuttle | Topology doc + README |

**Refs:** modelcontextprotocol.io · sherpa-onnx / Piper · fd-benchmark (class harness) · repo `me3/voice-agent` · `docs/SYSTEM_TOPOLOGY.md` · `docs/benchmarks/2026-11-20_09-04-05_RESULTS.md`

---

## How we fare vs “others” (quick scorecard)

| Deck section | ME3 draft readiness | Typical classmate (Pipecat + cloud speech) |
|---|---|---|
| §1 Diagram | Strong if we keep **honest** custom harness + local speech | Strong textbook MCP arrows |
| §2 Prompt | Ready (v1) | Usually ready |
| §3 Context | Ready | Usually ready |
| §4 MCP table | Ready **with caveat** (in-process live ASR/TTS) | Cleaner “MCP holds API key” story |
| §5 Latency table | **Numbers exist**; ASR-final over; barge-in empty | Often cloud ASR → better ASR-final; still need barge suite |
| Results card | **Laptop only** — behind until Pi | Often already on Pi |
| Demo story | Strong (tools + barge + local speech) | Strong (SOTA cloud voice) |

**Bottom line for you:** Content-wise you can fill the 2-pager today. Politically/visually, classmates look closer to the template’s “Pipecat + ASR/TTS MCP” cartoon; you look stronger on **local speech + WER proof**, weaker on **named harness + Pi Results + barge-in row**.
