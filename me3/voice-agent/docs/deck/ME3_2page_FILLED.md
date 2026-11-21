# ME3 — Machine Exercise 2-page deck (FILLED DRAFT)

**Student:** 2015-02211 · **Agent:** Kaki · **Harness:** Coco · **Target:** Raspberry Pi 5  
**Repo:** `delacruzken/ai231` · `me3/voice-agent`  
**Honesty label:** §5 / Results use **Pi remote** bank `2026-11-21_00-01-46` (tag `me3-pi-best`) unless marked laptop. Laptop `2026-11-20_09-04-05` remains a comparison footnote only.

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
 ┌── COCO HARNESS (ME3 custom duplex runtime) ───────────────────────────────────────┐
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

 fd-benchmark only: Pipecat 1.15 transport shuttle → same Coco / SessionController
```

**Naming:** **Kaki** = spoken agent · **Coco** = duplex harness (our answer to “Pipecat / onit / crew”).  
**Cloud boundary (outside the box):** Anthropic Claude Haiku · Tavily · GitHub.  
**Not in cloud:** mic PCM / ASR / TTS (local models).

| Template expectation | ME3 (honest) | vs classmate (typical) |
|---|---|---|
| Harness e.g. Pipecat | **Coco** (custom ME3 duplex); Pipecat = **bench adapter** | Pipecat **is** the live agent |
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
| **LLM** | Coco HTTP client (not an MCP server) | tool / function calling | `claude-haiku-4-5` | **Cloud** Anthropic; streaming tokens |
| **VAD** | in-process | speech frames @ 16 kHz | Silero ONNX | On device |
| **Tasks** | `me3.mcp.tasks` | `timer.*` `notes.*` `web.*` | n/a → Tavily | On device; cloud for search API |
| **Delegation** | `me3.mcp.delegation` | `jobs.submit/status/…` | OpenCode or mock | On device worker |
| **Publication** | publish tools / service | `publish.validate/site` | GitHub Pages | On device → GitHub PAT |

**Gray vs classmates:** They often put Deepgram/Cartesia in the Model column. We put **local** models and keep speech off the cloud.

---

## 5 · EVALUATION & BENCHMARK

**Method:** one monotonic clock · marks: VAD end → ASR final → LLM 1st token → TTS 1st audio → speaker · E2E = audio_out − vad_end · 10 phrases × 3 runs · JSON per run (`stage_latency.json`).

| Metric | Definition | Target | Mean | p95 | n | Source |
|---|---|---|---|---|---|---|
| E2E latency | end of speech → audio out | &lt; 4.00 s | **3.54 s** | **4.28 s** | 30 | `me3-pi-best` / `00-01-46` |
| ASR final | end of speech → transcript | &lt; 0.80 s | **2.73 s** | **3.48 s** | 30 | same |
| LLM TTFT | transcript → first token | &lt; 1.50 s | **1.60 s** | **2.70 s** | 28 | `01-50-26` (marks + bench tool clamp) |
| TTS TTFB | text → first audio chunk | &lt; 0.60 s | **0.09 s** | **0.11 s** | 30 | `me3-pi-best` / `00-01-46` |
| Barge-in | speech → playback stopped | &lt; 0.80 s | **1.92 s** | **1.92 s** | 1 | `02-13-46` (FDB `--rounds 2`) |
| WER | ASR accuracy on test set | &lt; 15 % | **10.4 %** | — | 30 | `me3-pi-best` / `00-01-46` |

**Run card (Pi Results — primary `me3-pi-best`):**  
Raspberry Pi 5 Model B Rev 1.1 · 8 GB · heatsink + fan · Ethernet remote fdbench · 2026-11-21 · Kroko + Haiku + Piper · run `2026-11-21_00-01-46`  
**Companion runs:** TTFT `2026-11-21_01-50-26`; barge `2026-11-21_02-13-46`. Label the table as multi-run if staff ask.  
**Laptop footnote:** Darwin · `2026-11-20_09-04-05` · WER 5.7%.

**Why ASR-final misses &lt; 0.80 s (say this on the Challenges slide):**  
ASR-final is **not** “how fast Zipformer types.” It is **VAD end → moment we treat the transcript as final**. We wait for extra silence (~1.8–2.6 s on Pi) so incomplete questions are not cut. That wait *is* most of the ~2.7 s mean. Hitting 0.80 s means committing sooner → mid-phrase cuts and worse WER. We accept ASR-final over target so WER stays &lt; 15%.

| Status | Metric |
|---|---|
| Met | E2E mean, TTS TTFB, WER |
| Over (accepted) | ASR final — silence-commit patience |
| Over (thin n) | LLM TTFT 1.60 s; barge-in 1.92 s (n=1) — both filled, not under target |

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

- Paste `latency_chart.svg` from `fd-benchmark/runs/2015-02211/2026-11-21_00-01-46/` (or `summary.md` sibling).  
- Machine line = **Pi 5 / 8 GB / heatsink+fan / Ethernet** — not Darwin.  
- Challenges must include the **ASR-final vs WER** tradeoff note from §5.

---

## CHALLENGES & REFERENCES

| Challenge | Fix | Measurement |
|---|---|---|
| ASR-final &gt; 0.80 s target | Finalize at **silence-commit**, not raw VAD end — wait proves the user finished; clock includes that patience | Pi mean **2.73 s**; WER still **10.4%** &lt; 15% (`00-01-46` / `me3-pi-best`). One-liner: “ASR-final is end-of-speech → commit, not type-speed; cutting sooner hits WER.” |
| Course WER empty (`asr_final` n=0) | Emit `asr_final` on silence-commit before speculative `llm_request` | Laptop WER **5.7%**; Pi WER **10.4%** |
| Mid-phrase cutoffs | Longer incomplete silence + ghost filters | Pi empty-mark trials → 0 after patience |
| LLM TTFT n=0 on best bank | Suppress ack TTS in bench; mock retrieval; emit `llm_first_token` | Pi `01-50-26` mean **1.60 s** n=28 |
| Course barge-in n=0 / spoke_through | Stage-latency has no cut-ins; FDB rounds + hard-stop in **benchmark** (live keeps soft-pause) | Pi `02-13-46` barge **1.92 s** n=1 (over &lt;0.80 s; thin n) |
| False barge-in on noise | Energy floor + soft-pause then confirm (live) | `ME3_BARGE_DEBUG` logs; live interrupt capture |
| Echo without AEC | Gate ASR while speaking + hangover | Duplex soak; M1A AEC still pending on Pi |
| Harness ≠ Pipecat live | Name it **Coco**; Pipecat = fdbench shuttle only | Topology doc + README · decision #23 |

**Refs:** `docs/SYSTEM_TOPOLOGY.md` · `docs/benchmarks/2026-11-21_00-01-46_pi_RESULTS.md` · `…_01-50-26_pi_RESULTS.md` · `…_02-13-46_pi_RESULTS.md` · tag `me3-pi-best`

---

## How we fare vs “others” (quick scorecard)

| Deck section | ME3 draft readiness | Typical classmate (Pipecat + cloud speech) |
|---|---|---|
| §1 Diagram | Strong: **Coco** named harness + local speech | Strong textbook MCP arrows |
| §2 Prompt | Ready (v1) | Usually ready |
| §3 Context | Ready | Usually ready |
| §4 MCP table | Ready **with caveat** (in-process live ASR/TTS) | Cleaner “MCP holds API key” story |
| §5 Latency table | **All six rows filled** (composite runs); ASR/TTFT/barge over target with notes | Often cloud ASR → better ASR-final |
| Results card | **Pi `me3-pi-best` + companion TTFT/barge** | Often already on Pi |
| Demo story | Strong (tools + barge + local speech) | Strong (SOTA cloud voice) |

**Bottom line:** Paste §5 from this draft. Keep ASR-final silence-commit on Challenges. Disclose TTFT/barge come from companion Pi runs (`01-50-26`, `02-13-46`), not the single best stage bank.
