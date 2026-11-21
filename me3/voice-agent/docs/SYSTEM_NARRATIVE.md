# ME3 system narrative — how to explain the design

Use this when walking someone through the build. Start with the **elevator story**, then the **happy-path turn**, then **barge-in**, then **tools / cloud**, then **benchmark**. Each step has a *high-level* line (what you’d say out loud) and a *low-level* note (what the code actually does).

**Cast of characters**

| Name | Role in one sentence |
|---|---|
| **Session UI** (`:8780`) | Laptop browser ON/OFF + transcript; not the brain |
| **Coco** (`SessionController`) | Our **named duplex harness** — mic pump, states, echo gate, barge-in, ASR/TTS engines |
| **ConversationOrchestrator** | Turn-taking, LLM call, tools, when to speak |
| **Silero VAD** | “Is this frame speech?” |
| **Kroko Zipformer (ASR)** | Streaming speech → text (local) |
| **Claude Haiku (LLM)** | Reason + choose tools (cloud) |
| **Piper (TTS)** | Text → audio (local) |
| **Tasks / Delegation / Publication** | MCP-shaped tools (timers, web, jobs, GitHub Pages) |
| **Pipecat** | Only for **fdbench** audio shuttle — not the live harness |

---

## 0 · Elevator story (30 seconds)

> We built a **full-duplex voice agent** called **Kaki** on a custom harness we call **Coco** (`SessionController`): it reads the mic, runs local VAD + ASR, decides when the user finished speaking, calls Claude Haiku with tools, speaks back with local Piper TTS, and can be interrupted mid-sentence. Speech stays on the device; only the LLM and a few APIs (Tavily, GitHub) go to the cloud. **Pipecat** is installed mainly so the **class benchmark** can inject audio and collect stage marks — classmates often use Pipecat as the whole agent; we use it as a transport adapter into Coco.

---

## 1 · Boot and session ON

### High-level

You start the backend, open the UI, flip the session **ON**. The agent goes to **listening**.

### Low-level

1. `python -m me3.app.main --serve` starts FastAPI + WebSocket UI and constructs a `SessionController`.
2. Capture/playback devices open (laptop default or M1A hints); ASR/TTS/VAD engines load **in-process**.
3. A background **pump loop** continuously reads 16 kHz mono chunks from the mic.
4. UI `ON` → state `LISTENING`; OFF flushes playback and stops treating mic as conversation.

**States to remember:** `OFF → LISTENING → THINKING → SPEAKING` (plus overlap/barge paths).

---

## 2 · Happy path — one user turn (step by step)

```
Mic → [resample] → VAD + ASR → silence commit → LLM (+tools) → TTS → Speaker
         ▲                                              │
         └──────── echo gate while speaking ────────────┘
```

### Step A — Capture audio

| | |
|---|---|
| **High** | Mic audio is normalized to what ASR expects. |
| **Low** | Device may be 48 kHz; we resample to **16 kHz mono float32**. Chunks are small (often <512 samples after resample). Soft **AGC** can boost quiet levels for ASR only (doesn’t change barge energy path the same way). |

### Step B — While listening: feed ASR + VAD

| | |
|---|---|
| **High** | As you talk, the system both detects “speech vs silence” and builds a live transcript. |
| **Low** | Pump calls `_forward_asr`: Silero scores each 512-sample window; Zipformer `accept_pcm` emits **partial** transcripts. Partials go to the turn-taker (`on_asr_partial`). Partials alone do **not** call the LLM. |

### Step C — Decide the user is done (turn-taking)

| | |
|---|---|
| **High** | We don’t cut you off mid-thought. After you pause long enough, we treat the partial as the question. |
| **Low** | `TurnTaker` watches silence after last speech/partial growth. ~**700 ms** for a complete-looking utterance; longer (~**1.5 s**) if the text looks **incomplete** (“Who is the Pre…”). On commit → `commit_user_turn(..., speculative=True)`. |

**Why “speculative”?** We commit from a **partial** before Zipformer’s own endpoint (~1.8–2.8 s) fires. For the class benchmark we also emit an `asr_final` **mark** at commit time so WER can be scored; then we force-finalize the ASR stream when entering THINKING.

### Step D — Thinking (LLM)

| | |
|---|---|
| **High** | State becomes thinking; Haiku streams an answer and may call tools. |
| **Low** | Orchestrator builds messages (system prompt v1 + recent turns + user text), marks `llm_request` (speculative), streams tokens. First token → `llm_first_token`. Tool calls go through dispatch → Tasks / Delegation / Publication. World facts must use `web_search` before answering. |

### Step E — Speaking (TTS + playback)

| | |
|---|---|
| **High** | Sentences are spoken aloud as they become ready. |
| **Low** | Piper synthesizes audio; playback queue plays it. Marks: `tts_request`, `tts_first_audio`, `audio_out`. State `SPEAKING`. |

### Step F — Echo gate (why we don’t hear ourselves)

| | |
|---|---|
| **High** | While the bot talks (and briefly after), we don’t treat speaker bleed as a new user question. |
| **Low** | `_echo_gate_active`: if playing or state is SPEAKING/THINKING, mic chunks are **gated** — not forwarded to normal ASR. Instead they only feed the **barge-in watcher**. After speech, a short hangover keeps the gate up. |

### Step G — Back to listening

| | |
|---|---|
| **High** | When TTS drains, we’re ready for the next question. |
| **Low** | Playback empty + hangover done → gate lifts; ASR/turn-taker reset for a fresh listen window (unless a barge kept ASR live). |

---

## 3 · Barge-in (interrupt while speaking)

### High-level story

> If you talk over Kaki, we don’t want a cough to cancel the answer. We **pause** first; if you keep talking, we **cancel** the LLM, keep what you said, and answer *that*.

### Step-by-step

1. **You are in SPEAKING/THINKING** → echo gate on; mic goes to `_watch_barge_in`, not normal ASR.
2. **Energy floor** — RMS must clear a minimum (and beat echo baseline). Near-silence never barges (fixes false cancels).
3. **Soft-pause (~160 ms sustained)** — flush TTS playback immediately; **start capturing** interrupt audio into ASR (`_barge_asr_live`). LLM may still be running.
4. **Confirm (~320 ms sustained)** — cancel active LLM turn; state → LISTENING; **keep** ASR/turn-taker partials (do not wipe like a normal echo-gate exit).
5. **You finish the interrupt** → normal silence commit → new answer.

### Low-level one-liner

Barge-in is **e**nergy-mandatory + two-phase; soft-pause seeds ASR so the interrupt becomes the next user turn instead of “TTS stopped and we forgot what you said.”

---

## 4 · Tools and the cloud boundary

### High-level

> Speech is local. Brains and a few web services are remote. Tools are **our** servers/APIs the LLM can call.

### What leaves the device

| Leaves | Stays |
|---|---|
| Transcript + prompt context → **Haiku** | Mic PCM, Zipformer, Piper, VAD |
| Search queries → **Tavily** | Timers/notes SQLite, job workspaces |
| Publish content → **GitHub** | Session UI token, local logs |

### Tool groups (MCP-shaped)

- **Tasks:** timers, notes, `web_search` / crawl  
- **Delegation:** one coding job at a time (OpenCode or mock)  
- **Publication:** validate + publish to Pages

### Important honesty for the deck

Live `--serve` runs ASR/TTS **in-process** inside the harness (fast path). We also package the **same engines** as `me3.mcp.asr` / `me3.mcp.tts` WebSocket servers. Diagram should say *local engines (+ MCP packaging)*, not *every audio chunk crosses MCP* (unless you later wire Option A).

---

## 5 · How this differs from “Pipecat as harness” classmates

| | ME3 | Typical classmate |
|---|---|---|
| Live agent loop | **Coco** (`SessionController`) | **Pipecat** pipeline |
| ASR/TTS | Local models in-process | Often cloud API behind MCP |
| Pipecat role | fdbench shuttle | Whole product |
| Turn end | Our silence / incomplete heuristics | Often provider end-of-turn |

**Explain it this way:**  
“The assignment asks for *a* harness with MCP support — examples are Pipecat/onit/crew. We built **Coco**, our custom harness, and use Pipecat for the **benchmark transport**. Functionally we still do mic→VAD→ASR→LLM+tools→TTS with barge-in.”

---

## 6 · Benchmark path (fdbench) — same brain, different ears

### High-level

> The class harness feeds test WAV audio over a socket instead of a mic, and scores latency/WER from stage marks.

### Low-level

1. `fdbench run --connect direct` starts our entry via Pipecat transport.
2. Bridge injects 16 kHz PCM into `SessionController` (benchmark capture/playback).
3. Same turn-taking / LLM / TTS path runs.
4. We `mark()` stages (`vad_end`, `asr_final`, `llm_request`, …).
5. Scorer fills the course table (E2E, ASR final, TTFT, TTFB, WER).

**Laptop proof (2026-11-20):** WER **5.7%**, E2E met; ASR-final latency over target because we finalize at silence-commit; barge-in row empty on stage-latency-only banks. **Pi numbers still needed for the official Results card.**

---

## 7 · Oral walkthrough script (2 minutes)

1. **Goal:** full-duplex voice agent on Pi; interruptible; real tools.
2. **Harness:** custom ME3 session runtime; Pipecat only for fdbench.
3. **Speech local:** Silero → Kroko ASR → … → Piper TTS.
4. **One turn:** listen → partials → silence commit → Haiku (+ tools) → speak → echo gate.
5. **Barge-in:** soft-pause then confirm; keep interrupt as next question.
6. **Cloud:** Haiku, Tavily, GitHub only.
7. **Evidence:** fdbench course bank WER 5.7% on laptop; Pi re-run for deck Results.
8. **Tradeoff we accept:** ASR-final slower than 0.8 s because we wait for incomplete phrases; WER still measurable.

---

## 8 · Where to point in the repo

| Topic | Path |
|---|---|
| Topology / deck diagram | `docs/SYSTEM_TOPOLOGY.md` |
| Filled 2-pager | `docs/deck/ME3_2page_FILLED.md` / `.pdf` |
| Prompt v1 | `docs/PROMPT_v1.md` · `conversation/persona.py` |
| Pump / barge / echo gate | `session/controller.py` |
| Turn commit / LLM / tools | `conversation/pipeline.py` · `turn_taker.py` |
| ASR engine | `speech/asr_engine.py` |
| Laptop bench numbers | `docs/benchmarks/2026-11-20_09-04-05_RESULTS.md` |
| Guideline gaps | `docs/GUIDELINE_GAPS.md` |

---

## 9 · Cheat sheet — one sentence per stage

1. **Mic** — capture and resample to 16 kHz.
2. **VAD** — frame is speech or not.
3. **ASR** — streaming text (partials).
4. **Turn-taker** — silence means “user finished.”
5. **Commit** — speculative user turn + `asr_final` mark.
6. **LLM** — Haiku streams; may call tools.
7. **TTS** — Piper speaks; playback out.
8. **Echo gate** — ignore self-hearing while speaking.
9. **Barge-in** — real user speech pauses then cancels and listens.
10. **Tools/cloud** — timers/notes/web/jobs/publish; Haiku/Tavily/GitHub remote.
