# ME3 guideline gaps — address one by one

**Source:** Machine Exercise 2-page template (`machine_exercise_2page.pdf`) + written brief (full-duplex agent on RPi, MCP ASR/TTS you build, VAD, real tasks, standardized latency bench).  
**Compared to:** `ME3_Architecture_and_Development_Plan.md`, live `SessionController` path, fdbench Phase 8.  
**Date opened:** 2026-11-20  
**Status:** tracking — resolve in order below before claiming deck-final / Pi acceptance.

Use this file as a checklist. For each item: pick an option, do the work, close with a patch note + evidence, then mark **Status** here.

---

## How to use

1. Work **one gap at a time** (recommended order = section numbers).
2. Update **Status** (`open` → `in progress` → `closed` / `accepted risk`).
3. Link the patch note or PR when closed.
4. Do not treat laptop direct-feed numbers as RPi Results-card numbers until Gap 3 is closed.

---

## Gap 1 — Live ASR/TTS are not on the MCP path

| Field | Value |
|---|---|
| Status | `closed` — **Option B Hybrid** locked 2026-11-20 (`docs/SYSTEM_TOPOLOGY.md`, decision 14b) |
| Priority | — |
| Template | Mic → VAD → ASR MCP → agent → TTS MCP → speaker |
| Chosen design | **In-process engines on live `--serve` / fdbench**; **same** Zipformer/Piper code packaged as `me3.mcp.asr` / `tts` WS servers. Deck must say “local engines (+ MCP/WS packaging)”, not “PCM always crosses MCP” |

### Resolution

- Local speech already accepted (Gap 4); forcing Option A (live WS hop) is optional later, not required for correctness.
- Tasks / Delegation / Publication remain MCP-shaped tool surfaces used by the LLM.
- Optional demo: `python -m me3.mcp.asr.server` and `python -m me3.mcp.tts.server` alongside `--serve`.

### Acceptance criteria

- [x] Topology doc matches demo path
- [x] ASR/TTS startable as separate services (servers exist)
- [ ] Optional: Pi systemd units start those servers (Gap 6)
- [x] Design recorded (no silent claim that live path is MCP-piped)

### Reopen if

Instructor requires every audio chunk to traverse a separate MCP process (then implement Option A).

---

## Gap 2 — Pipecat is not the live duplex harness

| Field | Value |
|---|---|
| Status | `closed` — **Option A Honest labeling** 2026-11-20 |
| Priority | — |
| Chosen design | Live duplex = **ME3 `SessionController`**. Pipecat 1.15 = **fdbench transport shuttle only** |

### Resolution

- README + `DECISIONS.md` #9 + `SYSTEM_TOPOLOGY.md` use the same wording.
- Deck harness line: *“Custom ME3 duplex runtime; Pipecat used as the fdbench audio transport adapter.”*
- Option C (thin Pipecat shell) deferred unless course staff require a Pipecat process tree.

### Acceptance criteria

- [x] Topology / README / decisions updated
- [ ] Draw §1 slide from `SYSTEM_TOPOLOGY.md` (owner deck work)

---

## Gap 3 — Benchmark evidence is laptop / software-feed, not RPi + mic

| Field | Value |
|---|---|
| Status | `partial` — **Option A two-tier**; laptop bank written; **Pi deck numbers still open** |
| Priority | **P0 for Results card** |
| Chosen design | Label every run by rig; never paste Darwin Mean/p95 onto an “RPi” Results card |

### Done

- [x] Laptop course bank `2026-11-20_09-04-05` → [`docs/benchmarks/2026-11-20_09-04-05_RESULTS.md`](benchmarks/2026-11-20_09-04-05_RESULTS.md) (WER 5.7%, E2E met)

### Still open

- [ ] Pi direct/remote course bank for deck Results machine line
- [ ] M1A / cables run for physical barge-in + AEC (decisions #5/#6)

### Next steps

1. Flash/boot Pi 5; install voice-agent + models.
2. `fdbench run --connect direct --where remote` (or on-Pi) with same phrase/runs.
3. Write `docs/benchmarks/<id>_pi_RESULTS.md` and only then fill deck Results.

---

## Gap 4 — Template gray “SOTA ASR/TTS API” vs local Zipformer/Piper

| Field | Value |
|---|---|
| Status | `closed` — **accepted design** (owner 2026-11-20): local ASR/TTS OK if understanding + duplex behavior hold; no cloud speech required |
| Priority | — |
| Template table | Model examples: `<SOTA ASR API>`, `<SOTA TTS API>` (placeholders) |
| Brief / our plan (#1) | Local pretrained speech; commercial API for **LLM** only; raw mic audio not sent to cloud speech |

### Why it mattered

Gray cells look like Deepgram/Cartesia-style APIs (colleague deck). Local Zipformer + Piper is a valid fill-in if WER/functionality are demonstrated.

### Resolution

- Keep **local Kroko/Zipformer + Piper** (decision #1).
- Deck §4: list those models, “served on RPi” — not cloud speech APIs.
- Evidence: laptop course bank WER **5.7%** (n=20) on `2026-11-20_09-04-05`; re-measure on Pi later (Gap 3).
- Do **not** chase colleague cloud ASR/TTS unless quality on Pi/M1A fails acceptance.

### Acceptance criteria

- [x] Owner confirms local speech OK given understanding + same functionality
- [ ] Deck §4 table filled with Kroko + Piper (not gray API placeholders)
- [ ] Pi/M1A WER / live listen still acceptable when Gap 3 closes

---

## Gap 5 — Barge-in metric often empty on stage-latency banks

| Field | Value |
|---|---|
| Status | `deferred` — design **A+B**; run when preparing deck Results |
| Priority | **P1** before final §5 table |
| Chosen design | Keep live two-phase barge-in; score with interruption/FDB rounds (**A**) and cite `python -m me3.speech.barge_check` (**B**) until then |

### Acceptance criteria

- [ ] Course table barge-in n&gt;0 from automated run **or** linked microbench numbers
- [x] Live path: stop TTS + cancel LLM + capture interrupt (already implemented)

---

## Gap 6 — systemd / “all MCP servers running on RPi” lifecycle

| Field | Value |
|---|---|
| Status | `deferred` — Phase 9 / Pi bring-up; design locked under Gap 1 **B** |
| Priority | **P1** with Gap 3 Pi |
| Chosen design | systemd owns: `me3-session` (`--serve`), optional `me3-asr` / `me3-tts` (MCP WS demos), `opencode` worker separately. Session does **not** require ASR/TTS children for live audio (in-process). Single supervisor per unit |

### Acceptance criteria

- [ ] Unit files + start order documented
- [ ] Reboot → session ready smoke
- [ ] OpenCode labeled mock vs live on demo day

---

## Gap 7 — Deck content engineering (§2 prompt, §3 context)

| Field | Value |
|---|---|
| Status | `closed` (repo) — paste [`PROMPT_v1.md`](PROMPT_v1.md) onto deck slides |
| Priority | — |
| Chosen design | `PROMPT_VERSION = "v1"` in `persona.py`; deck excerpt + §3 table in `docs/PROMPT_v1.md` |

### Acceptance criteria

- [x] Versioned excerpt in repo
- [ ] Owner copies onto deck §2/§3
- [ ] Optional: pull context `messages`/`chars` from a scored run into Challenges/Results notes

---

## Gap 8 — ASR final timing vs template “final on VAD end”

| Field | Value |
|---|---|
| Status | `closed` — **Option A** keep silence-commit finals |
| Priority | — |
| Chosen design | `asr_final` = transcript at **turn commit** (after VAD end + silence patience). Explains ASR-final mean ~1.96 s over &lt;0.8 s target while WER still scores |

### Challenges slide one-liner

> “We finalize ASR at silence-commit (not raw VAD end) so incomplete questions aren’t cut; course ASR-final latency is higher, WER stays measurable (laptop 5.7%).”

### Acceptance criteria

- [x] Behavior documented here + laptop WER n=20
- [ ] Same Challenges line on deck
- [ ] Re-confirm WER on Pi (Gap 3)

---

## Suggested resolution order

| Step | Gap | Outcome |
|---|---|---|
| ✓ | **4** | Local ASR/TTS accepted |
| ✓ | **1** | Hybrid topology locked + `SYSTEM_TOPOLOGY.md` |
| ✓ | **2** | Honest Pipecat label |
| ✓ | **7** | `PROMPT_v1.md` + `PROMPT_VERSION` |
| ✓ | **8** | Silence-commit finals documented |
| ✓ partial | **3** | Laptop results written; **Pi still TODO** |
| next | **3** Pi | Course bank on Raspberry Pi for deck Results |
| next | **5** | Barge-in scored suite or microbench cite |
| next | **6** | systemd + soak on Pi |

---

## Quick reference — template checklist

Copy to the deck only when the matching gap is closed or explicitly accepted.

- [ ] §1 System diagram (on-RPi vs cloud; barge-in) — Gaps 1, 2, 3
- [ ] §2 Prompt versioned in repo — Gap 7
- [ ] §3 Context engineering — Gap 7
- [ ] §4 MCP table (ASR/TTS/VAD/Task rows filled) — Gaps 1, 4, 6
- [ ] §5 Latency table Mean/p95 + chart — Gaps 3, 5, 8
- [ ] Results: RPi model/RAM/cooling/network/date/models — Gap 3
- [ ] Challenges: one row per issue with fix + measurement — all gaps as closed

---

## Related paths

- Architecture: `me3/ME3_Architecture_and_Development_Plan.md`
- Decisions: `me3/voice-agent/docs/DECISIONS.md`
- Patch index: `me3/voice-agent/docs/PATCH_NOTES.md`
- ASR-final mark fix: `docs/patches/2026-11-20_asr-final-on-silence-commit.md`
- Benchmark results: `me3/voice-agent/docs/benchmarks/`
- Template attachment: course `machine_exercise_2page.pdf`
