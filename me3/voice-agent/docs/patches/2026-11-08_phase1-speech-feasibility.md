# Patch: 2026-11-08_phase1-speech-feasibility

| Field | Value |
|---|---|
| ID | `2026-11-08_phase1-speech-feasibility` |
| Date | 2026-11-08 |
| Phase | 1 |
| Verdict | `improved` |
| Related decisions | #1, #2, #4, #7, #14 |

## Challenge

Phase 0 had only MCP stubs — no proof that local Zipformer ASR can keep up while Piper TTS runs, or that queues stay bounded under contention.

## Change

- Added speech optional deps: `sherpa-onnx==2.0.1`, `piper-tts==1.8.0`, onnxruntime, soundfile, psutil, websockets
- `scripts/fetch_speech_models.sh` + `models/MANIFEST.json` (Zipformer INT8, Piper lessac-medium, Silero from fd-benchmark)
- In-process Silero VAD (`me3.speech.vad`) — not a fifth MCP
- Streaming ASR engine + WebSocket ASR service; Piper TTS engine + WebSocket TTS service
- Recorded-input concurrency runner `python -m me3.speech.runner`
- **Runtime note:** sherpa-onnx 2.0 deprecated VITS `data_dir` phonemization; TTS uses **piper-tts** with the **same** `en_US-lessac-medium.onnx` weights (decision #4 voice unchanged)

## Hypothesis

INT8 streaming Zipformer + Piper on CPU can sustain concurrent load on a laptop with RTF ≪ 1 and bounded queues, clearing the Phase 1 exit gate before Pi/M1A work.

## How measured

- Host: development laptop (macOS arm64), warm after model load
- Command: `python -m me3.speech.runner` (fixture WAV + concurrent TTS paragraph)
- Metrics: ASR RTF, queue high-water, TTS audio seconds, CPU/RSS in `metrics.json`
- Sample: one concurrency trial (see artifacts path below)

## Before / after

- **Before:** no speech engines; no RTF evidence
- **After (artifact `artifacts/speech_phase1/20261108T041540Z/`):**
  - ASR RTF ≈ **0.076** (target ≤ 0.5) while TTS produced ~7.0 s audio
  - Final transcript: `PLEASE SET A TIMER FOR FIVE MINUTES ALSO REMIND ME TO CHECK THE OVEN WHEN IT RINGS`
  - Queue high-water **64** (= max) — bounded (backpressure/sentinel drain fixed so finals are not dropped)
  - Runner RSS ≈ 0.5–0.6 GB; TTS WAV written for listening review
- Voice sample review: TTS WAV present at `tts_output.wav`; ASR partials readable and final matches fixture text (Piper→16 kHz fixture)

## Follow-up

Keep. **Pi re-measure required** before claiming Pi readiness. Next: Phase 2 device/session (M1A). Optionally lower `asr_queue_max` or increase ASR worker priority if queue sits at cap on Pi.
