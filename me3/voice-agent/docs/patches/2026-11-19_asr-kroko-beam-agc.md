# Patch: 2026-11-19_asr-kroko-beam-agc

| Field | Value |
|---|---|
| ID | `2026-11-19_asr-kroko-beam-agc` |
| Date | 2026-11-19 |
| Phase | 2 follow-up / speech quality |
| Verdict | `improved` (laptop smoke; live M1A retest pending) |
| Related decisions | #2, #5, #6 |

## Challenge

Live M1A session on laptop produced high WER on short conversational turns (`HELLO→LO`, `NEWS→RELUE`, mangled questions). Phase 1 pin was LibriSpeech Zipformer `en-2023-02-21` INT8 + `greedy_search`, with no mic AGC.

## Change

- Default ASR → `sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06` (fetch script + MANIFEST)
- Decode → `modified_beam_search` (`max_active_paths=4`); `asr_model_type=""` for sherpa auto-detect (Kroko Zipformer2)
- Soft AGC on ASR feed only (`audio.agc_*`) so quiet M1A capture is boosted without changing barge-in energy path
- Config knobs in `settings.example.yaml` / `me3.config`; unit tests for AGC + defaults

## Hypothesis

Conversational Kroko + beam search + level normalization should cut short-utterance WER vs LibriSpeech greedy INT8 on the same M1A path.

## How measured

- Host: macOS laptop, Kroko test wav decode
- Unit: `pytest tests/test_asr_agc.py`
- Live: retest phrases after restarting `--serve` (owner to confirm)

## Before / after

- **Before:** LibriSpeech 2023-02-21 INT8, greedy; live UI mangled short commands
- **After (smoke):** Kroko loads with empty `model_type`; test wav → `Ask not what your country can do for you. As…`; AGC boosts quiet frames; defaults assert Kroko + beam

## Follow-up

Keep. Re-measure RTF on Pi 5 before claiming Pi readiness (decision #2). Optional: hotword bias for timer/notes; remove old LibriSpeech tree from `models/` once soak is clean.
