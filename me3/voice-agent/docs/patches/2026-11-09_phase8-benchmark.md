# Patch: 2026-11-09_phase8-benchmark

| Field | Value |
|---|---|
| ID | `2026-11-09_phase8-benchmark` |
| Date | 2026-11-09 |
| Phase | 8 |
| Verdict | `improved` |
| Related decisions | #9, #21 |

## Challenge

Wire ME3 into the pinned `fd-benchmark` direct-feed path without a full Pipecat PCM pipeline: inject 16 kHz audio, return TTS, emit stage marks, and reset state between trials.

## Change

- `me3.benchmark/`: BenchCapture/Playback, Pipecat↔ME3 bridge, fdbench marks wrapper, trial reset
- `scripts/fdbench_assistant_entry.py` + local `fd-benchmark/assistant_entry.py`
- Stage marks on VAD/ASR/LLM/TTS/audio_out/interrupt (level 2)
- `POST /api/benchmark/reset` for `--reset-cmd`
- Pipecat 1.15.0 kept for transport only; speech still Zipformer/Piper/Silero
- Benchmark `PipelineWorker` sets `idle_timeout_secs=None` — ME3 speech does not emit Pipecat BotSpeaking/UserSpeaking frames, so the default 300s idle cancel killed the assistant mid-run (fdbench launches with `--once`, so it never reconnects)

## Hypothesis

A thin Pipecat shuttle around the existing SessionController preserves the real ME3 path after the transport boundary, satisfying fdbench’s socket protocol while custom services report marks explicitly.

## How measured

- `pytest tests/test_benchmark_phase8.py`
- `python -m me3.app.main --benchmark-check`
- `voice-agent/.venv/bin/python fd-benchmark/launch_assistant.py --check`
- Laptop `fdbench run --connect direct` course stage-latency bank (40 trials) completed after idle-timeout disable (run `2026-11-09_19-03-07`)

## Before / after

- **Before:** FDBENCH_URL logged as unwired; first long bank died at ~5 min with `assistant disconnected` / `not connected`
- **After:** assistant entry + bridge + marks + trial reset; launcher `--check` discovers ME3; idle timeout disabled for bench worker so multi-trial runs stay connected
- **Not claimed:** course gate numbers met; live OpenCode during bench (forced mock); M1A cable parity

## Follow-up

Phase 9 hardening. Score the software-feed bank; compare later M1A cable runs separately. Investigate suspicious `0.00 s` E2E replies and pause false starts (TOR).
