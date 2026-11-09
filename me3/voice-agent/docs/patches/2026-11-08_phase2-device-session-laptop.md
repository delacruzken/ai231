# Patch: 2026-11-08_phase2-device-session-laptop

| Field | Value |
|---|---|
| ID | `2026-11-08_phase2-device-session-laptop` |
| Date | 2026-11-08 |
| Phase | 2 |
| Verdict | `improved` |
| Related decisions | #5, #6, #8 |

## Challenge

Phase 1 proved speech engines on files only. Needed live laptop capture/playback, session ON/OFF with no stale replay, readiness UI, and a clean path to swap in the M1A later.

## Change

- `sounddevice` device layer + linear resample boundaries (`me3.speech.device`, `resample`)
- Duplex harness `python -m me3.speech.device_check`
- Session controller Off/Listening/Speaking with discard-while-OFF and playback generation flush
- FastAPI backend (`--serve`) + Vite UI (large ON/OFF, badges, transcript)
- `ME3_UI_TOKEN` auth for LAN UI

## Hypothesis

Laptop default devices are enough to validate session gating and duplex plumbing before M1A arrives; same device API will select M1A by name hint later.

## How measured

- Host: laptop (macOS), default mic/speakers
- Duplex: `python -m me3.speech.device_check --seconds 2.5`
- OFF/ON: scripted flush + stale generation reject; controller starts OFF
- Tests: `pytest` (17 passed including resample + API + OFF/ON)

## Before / after

- **Before:** no device I/O, no session UI
- **After:**
  - Duplex gate OK — `artifacts/device_phase2/20261108T044408Z/` (`capture_16k.wav` + `tts_played.wav`)
  - OFF flush cleared ~4.5 s queued audio to 0; stale `play(..., generation=old)` rejected — `artifacts/device_phase2/20261108T044428Z_off_on/off_on_metrics.json`
  - Backend/UI: `python -m me3.app.main --serve` with `ME3_UI_TOKEN`
- **Not claimed:** M1A Linux/Pi compatibility; laptop AEC (echo expected); conversational LLM

## Follow-up — M1A swap checklist

1. Plug in EMEET M1A; `python -m me3.speech.device_check --list-devices`
2. Set `audio.input_device_hint` / `output_device_hint` to `EMEET` or `M1A` (or explicit indices)
3. Re-run `device_check`; confirm duplex on M1A
4. Unplug/replug + session OFF/ON; on Pi, add OS reboot matrix
5. Set `pending.m1a_device_path: false` and open a new patch with Pi/M1A metrics

Keep laptop path as fallback.
