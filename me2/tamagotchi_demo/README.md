# Kaki Tamagotchi demo

Localhost-only 2D room that reacts to accepted voice commands (Alexa-style wake beep + spoken confirmations).

## Quick start (Mac)

Terminal A — UI server (binds `127.0.0.1:8765` only):

```bash
cd me2
python tamagotchi_demo/server.py
```

Open [http://127.0.0.1:8765/](http://127.0.0.1:8765/) and click **Enable sound**.

Terminal B — mic assistant:

```bash
cd me2
DEMO_UI=1 bash grammar_asr/scripts/live_debug_60s.sh
```

Or manually:

```bash
python -m grammar_asr.runtime.assistant \
  --model grammar_asr/runs/intent_hf_v1/command.onnx \
  --wake-model grammar_asr/runs/wake_kaki_v5/kaki.onnx \
  --reject-gate grammar_asr/runs/intent_hf_v1/reject_gate_live.json \
  --prefer intent_head --wake-threshold 0.70 \
  --mic --debug --demo-ui
```

## Behaviour

| Event | UI |
|-------|----|
| Idle / default | Music stopped, phone idle, curtains closed, clock HUD hidden, no reminders. Lights stay as last set. |
| Wake (`kaki`) | Resets other props, keeps lights, then chime + “Listening…” |
| Rejected / OOS | Caption “I did not understand that” + a low error cue (no speech, no prop change) |
| Accepted command | Props update + caption + `speechSynthesis` |
| `LIGHT_ON` / `LIGHT_OFF` | Lights stay on until an explicit off command (not cleared by the 20s timer) |
| `WEATHER` | Opens curtains and shows live Metro Manila weather (Open-Meteo) |
| `TIME` | Manila clock HUD (`Asia/Manila`) |
| 20s after a command | Other props return to idle; lights are unchanged (also `POST /event` `{"type":"reset"}` for a full blank) |

Voice-only: there is no clickable furniture, text box, or on-page mic. Drive the room with `--demo-ui` or `POST /event`.

## Pi

Run `server.py` on the Pi (or tunnel):

```bash
ssh -L 8765:127.0.0.1:8765 delacruzharv@rpi5-khdc.local
```

On the Pi, start the assistant with `--demo-ui` (same machine as the server). Open the page in a browser pointed at that localhost.

## Manual event smoke test

```bash
curl -s -X POST http://127.0.0.1:8765/event \
  -H 'Content-Type: application/json' \
  -d '{"type":"wake"}'

curl -s -X POST http://127.0.0.1:8765/event \
  -H 'Content-Type: application/json' \
  -d '{"type":"command","intent":"LIGHT_ON","slot":"","spoken":"Turned on the lights"}'
```
