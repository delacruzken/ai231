# Kaki Tamagotchi demo

Localhost-only 2D room that reacts to accepted voice commands (Alexa-style wake beep + spoken confirmations).

## Quick start (Mac)

The **website is optional**. Voice wake/commands run in the assistant terminal.
The site is only the den visualization + beeps.

**Voice only (no UI):**

```bash
cd me2
bash grammar_asr/scripts/live_debug_60s.sh
```

Watch the console for `[wake] HIT` / `[cmd]`. No browser needed.

**With den UI:**

Terminal A — UI server (`127.0.0.1:8765` only):

```bash
cd me2
python tamagotchi_demo/server.py
```

Open [http://127.0.0.1:8765/](http://127.0.0.1:8765/) and **click the page once** (browsers block audio until a gesture). Then:

```bash
cd me2
DEMO_UI=1 bash grammar_asr/scripts/live_debug_60s.sh
```

## Behaviour

| Event | UI |
|-------|----|
| Idle / default | Music stopped, phone idle, curtains closed, clock HUD hidden, no reminders. Lights stay as last set. |
| Wake (`kaki`) | Resets other props, keeps lights, then rising chime + “Listening…” (mic muted during chime) |
| Listen timeout | Soft double blip (no command before window closed); back to wake — not the reject sound |
| Rejected / OOS | Caption “I did not understand that” + low falling error cue; mic muted during the cue |
| Accepted command | Props update + caption + `speechSynthesis`; mic stays muted until speech finishes |
| `PLAY_MUSIC` | Den vinyl + optional Spotify shuffle-play on your playlist ([SPOTIFY.md](SPOTIFY.md)) |
| `CALL` / `MESSAGE` | Den phone/message props + optional FaceTime / Messages on this Mac ([MAC_COMMS.md](MAC_COMMS.md)) |
| `LIGHT_ON` / `LIGHT_OFF` | Lights stay on until an explicit off command (not cleared by the 20s timer) |
| `COLOR` (Red / Blue / Green) | Turns lights on and tints the lamp + room wash; color persists like lights |
| `BRIGHTNESS` (20 / 60 / 100%) | Scales **lamp** glow / beam only (not the whole den); turns lights on |
| `TEMPERATURE` | Turns AC on; digital readout on the AC shows the set degrees |
| `ALARM` | Coffee-table alarm clock (always present) lights up with the set time |
| `TIMER` | Coffee-table timer (left) lights up and counts down live |
| `WEATHER` | Opens curtains and shows live Metro Manila weather (Open-Meteo) |
| `TIME` | Manila clock HUD (`Asia/Manila`) |
| 20s after a command | Other props return to idle; lights are unchanged (also `POST /event` `{"type":"reset"}` for a full blank) |

Voice-only: there is no clickable furniture, text box, or on-page mic. Drive the room with `--demo-ui` or `POST /event`.

## Pi + Mac den (usual demo)

Den server stays on the **Mac**. Pi assistant uses `--demo-ui` → `http://127.0.0.1:8765`,
so you need a **reverse** SSH tunnel (Mac port → appear on the Pi):

**Terminal on the Mac** (keep this open while demoing):

```bash
# den UI
cd me2 && python tamagotchi_demo/server.py

# separate terminal — reverse tunnel
ssh -N -R 8765:127.0.0.1:8765 delacruzharv@rpi5-khdc.local
```

**On the Pi:**

```bash
# optional: confirm the tunnel
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8765/

python assistant.py --model model.onnx --wake-model wake.onnx \
  --reject-gate reject_gate.json --demo-ui --mic …
```

Open the den in a browser on the **Mac** at http://127.0.0.1:8765/ and click once for sound.

(`ssh -L` is the opposite direction — only if the server runs on the Pi and you browse from the Mac.)

## Manual event smoke test

```bash
curl -s -X POST http://127.0.0.1:8765/event \
  -H 'Content-Type: application/json' \
  -d '{"type":"wake"}'

curl -s -X POST http://127.0.0.1:8765/event \
  -H 'Content-Type: application/json' \
  -d '{"type":"command","intent":"LIGHT_ON","slot":"","spoken":"Turned on the lights"}'
```
