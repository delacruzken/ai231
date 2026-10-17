# Mac-in-the-middle: FaceTime / Messages

The Pi (or Mac) assistant recognizes **CALL** / **MESSAGE**. Real FaceTime and
Messages run on the **Mac** that hosts `tamagotchi_demo/server.py`.

```
Pi mic → assistant --demo-ui → SSH tunnel → Mac :8765
                                         → den UI + open facetime:/sms:
```

## Setup

1. Add targets to `me2/.env` (see `.env.example`):

```bash
ME2_CALL_TARGET=+63917xxxxxxx      # or an Apple ID email
ME2_CALL_MODE=facetime             # or facetime-audio
ME2_CALL_AUTOSEND=0                # 1 = auto-click FaceTime Call (Accessibility)
ME2_MESSAGE_TARGET=+63917xxxxxxx
ME2_MESSAGE_BODY=Hey — this is Kaki checking in.
ME2_MESSAGE_AUTOSEND=0             # 1 = try auto-send via AppleScript
```

2. Restart the den server on the Mac (`python tamagotchi_demo/server.py`).

3. Keep the reverse tunnel + Pi `--demo-ui` (or run the assistant on the Mac with `DEMO_UI=1`).

## Smoke test (Mac, den running)

```bash
curl -s -X POST http://127.0.0.1:8765/event \
  -H 'Content-Type: application/json' \
  -d '{"type":"command","intent":"CALL","spoken":"Calling"}'

curl -s -X POST http://127.0.0.1:8765/event \
  -H 'Content-Type: application/json' \
  -d '{"type":"command","intent":"MESSAGE","spoken":"Sending a message"}'
```

You should see `[comms] open facetime://…` / `sms:…` in the den server terminal,
and FaceTime / Messages should open.

## Notes

- **MESSAGE** with `AUTOSEND=0` opens a compose window; you may still tap **Send**.
- `ME2_MESSAGE_AUTOSEND=1` uses Messages AppleScript — grant Automation access if macOS prompts.
- **CALL** always opens FaceTime with a confirm step on macOS (Apple policy for URL schemes).
- `ME2_CALL_AUTOSEND=1` opens FaceTime then UI-scripts a click on **Call**. Grant
  **System Settings → Privacy & Security → Accessibility** to the app that runs
  the den server (Terminal, iTerm, or your Python). On newer macOS the Call
  button may live in Notification Center; the script tries both.
- Continuity can surface FaceTime on your iPhone when signed into the same Apple ID.
- No targets in `.env` → den animation still runs; server prints `[comms] … skipped`.
