"""Device / demo actions for accepted intents (recognition stays independent)."""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from typing import Optional

from .console import tag, paint, C


def confirm_phrase(intent: str, slot: str = "") -> str:
    """Alexa-style spoken confirmation for an accepted command."""
    slot = (slot or "").strip()
    fixed = {
        "PLAY_MUSIC": "Playing music",
        "WEATHER": "Here is the weather",
        "TIME": "Here is the time",
        "LIGHT_ON": "Turned on the lights",
        "LIGHT_OFF": "Turned off the lights",
        "PAUSE": "Paused",
        "STOP": "Stopped",
        "NEXT": "Playing the next song",
        "VOLUME_UP": "Volume up",
        "VOLUME_DOWN": "Volume down",
        "CALL": "Calling",
        "MESSAGE": "Sending a message",
        "LIST_REMINDERS": "Here are your reminders",
    }
    if intent in fixed:
        return fixed[intent]
    if intent == "TIMER":
        return f"Timer set for {slot}" if slot else "Timer set"
    if intent == "ALARM":
        return f"Alarm set for {slot}" if slot else "Alarm set"
    if intent == "TEMPERATURE":
        return f"Temperature set to {slot}" if slot else "Temperature updated"
    if intent == "BRIGHTNESS":
        return f"Brightness set to {slot}" if slot else "Brightness updated"
    if intent == "COLOR":
        return f"Color set to {slot}" if slot else "Color updated"
    if intent == "CREATE_REMINDER":
        return f"Reminder created: {slot}" if slot else "Reminder created"
    return intent.replace("_", " ").title()


class MockDispatcher:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.last = None

    def dispatch(self, intent: str, slot: str = "") -> str:
        msg = confirm_phrase(intent, slot)
        self.last = {"intent": intent, "slot": slot, "message": msg}
        if self.verbose:
            print(
                f"{tag('action', 'action')} {paint(msg, C.BOLD, C.MAGENTA)}",
                flush=True,
            )
        return msg

    def notify(self, event_type: str, **payload) -> None:
        """Optional side-channel (wake / listen_end). No-op for mock."""
        return None


class HttpDemoDispatcher:
    """Fan out to the localhost Tamagotchi demo UI (fail-soft)."""

    def __init__(self, base_url: str = "http://127.0.0.1:8765",
                 verbose: bool = True, inner: Optional[MockDispatcher] = None):
        self.base_url = base_url.rstrip("/")
        self.verbose = verbose
        self.inner = inner or MockDispatcher(verbose=verbose)
        self.last = None

    def dispatch(self, intent: str, slot: str = "") -> str:
        spoken = self.inner.dispatch(intent, slot)
        self.last = {"intent": intent, "slot": slot, "message": spoken}
        self._post({
            "type": "command",
            "intent": intent,
            "slot": slot or "",
            "spoken": spoken,
            "ts": time.time(),
        })
        return spoken

    def notify(self, event_type: str, **payload) -> None:
        body = {"type": event_type, "ts": time.time()}
        body.update(payload)
        self._post(body)

    def _post(self, event: dict) -> None:
        # Never block the mic loop on a dead SSH tunnel / slow UI.
        def worker() -> None:
            data = json.dumps(event).encode("utf-8")
            req = urllib.request.Request(
                f"{self.base_url}/event",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=0.35) as resp:
                    resp.read()
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if self.verbose:
                    print(
                        f"{tag('demo', 'vad')} demo-ui unreachable "
                        f"({self.base_url}): {exc}",
                        flush=True,
                    )

        threading.Thread(target=worker, daemon=True).start()
