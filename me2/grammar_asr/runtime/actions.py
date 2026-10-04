"""Minimal mock device actions for demos (recognition stays independent)."""
from __future__ import annotations


class MockDispatcher:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.last = None

    def dispatch(self, intent: str, slot: str = "") -> str:
        msg = {
            "PLAY_MUSIC": "Playing music",
            "WEATHER": "Fetching weather",
            "TIME": "Announcing time",
            "LIGHT_ON": "Lights on",
            "LIGHT_OFF": "Lights off",
            "PAUSE": "Paused",
            "STOP": "Stopped",
            "NEXT": "Next track",
            "VOLUME_UP": "Volume up",
            "VOLUME_DOWN": "Volume down",
            "CALL": "Placing call",
            "MESSAGE": "Sending message",
            "LIST_REMINDERS": "Listing reminders",
            "TIMER": f"Timer set ({slot})",
            "ALARM": f"Alarm set ({slot})",
            "TEMPERATURE": f"Temperature set ({slot})",
            "BRIGHTNESS": f"Brightness set ({slot})",
            "COLOR": f"Color set ({slot})",
            "CREATE_REMINDER": f"Reminder created ({slot})",
            "OUT_OF_SCOPE": "Ignored (out of scope)",
        }.get(intent, f"Unhandled intent {intent}")
        self.last = {"intent": intent, "slot": slot, "message": msg}
        if self.verbose:
            print(f"[action] {msg}")
        return msg
