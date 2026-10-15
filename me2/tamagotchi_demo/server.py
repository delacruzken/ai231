#!/usr/bin/env python3
"""Localhost-only Tamagotchi demo bridge (SSE + POST events).

Bind: 127.0.0.1:8765
  GET  /         → static room UI
  POST /event    → JSON event from the voice assistant
  GET  /events   → Server-Sent Events stream for the browser
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import queue
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
HOST = "127.0.0.1"
PORT = 8765
RESET_AFTER_SEC = 20.0
MANILA_WX_URL = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude=14.5995&longitude=120.9842"
    "&current=temperature_2m,weather_code,is_day"
    "&timezone=Asia%2FManila"
)
_WX_LOCK = threading.Lock()
_WX_CACHE: Dict[str, Any] = {"ts": 0.0, "data": None}

# PAGASA NCR fallback (partly cloudy to cloudy, isolated showers).
_WX_FALLBACK = {
    "key": "cloudy",
    "label": "Partly cloudy to cloudy in Metro Manila",
    "spoken": "Partly cloudy to cloudy in Metro Manila, with isolated showers possible",
    "temperature": None,
    "is_day": True,
    "code": 2,
}


def wmo_to_weather(code: int, is_day: bool) -> tuple[str, str]:
    if code in (51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82, 95, 96, 99):
        if code >= 95:
            return "rain", "Thunderstorms in Metro Manila"
        if code in (80, 81, 82):
            return "rain", "Rain showers in Metro Manila"
        return "rain", "Rain in Metro Manila"
    if code in (3, 45, 48):
        return "cloudy", "Overcast skies in Metro Manila"
    if code == 0:
        if is_day:
            return "sunny", "Clear skies in Metro Manila"
        return "fair", "Clear night in Metro Manila"
    if code in (1, 2):
        return "fair" if is_day else "cloudy", "Partly cloudy in Metro Manila"
    return "cloudy", "Partly cloudy to cloudy in Metro Manila"


def fetch_manila_weather(*, force: bool = False) -> Dict[str, Any]:
    now = time.time()
    with _WX_LOCK:
        cached = _WX_CACHE["data"]
        if cached and not force and now - float(_WX_CACHE["ts"]) < 300:
            return dict(cached)
    try:
        req = urllib.request.Request(MANILA_WX_URL, headers={"User-Agent": "kaki-den/1.0"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        cur = payload.get("current") or {}
        code = int(cur.get("weather_code") or 2)
        is_day = bool(cur.get("is_day", 1))
        temp = cur.get("temperature_2m")
        key, label = wmo_to_weather(code, is_day)
        spoken = label if temp is None else f"{label}, {int(round(float(temp)))} degrees"
        data = {
            "key": key,
            "label": label,
            "spoken": spoken,
            "temperature": int(round(float(temp))) if temp is not None else None,
            "is_day": is_day,
            "code": code,
        }
        with _WX_LOCK:
            _WX_CACHE["ts"] = now
            _WX_CACHE["data"] = data
        return dict(data)
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return dict(cached or _WX_FALLBACK)


def blank_room(*, listening: bool = False, lights: Optional[bool] = None) -> Dict[str, Any]:
    """Command props off: no music/phone/reminders/time/weather HUD.

    Lights stay as given — they only change on LIGHT_ON / LIGHT_OFF.
    """
    wx = fetch_manila_weather()
    return {
        "lights": False if lights is None else bool(lights),
        "music": "stopped",  # stopped | playing | paused
        "track": 1,
        "volume": 5,
        "phone": "idle",  # idle | ringing
        "message": "",
        "weather": wx["key"],
        "live_weather": wx["key"],
        "weather_label": wx["label"],
        "weather_spoken": wx["spoken"],
        "timer": "",
        "alarm": "",
        "brightness": 60,
        "color": "warm",
        "temperature": 24,
        "ac": False,
        "reminders": [],
        "listening": listening,
        "caption": "Listening…" if listening else "",
        "spoken": "",
        "show_time": False,
        "curtains_open": False,
    }


class RoomHub:
    """Broadcast JSON events to SSE clients; keep latest room snapshot."""

    def __init__(self, reset_after: float = RESET_AFTER_SEC) -> None:
        self._lock = threading.Lock()
        self._clients: List[queue.Queue] = []
        self._reset_after = max(0.0, float(reset_after))
        self._reset_timer: Optional[threading.Timer] = None
        self._epoch = 0
        self.state: Dict[str, Any] = blank_room()
        # Browser / assistant set this while speakers play (TTS, chimes).
        self._audio_busy_until = 0.0

    def set_audio_busy(self, ms: float) -> None:
        """Hold mic gate closed for at least ``ms`` milliseconds from now."""
        until = time.time() + max(0.0, float(ms)) / 1000.0
        with self._lock:
            self._audio_busy_until = max(self._audio_busy_until, until)

    def clear_audio_busy(self) -> None:
        with self._lock:
            self._audio_busy_until = 0.0

    def audio_gate(self) -> Dict[str, Any]:
        now = time.time()
        with self._lock:
            until = float(self._audio_busy_until)
        return self._gate_dict(now, until)

    @staticmethod
    def _gate_dict(now: float, until: float) -> Dict[str, Any]:
        busy = now < until
        return {"busy": busy, "until": until, "remaining_ms": max(0.0, (until - now) * 1000.0)}

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=64)
        with self._lock:
            self._clients.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._clients:
                self._clients.remove(q)

    def _broadcast(self, event: Dict[str, Any]) -> None:
        dead: List[queue.Queue] = []
        with self._lock:
            clients = list(self._clients)
        for q in clients:
            try:
                q.put_nowait(event)
            except queue.Full:
                dead.append(q)
        for q in dead:
            self.unsubscribe(q)

    def apply(self, event: Dict[str, Any]) -> Dict[str, Any]:
        et = str(event.get("type") or "command")
        now = time.time()
        out = {"type": et, "ts": event.get("ts", now)}

        with self._lock:
            if et == "audio_busy":
                ms = float(event.get("ms") or event.get("duration_ms") or 500)
                until = time.time() + max(0.0, ms) / 1000.0
                self._audio_busy_until = max(self._audio_busy_until, until)
                out.update(self._gate_dict(time.time(), self._audio_busy_until))
                # No SSE spam for gate ticks.
                return out
            if et == "audio_idle":
                self._audio_busy_until = 0.0
                out.update(self._gate_dict(time.time(), 0.0))
                return out
            if et == "wake":
                self._cancel_reset_timer()
                keep_lights = bool(self.state.get("lights"))
                self.state = blank_room(listening=True, lights=keep_lights)
                self._schedule_reset()
                out.update({"listening": True, "state": dict(self.state)})
            elif et == "reset":
                self._cancel_reset_timer()
                self.state = blank_room()
                out.update({"state": dict(self.state)})
            elif et == "listen_end":
                self.state["listening"] = False
                if not self.state.get("caption") or self.state["caption"] == "Listening…":
                    self.state["caption"] = ""
                out.update({"listening": False, "state": dict(self.state)})
            elif et == "reject":
                # Command not understood: keep props, tell the user.
                line = "I did not understand that"
                self.state["listening"] = False
                self.state["caption"] = line
                self.state["spoken"] = line
                self._schedule_reset()
                out.update({"listening": False, "state": dict(self.state)})
            elif et == "command":
                intent = str(event.get("intent") or "")
                slot = str(event.get("slot") or "")
                spoken = str(event.get("spoken") or "")
                self.state["listening"] = False
                self.state["spoken"] = spoken
                self.state["caption"] = spoken
                self._apply_intent(intent, slot)
                self._schedule_reset()
                out.update({
                    "intent": intent,
                    "slot": slot,
                    "spoken": spoken,
                    "state": dict(self.state),
                })
            else:
                out["state"] = dict(self.state)

        self._broadcast(out)
        return out

    def _cancel_reset_timer(self) -> None:
        self._epoch += 1
        t = self._reset_timer
        self._reset_timer = None
        if t is not None:
            t.cancel()

    def _schedule_reset(self) -> None:
        self._cancel_reset_timer()
        if self._reset_after <= 0:
            return
        epoch = self._epoch
        timer = threading.Timer(self._reset_after, self._idle_reset, args=(epoch,))
        timer.daemon = True
        self._reset_timer = timer
        timer.start()

    def _idle_reset(self, epoch: int) -> None:
        with self._lock:
            if epoch != self._epoch:
                return
            self._reset_timer = None
            keep_lights = bool(self.state.get("lights"))
            self.state = blank_room(lights=keep_lights)
            snap = dict(self.state)
        self._broadcast({"type": "reset", "ts": time.time(), "state": snap})

    def _apply_intent(self, intent: str, slot: str) -> None:
        s = self.state
        if intent == "LIGHT_ON":
            s["lights"] = True
        elif intent == "LIGHT_OFF":
            s["lights"] = False
        elif intent == "PLAY_MUSIC":
            s["music"] = "playing"
        elif intent == "PAUSE":
            if s["music"] == "playing":
                s["music"] = "paused"
        elif intent == "STOP":
            s["music"] = "stopped"
        elif intent == "NEXT":
            s["track"] = int(s.get("track") or 1) % 9 + 1
            s["music"] = "playing"
        elif intent == "VOLUME_UP":
            s["volume"] = min(10, int(s.get("volume") or 5) + 1)
        elif intent == "VOLUME_DOWN":
            s["volume"] = max(0, int(s.get("volume") or 5) - 1)
        elif intent == "CALL":
            s["phone"] = "ringing"
        elif intent == "MESSAGE":
            s["message"] = "New message for you!"
        elif intent == "WEATHER":
            wx = fetch_manila_weather(force=True)
            s["weather"] = wx["key"]
            s["live_weather"] = wx["key"]
            s["weather_label"] = wx["label"]
            s["weather_spoken"] = wx["spoken"]
            s["curtains_open"] = True
            s["caption"] = wx["spoken"]
            s["spoken"] = wx["spoken"]
        elif intent == "TIME":
            s["show_time"] = True
        elif intent == "TIMER":
            s["timer"] = slot or "1 minute"
        elif intent == "ALARM":
            s["alarm"] = slot or "8:00 AM"
        elif intent == "BRIGHTNESS":
            # slot like "60 percent"
            digits = "".join(c for c in slot if c.isdigit())
            if digits:
                s["brightness"] = max(0, min(100, int(digits)))
        elif intent == "COLOR":
            s["color"] = (slot or "warm").lower()
        elif intent == "TEMPERATURE":
            s["ac"] = True
            digits = "".join(c for c in slot if c.isdigit())
            if digits:
                s["temperature"] = int(digits)
        elif intent == "LIST_REMINDERS":
            if not s["reminders"]:
                s["reminders"] = ["Walk the dog", "Water plants"]
        elif intent == "CREATE_REMINDER":
            note = slot or "New reminder"
            notes = list(s.get("reminders") or [])
            notes.append(note)
            s["reminders"] = notes[-6:]


HUB = RoomHub()


class QuietThreadingHTTPServer(ThreadingHTTPServer):
    """Browsers often reset EventSource/fetch sockets; don't dump a traceback."""

    def handle_error(self, request, client_address) -> None:  # noqa: ANN001
        import sys
        err = sys.exc_info()[1]
        if isinstance(err, (ConnectionResetError, BrokenPipeError, ConnectionAbortedError)):
            return
        super().handle_error(request, client_address)


class Handler(BaseHTTPRequestHandler):
    server_version = "KakiTamagotchi/1.0"

    def log_message(self, fmt: str, *args) -> None:
        # Quiet by default; uncomment for debug.
        pass

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/events":
            self._sse()
            return
        if path == "/weather":
            body = json.dumps(fetch_manila_weather()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/state":
            body = json.dumps({"state": HUB.state}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/audio-gate":
            body = json.dumps(HUB.audio_gate()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)
            return
        if path in ("/", "/index.html"):
            return self._file(STATIC / "index.html")
        # static assets
        rel = path.lstrip("/")
        cand = (STATIC / rel).resolve()
        if not str(cand).startswith(str(STATIC.resolve())):
            self.send_error(403)
            return
        if cand.is_file():
            return self._file(cand)
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path != "/event":
            self.send_error(404)
            return
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b"{}"
        try:
            event = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self.send_error(400, "invalid json")
            return
        if not isinstance(event, dict):
            self.send_error(400, "expected object")
            return
        result = HUB.apply(event)
        body = json.dumps(result).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path) -> None:
        data = path.read_bytes()
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self._cors()
        self.end_headers()
        self.wfile.write(data)

    def _sse(self) -> None:
        q = HUB.subscribe()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self._cors()
        self.end_headers()
        # snapshot
        snap = {"type": "hello", "state": dict(HUB.state), "ts": time.time()}
        try:
            self.wfile.write(f"data: {json.dumps(snap)}\n\n".encode())
            self.wfile.flush()
            while True:
                try:
                    ev = q.get(timeout=15.0)
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    continue
                self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        finally:
            HUB.unsubscribe(q)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument(
        "--reset-after",
        type=float,
        default=RESET_AFTER_SEC,
        help="Seconds after an accepted command before the room resets (0 disables).",
    )
    args = ap.parse_args(argv)
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        raise SystemExit("Refusing non-localhost bind (demo is localhost-only).")
    global HUB
    HUB = RoomHub(reset_after=args.reset_after)
    httpd = QuietThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Kaki Tamagotchi demo → http://{args.host}:{args.port}/", flush=True)
    print(f"Room resets {args.reset_after:g}s after a command, or immediately on wake.", flush=True)
    print("Open the URL (optional den UI). Voice works in the assistant terminal alone.", flush=True)
    print("Click the den page once to unlock browser sound (autoplay policy).", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
