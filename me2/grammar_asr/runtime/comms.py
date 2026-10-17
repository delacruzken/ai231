"""Fixed FaceTime / Messages actions via macOS URL schemes (Mac-in-the-middle).

Pi voice → POST /event → Mac den server (or Mac-local assistant) opens:

  facetime://TARGET          or facetime-audio://TARGET
  sms:TARGET&body=BODY       (Messages compose; optional AppleScript send)

Config in ``me2/.env`` (never commit):

  ME2_CALL_TARGET=+63917xxxxxxx
  ME2_CALL_MODE=facetime          # or facetime-audio
  ME2_CALL_AUTOSEND=0             # 1 = click Call via Accessibility (UI scripting)
  ME2_MESSAGE_TARGET=+63917xxxxxxx
  ME2_MESSAGE_BODY=Hey — this is Kaki checking in.
  ME2_MESSAGE_AUTOSEND=0          # 1 = try Messages AppleScript send
"""
from __future__ import annotations

import os
import platform
import subprocess
import time
import urllib.parse
from pathlib import Path
from typing import Dict, Optional

# Package file: me2/grammar_asr/runtime/*.py → parents[2] = me2/
# Flat Pi file:  ~/me2/*.py → parent = ~/me2/
_file = Path(__file__).resolve()
if _file.parent.name == "runtime" and _file.parents[1].name == "grammar_asr":
    ME2_ROOT = _file.parents[2]
else:
    ME2_ROOT = _file.parent
ENV_PATH = ME2_ROOT / ".env"


def load_dotenv(path: Path = ENV_PATH) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _env(key: str, default: str = "") -> str:
    dotenv = load_dotenv()
    return (os.environ.get(key) or dotenv.get(key) or default).strip()


def _normalize_target(raw: str) -> str:
    """Keep + and digits/email; strip spaces and dashes for tel-like targets."""
    t = (raw or "").strip()
    if "@" in t:
        return t
    return "".join(c for c in t if c.isdigit() or c == "+")


def _open_url(url: str, *, verbose: bool = True) -> bool:
    if platform.system() != "Darwin":
        if verbose:
            print(f"[comms] skip (not macOS): {url}", flush=True)
        return False
    try:
        subprocess.run(["open", url], check=False, timeout=10)
        if verbose:
            print(f"[comms] open {url}", flush=True)
        return True
    except (OSError, subprocess.SubprocessError) as exc:
        if verbose:
            print(f"[comms] open failed: {exc}", flush=True)
        return False


def _applescript_click_facetime_call(*, verbose: bool = True) -> bool:
    """Click the macOS FaceTime confirmation (UI scripting; needs Accessibility).

    Apple always shows a confirm step for ``facetime://`` on Mac. We open the
    URL, then try to press Call in FaceTime and/or Notification Center.
    """
    script = '''
tell application "System Events"
  set clicked to false
  repeat with i from 1 to 40
    try
      if exists (process "FaceTime") then
        tell process "FaceTime"
          set frontmost to true
          if exists (button "Call" of window 1) then
            click button "Call" of window 1
            set clicked to true
            exit repeat
          end if
        end tell
      end if
    end try
    try
      tell process "NotificationCenter"
        if exists (button "Call" of window 1) then
          click button "Call" of window 1
          set clicked to true
          exit repeat
        end if
      end tell
    end try
    try
      tell process "Notification Center"
        if exists (button "Call" of window 1) then
          click button "Call" of window 1
          set clicked to true
          exit repeat
        end if
      end tell
    end try
    delay 0.25
  end repeat
  return clicked
end tell
'''
    try:
        r = subprocess.run(
            ["osascript", "-e", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        ok = r.returncode == 0 and "true" in (r.stdout or "").lower()
        if verbose:
            if ok:
                print("[comms] FaceTime Call button clicked", flush=True)
            else:
                err = (r.stderr or r.stdout or "no Call button found").strip()
                print(f"[comms] FaceTime autosend click failed: {err}", flush=True)
        return ok
    except (OSError, subprocess.SubprocessError) as exc:
        if verbose:
            print(f"[comms] FaceTime autosend error: {exc}", flush=True)
        return False


def place_call(*, verbose: bool = True) -> bool:
    target = _normalize_target(_env("ME2_CALL_TARGET"))
    if not target:
        if verbose:
            print(
                "[comms] CALL skipped — set ME2_CALL_TARGET in me2/.env",
                flush=True,
            )
        return False
    mode = (_env("ME2_CALL_MODE", "facetime") or "facetime").lower()
    scheme = "facetime-audio" if mode in ("facetime-audio", "audio", "audio-only") else "facetime"
    opened = _open_url(f"{scheme}://{target}", verbose=verbose)
    if not opened:
        return False
    autosend = _env("ME2_CALL_AUTOSEND", "0").lower() in ("1", "true", "yes", "on")
    if not autosend:
        if verbose:
            print(
                "[comms] CALL opened — confirm in FaceTime "
                "(or set ME2_CALL_AUTOSEND=1)",
                flush=True,
            )
        return True
    # Give FaceTime a beat to show the confirm UI, then click Call.
    time.sleep(0.6)
    return _applescript_click_facetime_call(verbose=verbose)


def _applescript_send_imessage(target: str, body: str, *, verbose: bool = True) -> bool:
    """Best-effort auto-send; needs Messages Automation permission on the Mac."""
    # Escape for AppleScript string literals.
    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace('"', '\\"')

    script = f'''
tell application "Messages"
  set targetService to 1st account whose service type = iMessage
  set targetBuddy to participant "{esc(target)}" of targetService
  send "{esc(body)}" to targetBuddy
end tell
'''
    try:
        r = subprocess.run(
            ["osascript", "-e", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        if r.returncode != 0:
            # Fallback: SMS/RCS buddy style (phone number as chat id).
            script2 = f'''
tell application "Messages"
  set targetBuddy to buddy "{esc(target)}" of (service 1 whose service type is iMessage)
  send "{esc(body)}" to targetBuddy
end tell
'''
            r = subprocess.run(
                ["osascript", "-e", script2],
                check=False,
                capture_output=True,
                text=True,
                timeout=20,
            )
        if r.returncode != 0:
            if verbose:
                err = (r.stderr or r.stdout or "osascript failed").strip()
                print(f"[comms] Messages autosend failed: {err}", flush=True)
            return False
        if verbose:
            print(f"[comms] Messages sent to {target}", flush=True)
        return True
    except (OSError, subprocess.SubprocessError) as exc:
        if verbose:
            print(f"[comms] Messages autosend error: {exc}", flush=True)
        return False


def send_fixed_message(*, verbose: bool = True) -> bool:
    target = _normalize_target(_env("ME2_MESSAGE_TARGET"))
    body = _env("ME2_MESSAGE_BODY", "Hey — this is Kaki checking in.")
    if not target:
        if verbose:
            print(
                "[comms] MESSAGE skipped — set ME2_MESSAGE_TARGET in me2/.env",
                flush=True,
            )
        return False
    autosend = _env("ME2_MESSAGE_AUTOSEND", "0").lower() in ("1", "true", "yes", "on")
    if autosend:
        if _applescript_send_imessage(target, body, verbose=verbose):
            return True
        if verbose:
            print("[comms] falling back to Messages compose URL", flush=True)
    # Opens Messages with recipient + body; user may still need to tap Send.
    q_body = urllib.parse.quote(body, safe="")
    q_target = urllib.parse.quote(target, safe="+")
    return _open_url(f"sms:{q_target}&body={q_body}", verbose=verbose)


def maybe_handle_intent(intent: str, slot: str = "", *, verbose: bool = True) -> Optional[bool]:
    """Handle CALL / MESSAGE; return None if not a comms intent."""
    del slot  # fixed targets for now
    if intent == "CALL":
        return place_call(verbose=verbose)
    if intent == "MESSAGE":
        return send_fixed_message(verbose=verbose)
    return None
