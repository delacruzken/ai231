"""Spotify Web API playback for ME2 voice intents (Mac Connect / Premium).

Requires one-time OAuth via ``python -m scripts.spotify_auth`` (from me2/).
Credentials: ``me2/.env`` — tokens: ``me2/.spotify_tokens.json`` (gitignored).
"""
from __future__ import annotations

import base64
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

# Package file: me2/grammar_asr/runtime/*.py → parents[2] = me2/
# Flat Pi file:  ~/me2/*.py → parent = ~/me2/
_file = Path(__file__).resolve()
if _file.parent.name == "runtime" and _file.parents[1].name == "grammar_asr":
    ME2_ROOT = _file.parents[2]
else:
    ME2_ROOT = _file.parent
ENV_PATH = ME2_ROOT / ".env"
TOKEN_PATH = ME2_ROOT / ".spotify_tokens.json"
API = "https://api.spotify.com/v1"
ACCOUNTS = "https://accounts.spotify.com"

DEFAULT_PLAYLIST_URI = "spotify:playlist:3eurayPQQ7IvoqkkletA89"
DEFAULT_REDIRECT = "http://127.0.0.1:8888/callback"
# Minimal player scopes (enough to shuffle-play a playlist URI you already know).
SCOPES = " ".join([
    "user-modify-playback-state",
    "user-read-playback-state",
])


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


def _json_request(
    method: str,
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    data: Optional[bytes] = None,
    form: Optional[Dict[str, str]] = None,
) -> tuple[int, Any]:
    hdrs = dict(headers or {})
    body = data
    if form is not None:
        body = urllib.parse.urlencode(form).encode("utf-8")
        hdrs.setdefault("Content-Type", "application/x-www-form-urlencoded")
    req = urllib.request.Request(url, data=body, headers=hdrs, method=method)
    def _parse_body(raw: bytes) -> Any:
        if not raw or not raw.strip():
            return None
        text = raw.decode("utf-8", errors="replace").strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Spotify player endpoints often return 204 / empty / non-JSON.
            return {"_raw": text}

    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            raw = resp.read()
            code = resp.getcode()
            return code, _parse_body(raw)
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        payload = _parse_body(raw)
        if payload is None:
            payload = {"error": f"http_{exc.code}", "reason": str(exc.reason)}
        return int(exc.code), payload


class SpotifyController:
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        *,
        playlist_uri: str = DEFAULT_PLAYLIST_URI,
        redirect_uri: str = DEFAULT_REDIRECT,
        token_path: Path = TOKEN_PATH,
        verbose: bool = True,
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.playlist_uri = playlist_uri
        self.redirect_uri = redirect_uri
        self.token_path = token_path
        self.verbose = verbose
        self._access: Optional[str] = None
        self._refresh: Optional[str] = None
        self._expires_at = 0.0
        # Alexa duck: save the pre-duck volume once; never stack ducks.
        self._duck_restore: Optional[int] = None
        self._duck_epoch: int = 0
        self._duck_lock = threading.Lock()
        # Last known *real* listening volume (survives corrupt API reads /
        # half-restored ducked levels so we don't spiral 100→20→4→1).
        self._user_volume: Optional[int] = None
        self._load_tokens()

    @classmethod
    def from_env(cls, verbose: bool = True) -> Optional["SpotifyController"]:
        env = load_dotenv()
        # Prefer process env over file so shell exports still work.
        client_id = os.environ.get("SPOTIFY_CLIENT_ID") or env.get("SPOTIFY_CLIENT_ID")
        client_secret = (
            os.environ.get("SPOTIFY_CLIENT_SECRET") or env.get("SPOTIFY_CLIENT_SECRET")
        )
        if not client_id or not client_secret:
            return None
        playlist = (
            os.environ.get("SPOTIFY_PLAYLIST_URI")
            or env.get("SPOTIFY_PLAYLIST_URI")
            or DEFAULT_PLAYLIST_URI
        )
        redirect = (
            os.environ.get("SPOTIFY_REDIRECT_URI")
            or env.get("SPOTIFY_REDIRECT_URI")
            or DEFAULT_REDIRECT
        )
        return cls(
            client_id,
            client_secret,
            playlist_uri=playlist,
            redirect_uri=redirect,
            verbose=verbose,
        )

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[spotify] {msg}", flush=True)

    def _load_tokens(self) -> None:
        if not self.token_path.is_file():
            return
        try:
            data = json.loads(self.token_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        self._access = data.get("access_token")
        self._refresh = data.get("refresh_token")
        self._expires_at = float(data.get("expires_at") or 0)

    def _save_tokens(self) -> None:
        payload = {
            "access_token": self._access,
            "refresh_token": self._refresh,
            "expires_at": self._expires_at,
        }
        self.token_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def store_token_response(self, data: Dict[str, Any]) -> None:
        self._access = data.get("access_token")
        if data.get("refresh_token"):
            self._refresh = data["refresh_token"]
        expires_in = int(data.get("expires_in") or 3600)
        self._expires_at = time.time() + max(30, expires_in - 60)
        self._save_tokens()

    def _basic_auth_header(self) -> str:
        raw = f"{self.client_id}:{self.client_secret}".encode("utf-8")
        return "Basic " + base64.b64encode(raw).decode("ascii")

    def refresh_access_token(self) -> bool:
        if not self._refresh:
            self._log("no refresh token — run: python -m scripts.spotify_auth")
            return False
        # Prefer confidential refresh; fall back to PKCE-style (client_id only).
        attempts = [
            {
                "headers": {"Authorization": self._basic_auth_header()},
                "form": {
                    "grant_type": "refresh_token",
                    "refresh_token": self._refresh,
                },
            },
            {
                "headers": {},
                "form": {
                    "grant_type": "refresh_token",
                    "refresh_token": self._refresh,
                    "client_id": self.client_id,
                },
            },
        ]
        data = None
        code = 0
        for attempt in attempts:
            code, data = _json_request(
                "POST",
                f"{ACCOUNTS}/api/token",
                headers=attempt["headers"] or None,
                form=attempt["form"],
            )
            if code == 200 and isinstance(data, dict) and data.get("access_token"):
                self.store_token_response(data)
                self._log("access token refreshed")
                return True
        self._log(f"token refresh failed ({code}): {data}")
        return False

    def access_token(self) -> Optional[str]:
        if self._access and time.time() < self._expires_at:
            return self._access
        if self.refresh_access_token():
            return self._access
        return self._access

    def _api(
        self,
        method: str,
        path: str,
        *,
        query: Optional[Dict[str, str]] = None,
        body: Optional[dict] = None,
    ) -> tuple[int, Any]:
        token = self.access_token()
        if not token:
            return 401, {"error": "not_authorized"}
        url = f"{API}{path}"
        if query:
            url += "?" + urllib.parse.urlencode(query)
        headers = {"Authorization": f"Bearer {token}"}
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        code, payload = _json_request(method, url, headers=headers, data=data)
        if code == 401:
            if self.refresh_access_token():
                headers["Authorization"] = f"Bearer {self._access}"
                code, payload = _json_request(method, url, headers=headers, data=data)
        return code, payload

    def list_devices(self) -> List[dict]:
        code, data = self._api("GET", "/me/player/devices")
        if code != 200 or not isinstance(data, dict):
            self._log(f"devices failed ({code}): {data}")
            return []
        return list(data.get("devices") or [])

    def pick_device_id(self) -> Optional[str]:
        devices = self.list_devices()
        if not devices:
            self._log(
                "no Spotify Connect devices — open the Mac Spotify app and play "
                "any track once, then retry"
            )
            return None
        # Prefer active, then Computer, then first.
        for d in devices:
            if d.get("is_active"):
                return str(d.get("id"))
        for d in devices:
            name = str(d.get("name") or "").lower()
            dtype = str(d.get("type") or "").lower()
            if "mac" in name or dtype == "computer":
                return str(d.get("id"))
        return str(devices[0].get("id"))

    def play_playlist_shuffled(self, playlist_uri: Optional[str] = None) -> bool:
        uri = playlist_uri or self.playlist_uri
        device_id = self.pick_device_id()
        q: Dict[str, str] = {"state": "true"}
        if device_id:
            q["device_id"] = device_id

        code, data = self._api("PUT", "/me/player/shuffle", query=q)
        if code not in (200, 204):
            self._log(f"shuffle failed ({code}): {data}")

        play_q = {"device_id": device_id} if device_id else None
        code, data = self._api(
            "PUT",
            "/me/player/play",
            query=play_q,
            body={"context_uri": uri},
        )
        if code in (200, 204):
            self._log(f"playing shuffled {uri}")
            return True
        # Common: no active device until Spotify app is open / recently playing.
        self._log(f"play failed ({code}): {data}")
        if code == 404:
            self._log(
                "tip: open Mac Spotify, play any song once, then retry "
                "--test-play"
            )
        return False

    def pause(self) -> bool:
        device_id = self.pick_device_id()
        q = {"device_id": device_id} if device_id else None
        code, data = self._api("PUT", "/me/player/pause", query=q)
        ok = code in (200, 204)
        self._log("paused" if ok else f"pause failed ({code}): {data}")
        return ok

    def next_track(self) -> bool:
        device_id = self.pick_device_id()
        q = {"device_id": device_id} if device_id else None
        code, data = self._api("POST", "/me/player/next", query=q)
        ok = code in (200, 204)
        self._log("next track" if ok else f"next failed ({code}): {data}")
        return ok

    def get_playback(self) -> Dict[str, Any]:
        code, data = self._api("GET", "/me/player")
        if code != 200 or not isinstance(data, dict):
            return {"is_playing": False, "volume_percent": None, "raw": data}
        device = data.get("device") or {}
        vol = device.get("volume_percent")
        return {
            "is_playing": bool(data.get("is_playing")),
            "volume_percent": int(vol) if vol is not None else None,
            "device_id": device.get("id"),
        }

    def set_volume(self, percent: int, device_id: Optional[str] = None) -> bool:
        new_vol = max(0, min(100, int(percent)))
        q = {"volume_percent": str(new_vol)}
        dev = device_id or self.pick_device_id()
        if dev:
            q["device_id"] = str(dev)
        code, data = self._api("PUT", "/me/player/volume", query=q)
        ok = code in (200, 204)
        self._log(
            f"volume {new_vol}%" if ok else f"volume failed ({code}): {data}"
        )
        return ok

    # Below this, a reported volume is treated as "already ducked / corrupt"
    # unless the user explicitly lowered volume while not ducked.
    _MIN_SANE_BASELINE = 40

    def _remember_user_volume(self, percent: int) -> None:
        p = max(0, min(100, int(percent)))
        if p >= self._MIN_SANE_BASELINE:
            self._user_volume = p

    def _resolve_duck_baseline(self, reported_vol: int) -> int:
        """Pick a baseline that won't spiral down from a half-restored duck."""
        reported = max(0, min(100, int(reported_vol)))
        remembered = self._user_volume
        ducked_like = reported < self._MIN_SANE_BASELINE
        if remembered is not None and ducked_like:
            baseline = int(remembered)
            self._log(
                f"duck baseline recover: API {reported}% looks ducked; "
                f"using remembered {baseline}%"
            )
            return baseline
        if ducked_like and remembered is None:
            # Completely lost — prefer a normal listening level over 4%→1%.
            baseline = 100
            self._log(
                f"duck baseline recover: API {reported}% with no memory; "
                f"using {baseline}%"
            )
            self._user_volume = baseline
            return baseline
        self._remember_user_volume(reported)
        return reported

    def adjust_volume(self, delta: int) -> bool:
        pb = self.get_playback()
        vol = pb.get("volume_percent")
        if vol is None:
            vol = 50
        device_id = pb.get("device_id")
        with self._duck_lock:
            if self._duck_restore is not None:
                # Nudge the *baseline* only — do NOT bump epoch. A pending
                # unduck must still fire and restore to this new baseline.
                baseline = max(0, min(100, int(self._duck_restore) + int(delta)))
                self._duck_restore = baseline
                self._user_volume = baseline  # explicit voice volume wins
                ducked = max(1, min(100, int(round(baseline * 0.20))))
                self._log(
                    f"volume while ducked: baseline {baseline}%, play {ducked}% "
                    f"(unduck still epoch {self._duck_epoch})"
                )
                target, device = ducked, device_id
            else:
                target = max(0, min(100, int(vol) + int(delta)))
                self._user_volume = target
                device = device_id
        return self.set_volume(target, device)

    def duck_for_listen(self, fraction: float = 0.20) -> bool:
        """Alexa-style: drop to ``fraction`` of the *original* volume (no stacking)."""
        pb = self.get_playback()
        if not pb.get("is_playing"):
            return False
        vol = pb.get("volume_percent")
        if vol is None:
            return False
        device_id = pb.get("device_id")
        with self._duck_lock:
            if self._duck_restore is not None:
                # Still in a duck session — keep / heal baseline; cancel stale unducks.
                baseline = self._resolve_duck_baseline(int(self._duck_restore))
                self._duck_restore = baseline
                self._duck_epoch += 1
                ducked = max(1, min(100, int(round(baseline * float(fraction)))))
                self._log(
                    f"duck keep baseline {baseline}% → {ducked}% "
                    f"(epoch {self._duck_epoch})"
                )
            else:
                baseline = self._resolve_duck_baseline(int(vol))
                ducked = max(1, min(100, int(round(baseline * float(fraction)))))
                if ducked >= baseline:
                    return False
                self._duck_restore = baseline
                self._duck_epoch += 1
                self._log(
                    f"duck {baseline}% → {ducked}% for listen "
                    f"(epoch {self._duck_epoch})"
                )
            epoch = self._duck_epoch
        ok = self.set_volume(ducked, device_id)
        if not ok:
            with self._duck_lock:
                if self._duck_epoch == epoch and self._duck_restore == baseline:
                    self._duck_restore = None
        return ok

    def restore_after_duck(self, expected_epoch: Optional[int] = None) -> bool:
        """Restore pre-duck volume. Ignore if a newer duck superseded this unduck."""
        with self._duck_lock:
            if self._duck_restore is None:
                return False
            if expected_epoch is not None and expected_epoch != self._duck_epoch:
                self._log(
                    f"skip stale restore "
                    f"(epoch {expected_epoch}≠{self._duck_epoch})"
                )
                return False
            target = self._resolve_duck_baseline(int(self._duck_restore))
            self._duck_restore = target
            epoch = self._duck_epoch
        self._log(f"restore volume → {target}% (epoch {epoch})")
        ok = self.set_volume(target)
        with self._duck_lock:
            if expected_epoch is not None and expected_epoch != self._duck_epoch:
                self._log("restore superseded mid-flight; keeping duck state")
                return False
            if self._duck_epoch != epoch:
                return False
            if ok:
                self._duck_restore = None
                self._duck_epoch += 1
                self._remember_user_volume(target)
        return ok

    def handle_intent(self, intent: str, slot: str = "") -> None:
        intent = (intent or "").upper()
        if intent == "PLAY_MUSIC":
            # Don't leave the new track at ducked volume.
            self.restore_after_duck()
            self.play_playlist_shuffled()
        elif intent == "PAUSE":
            self.pause()
            self.restore_after_duck()
        elif intent == "STOP":
            self.pause()
            self.restore_after_duck()
        elif intent == "NEXT":
            self.next_track()
        elif intent == "VOLUME_UP":
            self.adjust_volume(+10)
        elif intent == "VOLUME_DOWN":
            self.adjust_volume(-10)


_CTRL: Optional[SpotifyController] = None


def _shared_controller(verbose: bool = True) -> Optional[SpotifyController]:
    global _CTRL
    if _CTRL is None:
        _CTRL = SpotifyController.from_env(verbose=verbose)
    elif verbose:
        _CTRL.verbose = verbose
    return _CTRL


def maybe_handle_intent(intent: str, slot: str = "", *, verbose: bool = True) -> None:
    """Fire-and-forget Spotify side effect when credentials exist."""
    ctrl = _shared_controller(verbose=verbose)
    if ctrl is None:
        return
    if not ctrl._refresh and not ctrl._access:
        if verbose:
            print(
                "[spotify] not authorized yet — run: "
                "python -m scripts.spotify_auth",
                flush=True,
            )
        return

    def worker() -> None:
        try:
            ctrl.handle_intent(intent, slot)
        except Exception as exc:  # pragma: no cover
            if verbose:
                print(f"[spotify] error: {exc}", flush=True)

    threading.Thread(target=worker, daemon=True).start()


def maybe_duck_for_listen(*, verbose: bool = True, fraction: float = 0.20) -> None:
    """Lower Spotify volume quickly on wake (background thread)."""
    ctrl = _shared_controller(verbose=verbose)
    if ctrl is None or (not ctrl._refresh and not ctrl._access):
        return

    def worker() -> None:
        try:
            ctrl.duck_for_listen(fraction=fraction)
        except Exception as exc:  # pragma: no cover
            if verbose:
                print(f"[spotify] duck error: {exc}", flush=True)

    threading.Thread(target=worker, daemon=True).start()


def maybe_restore_after_duck(*, verbose: bool = True, delay_s: float = 0.0) -> None:
    """Restore pre-wake Spotify volume after the listen window / TTS.

    Pass ``delay_s`` so volume stays low while the den speaks (weather, etc.).
    Captures the duck epoch so a later wake cancels this unduck (avoids
    restoring mid-listen or stacking ducks off a half-restored volume).
    """
    ctrl = _shared_controller(verbose=verbose)
    if ctrl is None or (not ctrl._refresh and not ctrl._access):
        return
    with ctrl._duck_lock:
        if ctrl._duck_restore is None:
            return
        epoch = ctrl._duck_epoch

    def worker() -> None:
        try:
            wait = max(0.0, float(delay_s))
            if wait:
                if verbose:
                    print(
                        f"[spotify] keep ducked {wait:.1f}s for spoken reply "
                        f"(epoch {epoch})",
                        flush=True,
                    )
                time.sleep(wait)
            ctrl.restore_after_duck(expected_epoch=epoch)
        except Exception as exc:  # pragma: no cover
            if verbose:
                print(f"[spotify] unduck error: {exc}", flush=True)

    threading.Thread(target=worker, daemon=True).start()
