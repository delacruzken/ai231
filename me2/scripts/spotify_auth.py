#!/usr/bin/env python3
"""One-time Spotify OAuth for ME2 (Authorization Code + PKCE).

Prereqs
  1. Spotify Developer Dashboard → your app → Settings
  2. Redirect URI allowlist must include EXACTLY:
       http://127.0.0.1:8888/callback
     (not localhost; click Save after editing)
  3. Development mode: add your Spotify account under User management
  4. me2/.env with SPOTIFY_CLIENT_ID (+ secret recommended)

Usage (from me2/):
  python -m scripts.spotify_auth
  python -m scripts.spotify_auth --test-play
"""
from __future__ import annotations

import base64
import hashlib
import secrets
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import sys

ME2 = Path(__file__).resolve().parents[1]
if str(ME2) not in sys.path:
    sys.path.insert(0, str(ME2))

from grammar_asr.runtime.spotify import (  # noqa: E402
    ACCOUNTS,
    ENV_PATH,
    SCOPES,
    SpotifyController,
    load_dotenv,
    _json_request,
)


def _pkce_pair() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)[:128]
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return verifier, challenge


def main() -> None:
    env = load_dotenv()
    client_id = (env.get("SPOTIFY_CLIENT_ID") or "").strip()
    client_secret = (env.get("SPOTIFY_CLIENT_SECRET") or "").strip()
    redirect = (env.get("SPOTIFY_REDIRECT_URI") or "http://127.0.0.1:8888/callback").strip()
    playlist = (
        env.get("SPOTIFY_PLAYLIST_URI") or "spotify:playlist:3eurayPQQ7IvoqkkletA89"
    ).strip()

    if not client_id:
        raise SystemExit(
            f"Missing SPOTIFY_CLIENT_ID in {ENV_PATH}\n"
            "Create me2/.env from me2/.env.example."
        )

    if redirect != "http://127.0.0.1:8888/callback":
        print(
            f"WARNING: redirect is {redirect!r}\n"
            "Dashboard allowlist must match this EXACTLY "
            "(prefer http://127.0.0.1:8888/callback).",
            flush=True,
        )

    state = secrets.token_urlsafe(16)
    verifier, challenge = _pkce_pair()
    params = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": redirect,
        "scope": SCOPES,
        "state": state,
        "code_challenge_method": "S256",
        "code_challenge": challenge,
    }
    auth_url = f"{ACCOUNTS}/authorize?{urllib.parse.urlencode(params)}"

    print("Checklist before browser opens:", flush=True)
    print(f"  Client ID length : {len(client_id)} (expect 32)", flush=True)
    print(f"  Redirect URI     : {redirect}", flush=True)
    print(f"  Scopes           : {SCOPES}", flush=True)
    print(
        "  Dashboard must list that redirect URI, then Save.\n"
        "  Development mode: add your Spotify user under User management.",
        flush=True,
    )

    result: dict = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # noqa: ANN001
            return

        def do_GET(self):  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path.rstrip("/") != "/callback":
                self.send_error(404)
                return
            qs = urllib.parse.parse_qs(parsed.query)
            if qs.get("state", [None])[0] != state:
                self.send_error(400, "state mismatch")
                return
            if "error" in qs:
                result["error"] = qs["error"][0]
                err = qs["error"][0].encode()
                body = (
                    b"<h1>Authorization failed</h1><p>error="
                    + err
                    + b"</p><p>Usually: redirect URI not saved on the "
                    b"Spotify dashboard, or your account is not added as "
                    b"a Development Mode user.</p>"
                )
            else:
                result["code"] = qs.get("code", [None])[0]
                body = (
                    b"<h1>Spotify authorized</h1>"
                    b"<p>Return to the terminal. You can close this tab.</p>"
                )
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            threading.Thread(target=httpd.shutdown, daemon=True).start()

    host, _, port_s = redirect.removeprefix("http://").partition(":")
    port = int(port_s.split("/")[0])
    try:
        httpd = HTTPServer((host, port), Handler)
    except OSError as exc:
        raise SystemExit(
            f"Cannot bind {redirect}: {exc}\n"
            "Is another spotify_auth still running? "
            "Or is port 8888 already in use?"
        ) from exc

    print("\nOpening Spotify login…", flush=True)
    print(auth_url, flush=True)
    webbrowser.open(auth_url)
    print(f"Listening for callback on {redirect}", flush=True)
    httpd.serve_forever()

    if result.get("error"):
        err = result["error"]
        tips = {
            "server_error": (
                "Spotify rejected the authorize request.\n"
                "  1) Settings → Redirect URIs → add exactly "
                "http://127.0.0.1:8888/callback → Save\n"
                "  2) User management → Add your Spotify account email\n"
                "  3) Confirm Client ID in .env matches this app\n"
                "  4) Try again in a private browser window"
            ),
            "access_denied": "You clicked Cancel on the consent screen.",
            "invalid_client": "Client ID in .env does not match the dashboard app.",
        }.get(err, "See Spotify authorize error docs.")
        raise SystemExit(f"Authorization error: {err}\n{tips}")

    code = result.get("code")
    if not code:
        raise SystemExit("No authorization code received")

    # PKCE token exchange (client_id + code_verifier). Secret optional.
    form = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect,
        "client_id": client_id,
        "code_verifier": verifier,
    }
    headers = {}
    if client_secret:
        ctrl_tmp = SpotifyController(client_id, client_secret, redirect_uri=redirect)
        headers["Authorization"] = ctrl_tmp._basic_auth_header()

    status, data = _json_request(
        "POST",
        f"{ACCOUNTS}/api/token",
        headers=headers or None,
        form=form,
    )
    if status != 200 or not isinstance(data, dict):
        raise SystemExit(f"Token exchange failed ({status}): {data}")

    ctrl = SpotifyController(
        client_id,
        client_secret or "pkce",
        playlist_uri=playlist,
        redirect_uri=redirect,
    )
    ctrl.store_token_response(data)
    print(f"Saved tokens → {ctrl.token_path}", flush=True)
    print(f"Playlist URI → {playlist}", flush=True)
    print("Open the Mac Spotify app, then test:", flush=True)
    print("  python -m scripts.spotify_auth --test-play", flush=True)


def test_play() -> None:
    ctrl = SpotifyController.from_env(verbose=True)
    if ctrl is None:
        raise SystemExit("Missing credentials in me2/.env")
    if not ctrl.access_token():
        raise SystemExit("Not authorized — run without --test-play first")
    ok = ctrl.play_playlist_shuffled()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    if "--test-play" in sys.argv:
        test_play()
    else:
        main()
