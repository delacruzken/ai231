"""HTTP session API smoke with TestClient (no mic)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("ME3_UI_TOKEN", "test-token")
    from me3.config import load_secrets
    from me3.session.controller import SessionController, SessionState
    from me3.app.server import create_app

    ctrl = SessionController()
    # Avoid real audio/model startup in lifespan
    ctrl.start_backend = MagicMock()
    ctrl.shutdown = MagicMock()
    ctrl.probe_readiness = MagicMock(return_value=ctrl.readiness)
    ctrl.readiness.audio = True
    ctrl.readiness.vad = True
    ctrl.readiness.asr = True
    ctrl.readiness.tts = True

    def _set(enabled: bool):
        ctrl.enabled = enabled
        ctrl.state = SessionState.LISTENING if enabled else SessionState.OFF
        return ctrl.snapshot()

    ctrl.set_enabled = _set  # type: ignore

    app = create_app(controller=ctrl)
    # bypass lifespan heavy work: override
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_ui_bootstrap_returns_token(client):
    r = client.get("/api/ui-bootstrap")
    assert r.status_code == 200
    body = r.json()
    assert body["token"] == "test-token"
    assert body["require_token"] is True


def test_session_requires_token(client):
    r = client.get("/api/session")
    assert r.status_code == 401


def test_session_toggle(client):
    headers = {"Authorization": "Bearer test-token"}
    r = client.get("/api/session", headers=headers)
    assert r.status_code == 200
    assert r.json()["enabled"] is False

    r = client.post("/api/session", headers=headers, json={"enabled": True})
    assert r.status_code == 200
    assert r.json()["enabled"] is True


def test_session_event_helper_accepts_snapshot():
    from me3.app.server import _session_event
    from me3.session.controller import SessionController

    ctrl = SessionController()
    ctrl.start_backend = MagicMock()
    ctrl.shutdown = MagicMock()
    payload = _session_event(ctrl.snapshot())
    assert payload["type"] == "session"
    assert "enabled" in payload
    assert "state" in payload
    # Must not raise TypeError when building JSON-like dict
    assert isinstance(payload, dict)
