"""HTTP adapter for OpenCode serve v2 (/api/* + basic auth)."""

from __future__ import annotations

import base64
import json
import logging
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urljoin

logger = logging.getLogger("me3.delegation.opencode")


class OpenCodeClient:
    """OpenCode 2.x HTTP API (paths under /api/)."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:4096",
        *,
        username: str = "opencode",
        password: str = "",
        timeout_s: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/") + "/"
        self.username = username or "opencode"
        self.password = password or ""
        self.timeout_s = timeout_s

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.password:
            token = base64.b64encode(f"{self.username}:{self.password}".encode()).decode()
            headers["Authorization"] = f"Basic {token}"
        return headers

    def request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        url = urljoin(self.base_url, path.lstrip("/"))
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=self._headers(), method=method.upper())
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read()
                if not raw:
                    return None
                parsed = json.loads(raw.decode("utf-8"))
                # v2 wraps many payloads as {"data": ...}
                if isinstance(parsed, dict) and "data" in parsed and len(parsed.keys()) <= 3:
                    return parsed["data"]
                return parsed
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"OpenCode {method} {path} HTTP {exc.code}: {detail}") from exc
        except Exception as exc:
            raise RuntimeError(f"OpenCode {method} {path} failed: {exc}") from exc

    def health(self) -> dict[str, Any]:
        """v2 has no /global/health; probe an authenticated API route instead."""
        try:
            # Prefer session list — exists on v2 and requires auth when password set.
            self.request("GET", "/api/session")
            return {"healthy": True, "provider": "opencode", "api": "v2"}
        except Exception as exc:
            # Fallback: older docs path (v1)
            try:
                data = self.request("GET", "/global/health")
                if isinstance(data, dict) and data.get("healthy"):
                    return {"healthy": True, "provider": "opencode", "api": "v1", "version": data.get("version")}
            except Exception:
                pass
            return {"healthy": False, "provider": "opencode", "error": str(exc)[:200]}

    def create_session(self, title: str | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if title:
            body["title"] = title
        data = self.request("POST", "/api/session", body)
        if not isinstance(data, dict):
            raise RuntimeError(f"unexpected create session response: {data!r}")
        return data

    def send_message_async(self, session_id: str, text: str, *, system: str | None = None) -> None:
        # v2 prompt API uses {text}; system prepended into text if provided.
        payload_text = text
        if system:
            payload_text = f"{system.strip()}\n\n{text}"
        body: dict[str, Any] = {"text": payload_text}
        # Prefer async-friendly prompt; v2 prompt returns when accepted / completed per server.
        try:
            self.request("POST", f"/api/session/{session_id}/prompt", body)
        except RuntimeError:
            # v1 fallback
            parts = [{"type": "text", "text": text}]
            legacy = {"parts": parts}
            if system:
                legacy["system"] = system
            self.request("POST", f"/session/{session_id}/prompt_async", legacy)

    def abort(self, session_id: str) -> bool:
        try:
            self.request("POST", f"/api/session/{session_id}/interrupt")
            return True
        except RuntimeError:
            try:
                return bool(self.request("POST", f"/session/{session_id}/abort"))
            except RuntimeError:
                return False

    def respond_permission(self, session_id: str, permission_id: str, response: str) -> bool:
        decision = "allow" if response.lower() in ("allow", "approve", "yes", "approved") else "deny"
        try:
            self.request(
                "POST",
                f"/api/session/{session_id}/permission/{permission_id}/reply",
                {"decision": decision, "message": response},
            )
            return True
        except RuntimeError:
            try:
                return bool(
                    self.request(
                        "POST",
                        f"/session/{session_id}/permissions/{permission_id}",
                        {"response": response},
                    )
                )
            except RuntimeError:
                return False

    def get_session(self, session_id: str) -> dict[str, Any]:
        try:
            data = self.request("GET", f"/api/session/{session_id}")
        except RuntimeError:
            data = self.request("GET", f"/session/{session_id}")
        if isinstance(data, dict):
            return data
        return {}

    def list_messages(self, session_id: str, *, limit: int = 20) -> list[dict[str, Any]]:
        try:
            data = self.request("GET", f"/api/session/{session_id}/message?limit={limit}")
        except RuntimeError:
            data = self.request("GET", f"/session/{session_id}/message?limit={limit}")
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            # possible shapes: {items:[]} or nested
            for key in ("items", "messages", "data"):
                if isinstance(data.get(key), list):
                    return data[key]
        return []
