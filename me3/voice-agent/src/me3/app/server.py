"""Phase 2 session HTTP + WebSocket backend.

Endpoints:
  GET  /api/health
  GET  /api/session
  POST /api/session  {"enabled": bool}
  POST /api/probe    {"text": "..."} optional TTS probe
  GET  /api/devices
  WS   /ws/session?token=...
  static ui/dist when present

Auth: ME3_UI_TOKEN via Authorization: Bearer <token> or ?token=.
Same-origin UI loads the token from GET /api/ui-bootstrap (no manual paste).
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from me3 import __version__
from me3.config import ROOT, load_config, load_secrets
from me3.session.controller import SessionController
from me3.speech.device import device_snapshot

logger = logging.getLogger("me3.app.server")

UI_DIST = ROOT / "ui" / "dist"


class SessionBody(BaseModel):
    enabled: bool


class ProbeBody(BaseModel):
    text: str = "Session is on."


class SayBody(BaseModel):
    text: str


class OwnerProfileBody(BaseModel):
    display_name: str | None = None
    headline: str | None = None
    about: str | None = None
    links: list[str] | dict | None = None
    locale: str | None = None
    preferences: dict | None = None
    approved: bool = True


class JobSubmitBody(BaseModel):
    specification: str
    instructions: str | None = None


class JobRespondBody(BaseModel):
    response: str = ""
    approve: bool = True


class PublishBody(BaseModel):
    source_dir: str
    message: str | None = None
    confirm: bool = False
    expect_substring: str | None = None


def _token_ok(provided: str | None, required: str, require: bool) -> bool:
    if not require:
        return True
    if not required:
        return False
    return bool(provided) and provided == required


def _session_event(snap: Any) -> dict[str, Any]:
    """SessionSnapshot → WS/JSON payload (never unpack the dataclass itself)."""
    if hasattr(snap, "as_dict"):
        data = snap.as_dict()
    elif isinstance(snap, dict):
        data = snap
    else:
        raise TypeError(f"expected SessionSnapshot or dict, got {type(snap)!r}")
    return {"type": "session", **data}


def create_app(controller: SessionController | None = None) -> FastAPI:
    cfg = load_config()
    secrets = load_secrets()
    ctrl = controller or SessionController(cfg=cfg)
    clients: set[WebSocket] = set()
    loop_holder: dict[str, Any] = {"loop": None}

    def broadcast(event: dict[str, Any]) -> None:
        loop = loop_holder.get("loop")
        if loop is None:
            return
        dead: list[WebSocket] = []

        async def _send_all():
            for ws in list(clients):
                try:
                    await ws.send_json(event)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                clients.discard(ws)

        asyncio.run_coroutine_threadsafe(_send_all(), loop)

    ctrl.on_event = broadcast

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loop_holder["loop"] = asyncio.get_running_loop()
        # Probe in a worker so startup is not blocked forever on model load
        await asyncio.to_thread(ctrl.start_backend)
        logger.info("session backend ready (starts OFF)")
        yield
        await asyncio.to_thread(ctrl.shutdown)

    app = FastAPI(title="ME3 voice-agent", version=__version__, lifespan=lifespan)
    app.state.controller = ctrl
    app.state.cfg = cfg
    app.state.secrets = secrets

    def auth(
        authorization: str | None = Header(default=None),
        token: str | None = Query(default=None),
    ) -> None:
        provided = token
        if authorization and authorization.lower().startswith("bearer "):
            provided = authorization.split(" ", 1)[1].strip()
        required = secrets.me3_ui_token or ""
        if not _token_ok(provided, required, cfg.ui.require_token):
            raise HTTPException(status_code=401, detail="invalid or missing UI token")

    @app.get("/api/health")
    def health():
        return {
            "ok": True,
            "version": __version__,
            "phase": 8,
            "retrieval": getattr(ctrl.tasks, "retrieval_provider", None),
            "delegation": getattr(ctrl.delegation, "provider", None),
            "publication": getattr(ctrl.publication, "provider", None),
        }

    @app.get("/api/ui-bootstrap")
    def ui_bootstrap():
        """Hand the UI its access token so load → connect needs no paste.

        Same-origin only in practice (served with the static UI). API/WS still
        require the token; this just stops operators from typing ME3_UI_TOKEN.
        """
        return {
            "token": secrets.me3_ui_token or "",
            "require_token": bool(cfg.ui.require_token),
        }

    @app.get("/api/session")
    def get_session(_: None = Depends(auth)):
        return ctrl.snapshot().as_dict()

    @app.post("/api/session")
    def post_session(body: SessionBody, _: None = Depends(auth)):
        return ctrl.set_enabled(body.enabled).as_dict()

    @app.post("/api/probe")
    def post_probe(body: ProbeBody, _: None = Depends(auth)):
        try:
            uid = ctrl.speak_probe(body.text)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"utterance_id": uid, "session": ctrl.snapshot().as_dict()}

    @app.post("/api/say")
    def post_say(body: SayBody, _: None = Depends(auth)):
        """Typed user turn (speech-failure recovery / sample bank)."""
        try:
            turn_id = ctrl.inject_user_text(body.text)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"turn_id": turn_id, "session": ctrl.snapshot().as_dict()}

    @app.get("/api/timers")
    def get_timers(status: str | None = Query(default="active"), _: None = Depends(auth)):
        return {"timers": ctrl.tasks.store.list_timers(status=status or None)}

    @app.get("/api/notes")
    def get_notes(q: str = Query(default=""), _: None = Depends(auth)):
        return {"notes": ctrl.tasks.store.search_notes(q)}

    @app.get("/api/notifications")
    def get_notifications(_: None = Depends(auth)):
        return {"pending": ctrl.tasks.drain_pending_notifications()}

    @app.get("/api/profile")
    def get_profile(_: None = Depends(auth)):
        return ctrl.owner_store.get()

    @app.put("/api/profile")
    def put_profile(body: OwnerProfileBody, _: None = Depends(auth)):
        dumped = body.model_dump()
        preferences = dumped.pop("preferences", None)
        approved = bool(dumped.pop("approved", True))
        public = {k: v for k, v in dumped.items() if v is not None}
        profile = ctrl.owner_store.set_public(
            public,
            preferences=preferences,
            approved=approved,
        )
        if ctrl.conversation:
            ctrl.conversation.refresh_system_prompt()
        if approved and public:
            cfg.pending.owner_public_profile = False
        return profile

    @app.get("/api/jobs")
    def get_jobs(_: None = Depends(auth)):
        return {
            "provider": ctrl.delegation.provider,
            "active": ctrl.delegation.registry.active_job(),
            "jobs": ctrl.delegation.registry.list_jobs(limit=20),
        }

    @app.post("/api/jobs")
    def post_job(body: JobSubmitBody, _: None = Depends(auth)):
        result = ctrl.delegation.handle(
            "jobs.submit",
            specification=body.specification,
            instructions=body.instructions,
        )
        if not result.ok:
            raise HTTPException(
                status_code=400,
                detail=result.error.message if result.error else "failed",
            )
        return result.data

    @app.post("/api/jobs/{job_id}/respond")
    def post_job_respond(job_id: str, body: JobRespondBody, _: None = Depends(auth)):
        result = ctrl.delegation.handle(
            "jobs.update",
            job_id=job_id,
            response=body.response,
            approve=body.approve,
        )
        if not result.ok:
            raise HTTPException(
                status_code=400,
                detail=result.error.message if result.error else "failed",
            )
        return result.data

    @app.post("/api/jobs/{job_id}/cancel")
    def post_job_cancel(job_id: str, _: None = Depends(auth)):
        result = ctrl.delegation.handle("jobs.cancel", job_id=job_id)
        if not result.ok:
            raise HTTPException(
                status_code=400,
                detail=result.error.message if result.error else "failed",
            )
        return result.data

    @app.get("/api/publish/status")
    def get_publish_status(_: None = Depends(auth)):
        result = ctrl.publication.handle("publish.status")
        if not result.ok:
            raise HTTPException(
                status_code=400,
                detail=result.error.message if result.error else "failed",
            )
        return result.data

    @app.post("/api/publish/validate")
    def post_publish_validate(body: PublishBody, _: None = Depends(auth)):
        result = ctrl.publication.handle("publish.validate", source_dir=body.source_dir)
        if not result.ok:
            raise HTTPException(
                status_code=400,
                detail=result.error.message if result.error else "failed",
            )
        return result.data

    @app.post("/api/publish")
    def post_publish(body: PublishBody, _: None = Depends(auth)):
        result = ctrl.publication.handle(
            "publish.site",
            source_dir=body.source_dir,
            message=body.message,
            confirm=body.confirm,
            expect_substring=body.expect_substring,
        )
        if not result.ok:
            raise HTTPException(
                status_code=400,
                detail=result.error.message if result.error else "failed",
            )
        if ctrl.publication.provider == "github":
            cfg.pending.github_pat = False
        return result.data

    @app.post("/api/benchmark/reset")
    def post_benchmark_reset(_: None = Depends(auth)):
        """Explicit trial reset for fdbench (--reset-cmd). Silence alone is not a reset."""
        from me3.benchmark.reset import reset_controller_for_trial

        return reset_controller_for_trial(ctrl)

    @app.get("/api/devices")
    def devices(_: None = Depends(auth)):
        return device_snapshot(cfg)

    @app.websocket("/ws/session")
    async def ws_session(websocket: WebSocket, token: str | None = Query(default=None)):
        required = secrets.me3_ui_token or ""
        if not _token_ok(token, required, cfg.ui.require_token):
            await websocket.close(code=4401)
            return
        await websocket.accept()
        clients.add(websocket)
        await websocket.send_json(_session_event(ctrl.snapshot()))
        try:
            while True:
                msg = await websocket.receive_json()
                mtype = msg.get("type")
                if mtype == "set_enabled":
                    snap = await asyncio.to_thread(ctrl.set_enabled, bool(msg.get("enabled")))
                    await websocket.send_json(_session_event(snap))
                elif mtype == "set_mic_muted":
                    snap = await asyncio.to_thread(
                        ctrl.set_mic_muted, bool(msg.get("muted", msg.get("mic_muted")))
                    )
                    await websocket.send_json(_session_event(snap))
                elif mtype == "probe":
                    try:
                        uid = await asyncio.to_thread(
                            ctrl.speak_probe, str(msg.get("text") or "Session is on.")
                        )
                        await websocket.send_json({"type": "probe_started", "utterance_id": uid})
                    except RuntimeError as exc:
                        await websocket.send_json({"type": "error", "message": str(exc)})
                elif mtype == "ping":
                    await websocket.send_json({"type": "pong"})
                elif mtype == "get_session":
                    await websocket.send_json(_session_event(ctrl.snapshot()))
        except WebSocketDisconnect:
            pass
        finally:
            clients.discard(websocket)

    if UI_DIST.is_dir():
        app.mount("/assets", StaticFiles(directory=UI_DIST / "assets"), name="assets")

        @app.get("/")
        def index():
            # Always revalidate shell HTML so hashed asset refs stay current.
            return FileResponse(
                UI_DIST / "index.html",
                headers={"Cache-Control": "no-cache"},
            )

    return app


def run() -> None:
    import uvicorn

    cfg = load_config()
    secrets = load_secrets()
    if cfg.ui.require_token and not secrets.me3_ui_token:
        raise SystemExit(
            "ME3_UI_TOKEN is required (set in .env). Example: ME3_UI_TOKEN=dev-token"
        )
    app = create_app()
    uvicorn.run(app, host=cfg.ui.host, port=cfg.ui.port, log_level="info")
