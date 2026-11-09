"""ASR service: loopback WebSocket PCM + JSON control; Zipformer streaming.

Protocol (binary = float32 LE mono @ 16 kHz; text = JSON):
  client -> {"type":"session.start","request_id":"..."}
  client -> binary PCM chunks
  server -> {"type":"partial"|"final","text":"...","segment_id":"...","t_mono":...}
  client -> {"type":"session.stop"}
  client -> {"type":"ping"}
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from typing import Any

import numpy as np

from me3.config import load_config
from me3.mcp.common import RequestMeta, ToolResult, unsupported
from me3.speech.asr_engine import AsrEvent, StreamingAsrEngine

TOOLS = ("session.start", "session.stop", "transcribe")
logger = logging.getLogger("me3.asr")


def handle(tool: str, meta: RequestMeta | None = None, **_kwargs) -> ToolResult:
    """MCP tool surface (illustrative); live path uses the WebSocket server."""
    meta = meta or RequestMeta()
    if tool not in TOOLS:
        return unsupported(meta, tool)
    return ToolResult(ok=True, request_id=meta.request_id, data={"note": f"asr.{tool} via websocket"})


async def _serve(host: str, port: int) -> None:
    import websockets
    from websockets.asyncio.server import serve

    cfg = load_config()
    engine = StreamingAsrEngine(cfg=cfg)
    engine.start()
    clients: set[Any] = set()
    loop = asyncio.get_running_loop()

    def on_event(evt: AsrEvent) -> None:
        payload = json.dumps(
            {
                "type": evt.type,
                "text": evt.text,
                "segment_id": evt.segment_id,
                "t_mono": evt.t_mono,
            }
        )
        for ws in list(clients):
            asyncio.run_coroutine_threadsafe(ws.send(payload), loop)

    engine.on_event = on_event

    async def handler(ws):
        clients.add(ws)
        logger.info("ASR client connected")
        try:
            await ws.send(
                json.dumps(
                    {
                        "type": "ready",
                        "sample_rate": 16000,
                        "format": "pcm_f32le",
                        "channels": 1,
                        "queue_max": engine.queue_max,
                    }
                )
            )
            async for message in ws:
                if isinstance(message, bytes):
                    samples = np.frombuffer(message, dtype=np.float32).copy()
                    await asyncio.to_thread(engine.accept_pcm, samples, True)
                    continue
                msg = json.loads(message)
                mtype = msg.get("type")
                if mtype == "session.start":
                    seg = engine.reset_segment()
                    await ws.send(
                        json.dumps(
                            {
                                "type": "session.started",
                                "request_id": msg.get("request_id"),
                                "segment_id": seg,
                                "t_mono": time.monotonic(),
                            }
                        )
                    )
                elif mtype == "session.stop":
                    await asyncio.to_thread(engine.stop, True)
                    engine.start()
                    await ws.send(
                        json.dumps(
                            {
                                "type": "session.stopped",
                                "request_id": msg.get("request_id"),
                                "stats": {
                                    "rtf": engine.stats.rtf,
                                    "audio_seconds": engine.stats.audio_seconds,
                                    "process_seconds": engine.stats.process_seconds,
                                    "queue_high_water": engine.stats.queue_high_water,
                                    "chunks_blocked_ms": engine.stats.chunks_blocked_ms,
                                    "partials": engine.stats.partials,
                                    "finals": engine.stats.finals,
                                },
                            }
                        )
                    )
                elif mtype == "ping":
                    await ws.send(json.dumps({"type": "pong", "t_mono": time.monotonic()}))
                else:
                    await ws.send(json.dumps({"type": "error", "message": f"unknown {mtype}"}))
        finally:
            clients.discard(ws)
            logger.info("ASR client disconnected")

    logger.info("ASR listening on ws://%s:%s", host, port)
    async with serve(handler, host, port, max_size=8 * 1024 * 1024):
        await asyncio.Future()


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = load_config()
    parser = argparse.ArgumentParser(description="ME3 ASR Zipformer WebSocket service")
    parser.add_argument("--host", default=cfg.speech.asr_host)
    parser.add_argument("--port", type=int, default=cfg.speech.asr_port)
    args = parser.parse_args(argv)
    try:
        asyncio.run(_serve(args.host, args.port))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
