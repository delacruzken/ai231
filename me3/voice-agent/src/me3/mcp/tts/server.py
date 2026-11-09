"""TTS service: loopback WebSocket; Piper en_US-lessac-medium.

Protocol (text = JSON; binary = float32 LE mono at Piper native rate):
  client -> {"type":"speech.start","utterance_id":"...","text":"..."}
  server -> binary PCM chunks + {"type":"speech.chunk_meta","utterance_id":"...","sequence":N,"sample_rate":R}
  server -> {"type":"speech.complete","utterance_id":"...","sample_rate":R}
  client -> {"type":"speech.cancel","utterance_id":"..."}
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from typing import Any

from me3.config import load_config
from me3.mcp.common import RequestMeta, ToolResult, unsupported
from me3.speech.tts_engine import PiperTtsEngine

TOOLS = ("speech.start", "speech.cancel")
logger = logging.getLogger("me3.tts")


def handle(tool: str, meta: RequestMeta | None = None, **_kwargs) -> ToolResult:
    meta = meta or RequestMeta()
    if tool not in TOOLS:
        return unsupported(meta, tool)
    return ToolResult(ok=True, request_id=meta.request_id, data={"note": f"tts.{tool} via websocket"})


async def _serve(host: str, port: int) -> None:
    from websockets.asyncio.server import serve

    cfg = load_config()
    engine = PiperTtsEngine(cfg=cfg)

    async def handler(ws):
        logger.info("TTS client connected")
        try:
            await ws.send(
                json.dumps(
                    {
                        "type": "ready",
                        "sample_rate": engine.sample_rate,
                        "format": "pcm_f32le",
                        "channels": 1,
                        "voice": cfg.models.tts,
                    }
                )
            )
            async for message in ws:
                if isinstance(message, bytes):
                    continue
                msg = json.loads(message)
                mtype = msg.get("type")
                if mtype == "speech.start":
                    text = msg.get("text") or ""
                    uid = msg.get("utterance_id") or msg.get("request_id")
                    await ws.send(
                        json.dumps(
                            {
                                "type": "speech.started",
                                "utterance_id": uid,
                                "t_mono": time.monotonic(),
                            }
                        )
                    )

                    def _gen():
                        return list(engine.synthesize(text, utterance_id=uid))

                    chunks = await asyncio.to_thread(_gen)
                    for chunk in chunks:
                        if chunk.done:
                            await ws.send(
                                json.dumps(
                                    {
                                        "type": "speech.complete",
                                        "utterance_id": chunk.utterance_id,
                                        "sample_rate": chunk.sample_rate,
                                        "sequence": chunk.sequence,
                                        "t_mono": time.monotonic(),
                                        "stats": {
                                            "rtf": engine.stats.rtf,
                                            "audio_seconds": engine.stats.audio_seconds,
                                            "process_seconds": engine.stats.process_seconds,
                                            "cancelled": engine.stats.cancelled,
                                        },
                                    }
                                )
                            )
                        else:
                            await ws.send(
                                json.dumps(
                                    {
                                        "type": "speech.chunk_meta",
                                        "utterance_id": chunk.utterance_id,
                                        "sequence": chunk.sequence,
                                        "sample_rate": chunk.sample_rate,
                                        "samples": int(len(chunk.pcm)),
                                    }
                                )
                            )
                            await ws.send(chunk.pcm.tobytes())
                elif mtype == "speech.cancel":
                    ok = engine.cancel(msg.get("utterance_id"))
                    await ws.send(
                        json.dumps(
                            {
                                "type": "speech.cancelled",
                                "utterance_id": msg.get("utterance_id"),
                                "ok": ok,
                                "t_mono": time.monotonic(),
                            }
                        )
                    )
                elif mtype == "ping":
                    await ws.send(json.dumps({"type": "pong", "t_mono": time.monotonic()}))
                else:
                    await ws.send(json.dumps({"type": "error", "message": f"unknown {mtype}"}))
        finally:
            logger.info("TTS client disconnected")

    logger.info("TTS listening on ws://%s:%s", host, port)
    async with serve(handler, host, port, max_size=16 * 1024 * 1024):
        await asyncio.Future()


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = load_config()
    parser = argparse.ArgumentParser(description="ME3 TTS Piper WebSocket service")
    parser.add_argument("--host", default=cfg.speech.tts_host)
    parser.add_argument("--port", type=int, default=cfg.speech.tts_port)
    args = parser.parse_args(argv)
    try:
        asyncio.run(_serve(args.host, args.port))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
