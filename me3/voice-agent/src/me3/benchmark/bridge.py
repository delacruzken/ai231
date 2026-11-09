"""Pipecat transport ↔ ME3 SessionController bridge for fdbench direct feed."""

from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

from me3.benchmark.marks import describe_models, me3_model_card
from me3.benchmark.reset import reset_controller_for_trial
from me3.config import load_config
from me3.session.controller import SessionController

logger = logging.getLogger("me3.benchmark.bridge")


class Me3AudioBridge:
    """Holds the ME3 controller and pushes/pulls PCM for the Pipecat transport."""

    def __init__(self, log_dir: Path | None = None):
        self.cfg = load_config()
        self.log_dir = log_dir
        self.ctrl = SessionController(cfg=self.cfg, benchmark=True)
        self._out_q: asyncio.Queue[bytes | None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self.ctrl.start_backend()
        # Route TTS into the async out queue (int16 LE @ 16 kHz)
        assert self.ctrl.playback is not None
        self.ctrl.playback.set_callback(self._on_playback)
        self.ctrl.set_enabled(True)
        describe_models(me3_model_card())
        self._started = True
        logger.info("ME3 benchmark bridge started (session ON)")

    def stop(self) -> None:
        try:
            self.ctrl.set_enabled(False)
        except Exception:
            pass
        try:
            self.ctrl.shutdown()
        except Exception:
            logger.exception("controller shutdown failed")
        self._started = False

    def reset_trial(self) -> dict[str, Any]:
        return reset_controller_for_trial(self.ctrl)

    def bind_loop(self, loop: asyncio.AbstractEventLoop, out_q: asyncio.Queue[bytes | None]) -> None:
        self._loop = loop
        self._out_q = out_q

    def push_input_pcm(self, pcm: bytes, sample_rate: int = 16000) -> None:
        if not self._started or not self.ctrl.capture:
            return
        if sample_rate != 16000:
            # fdbench always sends 16 kHz; ignore other rates
            return
        self.ctrl.capture.push_pcm16(pcm)

    def _on_playback(self, samples: np.ndarray, rate: int) -> None:
        if self._out_q is None or self._loop is None:
            return
        pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        try:
            self._loop.call_soon_threadsafe(self._out_q.put_nowait, pcm)
        except Exception:
            logger.debug("audio out enqueue failed", exc_info=True)


async def build_me3_benchmark(transport: Any, log_dir: Path):
    """Build a Pipecat pipeline that only shuttles PCM to/from ME3."""
    from pipecat.frames.frames import (
        Frame,
        InputAudioRawFrame,
        InterruptionFrame,
        OutputAudioRawFrame,
        StartFrame,
    )
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import WorkerRunner
    from pipecat.pipeline.worker import PipelineParams, PipelineWorker
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

    bridge = Me3AudioBridge(log_dir=log_dir)
    out_q: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=64)

    class Me3Ingress(FrameProcessor):
        def __init__(self):
            super().__init__()
            self._started = False

        async def process_frame(self, frame: Frame, direction: FrameDirection):
            await super().process_frame(frame, direction)
            if isinstance(frame, StartFrame) and not self._started:
                self._started = True
                bridge.bind_loop(asyncio.get_running_loop(), out_q)
                await asyncio.to_thread(bridge.start)
            if isinstance(frame, InputAudioRawFrame):
                await asyncio.to_thread(bridge.push_input_pcm, frame.audio, frame.sample_rate)
            elif isinstance(frame, InterruptionFrame):
                await asyncio.to_thread(bridge.ctrl._cancel_speech)
            await self.push_frame(frame, direction)

    class Me3Egress(FrameProcessor):
        """Pull PCM from ME3 TTS and emit OutputAudioRawFrame toward the transport."""

        def __init__(self):
            super().__init__()
            self._task: asyncio.Task | None = None

        async def process_frame(self, frame: Frame, direction: FrameDirection):
            await super().process_frame(frame, direction)
            if isinstance(frame, StartFrame) and self._task is None:
                self._task = asyncio.create_task(self._pump())
            if isinstance(frame, InterruptionFrame):
                # Drain pending out audio
                while not out_q.empty():
                    try:
                        out_q.get_nowait()
                    except asyncio.QueueEmpty:
                        break
            await self.push_frame(frame, direction)

        async def _pump(self):
            while True:
                pcm = await out_q.get()
                if pcm is None:
                    break
                await self.push_frame(
                    OutputAudioRawFrame(audio=pcm, sample_rate=16000, num_channels=1)
                )

    ingress = Me3Ingress()
    egress = Me3Egress()
    pipeline = Pipeline([transport.input(), ingress, egress, transport.output()])
    # ME3 speech runs outside Pipecat (Zipformer/Piper), so BotSpeaking/UserSpeaking
    # frames never appear. Pipecat's default 300s idle timeout would cancel the
    # worker mid-benchmark; fdbench also launches with --once, so it never reconnects.
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(audio_in_sample_rate=16000, audio_out_sample_rate=16000),
        idle_timeout_secs=None,
        cancel_on_idle_timeout=False,
    )
    runner = WorkerRunner(handle_sigint=False)

    async def run():
        await runner.add_workers(worker)
        await runner.run()

    async def cancel():
        try:
            out_q.put_nowait(None)
        except Exception:
            pass
        await worker.cancel()
        await asyncio.to_thread(bridge.stop)

    # Expose bridge for optional reset_cmd / diagnostics
    cancel.bridge = bridge  # type: ignore[attr-defined]
    return run(), cancel
