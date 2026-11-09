"""Null capture + callback playback for fdbench direct feed (no mic/speaker)."""

from __future__ import annotations

import logging
import queue
import threading
from typing import Callable

import numpy as np

from me3.config import AppConfig, load_config
from me3.speech.resample import resample_linear

logger = logging.getLogger("me3.benchmark.audio_io")

AudioOutCallback = Callable[[np.ndarray, int], None]


class BenchCapture:
    """Queue-fed capture that looks like AudioCapture to SessionController."""

    def __init__(self, cfg: AppConfig | None = None):
        self.cfg = cfg or load_config()
        self.target_rate = 16000
        self.native_rate = 16000
        self.device = None
        self._q: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=256)
        self._running = False

    def start(self) -> None:
        self._running = True
        logger.info("benchmark capture started (queue-fed, no device)")

    def stop(self) -> None:
        self._running = False
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass

    def drain(self) -> None:
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

    def push_pcm16(self, pcm_i16: bytes) -> None:
        if not self._running:
            return
        arr = np.frombuffer(pcm_i16, dtype="<i2").astype(np.float32) / 32768.0
        self.push_float(arr)

    def push_float(self, samples: np.ndarray) -> None:
        if not self._running or samples.size == 0:
            return
        try:
            self._q.put_nowait(np.asarray(samples, dtype=np.float32).reshape(-1))
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(np.asarray(samples, dtype=np.float32).reshape(-1))
            except queue.Full:
                pass

    def read_target_chunk(self, timeout: float = 0.05) -> np.ndarray | None:
        """Pop one float32 mono block already at 16 kHz."""
        try:
            item = self._q.get(timeout=timeout)
        except queue.Empty:
            return None
        if item is None:
            return None
        return item


class BenchPlayback:
    """Playback that forwards PCM to an fdbench/transport callback instead of a speaker."""

    def __init__(self, cfg: AppConfig | None = None, on_audio: AudioOutCallback | None = None):
        self.cfg = cfg or load_config()
        self.native_rate = 16000
        self.device = None
        self.underruns = 0
        self._on_audio = on_audio
        self._gen = 0
        self._lock = threading.Lock()
        self._running = False

    def set_callback(self, on_audio: AudioOutCallback | None) -> None:
        self._on_audio = on_audio

    def start(self) -> None:
        self._running = True
        logger.info("benchmark playback started (callback sink)")

    def stop(self) -> None:
        self._running = False

    def flush(self) -> int:
        with self._lock:
            self._gen += 1
            return self._gen

    def play(self, samples: np.ndarray, rate: int, *, generation: int | None = None) -> None:
        if not self._running:
            return
        with self._lock:
            if generation is not None and generation != self._gen:
                return
            cb = self._on_audio
        if cb is None or samples.size == 0:
            return
        mono = np.asarray(samples, dtype=np.float32).reshape(-1)
        out_rate = 16000
        if rate != out_rate:
            mono = resample_linear(mono, rate, out_rate)
        try:
            cb(mono, out_rate)
        except Exception:
            logger.exception("benchmark audio-out callback failed")

    def wait_drain(self, timeout: float = 30.0) -> None:
        _ = timeout
        return
