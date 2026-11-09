"""Laptop / M1A audio device layer (sounddevice).

Capture callback only copies into a bounded queue — never runs ASR/TTS on the
audio thread. Resample to 16 kHz mono for ASR/VAD at the consumer boundary.
Playback resamples Piper (typically 22050) to the device output rate.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from me3.config import AppConfig, AudioConfig, load_config
from me3.speech.resample import resample_linear, to_mono

logger = logging.getLogger("me3.speech.device")


@dataclass
class DeviceInfo:
    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float


def list_devices() -> list[DeviceInfo]:
    import sounddevice as sd

    out: list[DeviceInfo] = []
    for i, d in enumerate(sd.query_devices()):
        out.append(
            DeviceInfo(
                index=i,
                name=str(d["name"]),
                max_input_channels=int(d["max_input_channels"]),
                max_output_channels=int(d["max_output_channels"]),
                default_samplerate=float(d["default_samplerate"]),
            )
        )
    return out


def resolve_device_index(
    explicit: int | None,
    hint: str,
    *,
    want_input: bool,
) -> int | None:
    """Pick device index: explicit wins; else first name match for hint; else None (default)."""
    if explicit is not None:
        return int(explicit)
    hint = (hint or "").strip().lower()
    if not hint:
        return None
    for d in list_devices():
        if hint not in d.name.lower():
            continue
        if want_input and d.max_input_channels > 0:
            return d.index
        if not want_input and d.max_output_channels > 0:
            return d.index
    logger.warning("device hint %r not found; using system default", hint)
    return None


def _probe_rate(device: int | None, channels: int, prefer: int, kind: str) -> int:
    import sounddevice as sd

    candidates = [prefer, 48000, 44100, 16000, 22050]
    # unique preserve order
    seen: set[int] = set()
    ordered: list[int] = []
    for r in candidates:
        if r not in seen:
            seen.add(r)
            ordered.append(r)
    for rate in ordered:
        try:
            if kind == "input":
                sd.check_input_settings(device=device, channels=channels, samplerate=rate, dtype="float32")
            else:
                sd.check_output_settings(device=device, channels=channels, samplerate=rate, dtype="float32")
            return rate
        except Exception:
            continue
    # fall back to device default
    info = sd.query_devices(device)
    return int(info["default_samplerate"])


class AudioCapture:
    """Non-blocking capture into a queue; consumer resamples to target_rate (16 kHz)."""

    def __init__(self, cfg: AppConfig | None = None):
        self.cfg = cfg or load_config()
        audio = self.cfg.audio
        self.device = resolve_device_index(audio.input_device, audio.input_device_hint, want_input=True)
        self.channels = max(1, audio.input_channels)
        self.blocksize = audio.capture_blocksize
        self.queue_max = audio.capture_queue_max
        self.target_rate = audio.input_sample_rate  # 16 kHz for ASR/VAD
        self.native_rate = _probe_rate(self.device, self.channels, 48000, "input")
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=self.queue_max)
        self._stream = None
        self.dropped = 0
        self.frames = 0

    def _callback(self, indata, frames, time_info, status):  # noqa: ARG002
        if status:
            logger.debug("capture status: %s", status)
        try:
            self._q.put_nowait(indata.copy())
            self.frames += 1
        except queue.Full:
            self.dropped += 1
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(indata.copy())
            except queue.Full:
                pass

    def start(self) -> None:
        import sounddevice as sd

        if self._stream is not None:
            return
        self._stream = sd.InputStream(
            samplerate=self.native_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=self.blocksize,
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()
        logger.info(
            "capture started device=%s native_rate=%s target_rate=%s",
            self.device,
            self.native_rate,
            self.target_rate,
        )

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def read_target_chunk(self, timeout: float = 0.2) -> np.ndarray | None:
        """Pop one native block, convert to mono float32 at target_rate."""
        try:
            block = self._q.get(timeout=timeout)
        except queue.Empty:
            return None
        mono = to_mono(block)
        return resample_linear(mono, self.native_rate, self.target_rate)

    def drain(self) -> None:
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break


class AudioPlayback:
    """Queue-based playback; accepts float32 mono at any rate and resamples to device."""

    def __init__(self, cfg: AppConfig | None = None):
        self.cfg = cfg or load_config()
        audio = self.cfg.audio
        self.device = resolve_device_index(audio.output_device, audio.output_device_hint, want_input=False)
        self.channels = max(1, audio.output_channels)
        self.blocksize = audio.playback_blocksize
        self.native_rate = _probe_rate(self.device, self.channels, 48000, "output")
        self._buf = np.zeros(0, dtype=np.float32)
        self._lock = threading.Lock()
        self._stream = None
        self._generation = 0  # bump on flush to invalidate in-flight audio
        self.underruns = 0

    def _callback(self, outdata, frames, time_info, status):  # noqa: ARG002
        if status:
            logger.debug("playback status: %s", status)
        with self._lock:
            need = frames * self.channels
            if len(self._buf) < frames:
                self.underruns += 1
                chunk = np.zeros(frames, dtype=np.float32)
                if len(self._buf):
                    chunk[: len(self._buf)] = self._buf
                    self._buf = np.zeros(0, dtype=np.float32)
            else:
                chunk = self._buf[:frames]
                self._buf = self._buf[frames:]
        if self.channels == 1:
            outdata[:, 0] = chunk
        else:
            for c in range(self.channels):
                outdata[:, c] = chunk

    def start(self) -> None:
        import sounddevice as sd

        if self._stream is not None:
            return
        self._stream = sd.OutputStream(
            samplerate=self.native_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=self.blocksize,
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()
        logger.info("playback started device=%s native_rate=%s", self.device, self.native_rate)

    def stop(self) -> None:
        self.flush()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def flush(self) -> int:
        """Drop all queued playback (OFF / cancel). Returns new generation id."""
        with self._lock:
            self._buf = np.zeros(0, dtype=np.float32)
            self._generation += 1
            return self._generation

    def play(self, samples: np.ndarray, sample_rate: int, generation: int | None = None) -> bool:
        """Enqueue mono float32 audio. Returns False if generation was invalidated."""
        mono = to_mono(samples)
        converted = resample_linear(mono, sample_rate, self.native_rate)
        with self._lock:
            if generation is not None and generation != self._generation:
                return False
            self._buf = np.concatenate([self._buf, converted])
            return True

    def queue_seconds(self) -> float:
        with self._lock:
            return len(self._buf) / float(self.native_rate)

    def wait_drain(self, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.queue_seconds() < 0.05:
                return True
            time.sleep(0.05)
        return False


def device_snapshot(cfg: AppConfig | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    audio: AudioConfig = cfg.audio
    return {
        "devices": [d.__dict__ for d in list_devices()],
        "resolved_input": resolve_device_index(audio.input_device, audio.input_device_hint, want_input=True),
        "resolved_output": resolve_device_index(audio.output_device, audio.output_device_hint, want_input=False),
        "hints": {
            "input": audio.input_device_hint,
            "output": audio.output_device_hint,
        },
    }
