"""Streaming Zipformer ASR via sherpa-onnx OnlineRecognizer."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from queue import Empty, Full, Queue
from typing import Callable
from uuid import uuid4

import numpy as np

from me3.config import AppConfig, load_config
from me3.speech.agc import apply_agc
from me3.speech.paths import asr_paths


@dataclass
class AsrEvent:
    type: str  # partial | final
    text: str
    segment_id: str
    t_mono: float
    request_id: str | None = None


@dataclass
class AsrStats:
    chunks_accepted: int = 0
    chunks_blocked_ms: float = 0.0
    queue_high_water: int = 0
    audio_seconds: float = 0.0
    process_seconds: float = 0.0
    partials: int = 0
    finals: int = 0

    @property
    def rtf(self) -> float | None:
        if self.audio_seconds <= 0:
            return None
        return self.process_seconds / self.audio_seconds


def _result_text(recognizer, stream) -> str:
    if hasattr(recognizer, "get_result_all"):
        result = recognizer.get_result_all(stream)
        text = getattr(result, "text", None)
        if text is None and isinstance(result, dict):
            text = result.get("text", "")
        return (text or "").strip()
    raw = recognizer.get_result(stream)
    return (raw or "").strip() if isinstance(raw, str) else str(raw).strip()


class StreamingAsrEngine:
    """Accepts float32 PCM (16 kHz mono) on a worker thread; emits partial/final events."""

    def __init__(
        self,
        cfg: AppConfig | None = None,
        on_event: Callable[[AsrEvent], None] | None = None,
        queue_max: int | None = None,
    ):
        import sherpa_onnx

        self.cfg = cfg or load_config()
        paths = asr_paths(self.cfg)
        self.on_event = on_event
        self.queue_max = queue_max or self.cfg.speech.asr_queue_max
        self._audio_q: Queue[np.ndarray | None] = Queue(maxsize=self.queue_max)
        self.stats = AsrStats()
        self._segment_id = str(uuid4())
        self._last_text = ""
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._agc_ema: float | None = None

        decoding = (self.cfg.speech.asr_decoding_method or "greedy_search").strip()
        max_paths = max(1, int(self.cfg.speech.asr_max_active_paths))
        model_type = (self.cfg.models.asr_model_type or "").strip()

        kwargs = dict(
            tokens=str(paths["tokens"]),
            encoder=str(paths["encoder"]),
            decoder=str(paths["decoder"]),
            joiner=str(paths["joiner"]),
            num_threads=max(1, self.cfg.models.asr_threads),
            sample_rate=16000,
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=float(self.cfg.speech.asr_endpoint_rule1_trailing_silence),
            rule2_min_trailing_silence=float(self.cfg.speech.asr_endpoint_rule2_trailing_silence),
            rule3_min_utterance_length=float(self.cfg.speech.asr_endpoint_rule3_min_utterance_length),
            decoding_method=decoding,
            max_active_paths=max_paths,
            provider="cpu",
        )
        # Kroko Zipformer2 requires empty/auto model_type; legacy Zipformer needs "zipformer".
        if model_type:
            kwargs["model_type"] = model_type

        self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(**kwargs)
        self._stream = self._recognizer.create_stream()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._worker, name="asr-worker", daemon=True)
        self._thread.start()

    def stop(self, drain: bool = True) -> None:
        if drain:
            # Always deliver end-of-stream sentinel even if the queue is saturated.
            while True:
                try:
                    self._audio_q.put(None, block=False)
                    break
                except Full:
                    try:
                        self._audio_q.get_nowait()
                    except Empty:
                        break
        else:
            self._stop.set()
        if self._thread:
            self._thread.join(timeout=30.0)
            self._thread = None
            self._stop.set()

    def reset_segment(self) -> str:
        # Must not race the ASR worker: sherpa-onnx streams are not thread-safe;
        # barge soft-pause used to segfault on Pi by create_stream() mid-decode.
        with self._lock:
            self._segment_id = str(uuid4())
            self._last_text = ""
            self._agc_ema = None
            self._stream = self._recognizer.create_stream()
            return self._segment_id

    def force_finalize(self, *, emit: bool = False) -> str:
        """Close the current ASR segment without waiting for sherpa endpoint.

        ME3 turn-taking commits on silence (~0.7–1.5s) before Zipformer endpoint
        rules (~1.8–2.8s). After commit we gate ASR, so ``is_endpoint`` never fires
        and ``asr_final`` marks stay empty (fdbench WER n=0). Call this on commit
        to clear the stream; pass ``emit=True`` only when the caller did not already
        publish the transcript.
        """
        with self._lock:
            try:
                while self._recognizer.is_ready(self._stream):
                    self._recognizer.decode_stream(self._stream)
            except Exception:
                pass
            text = (_result_text(self._recognizer, self._stream) or self._last_text or "").strip()
            try:
                self._recognizer.reset(self._stream)
            except Exception:
                self._stream = self._recognizer.create_stream()
            self._last_text = ""
            self._segment_id = str(uuid4())
        if emit and text:
            self._emit("final", text)
        return text

    def accept_pcm(self, samples: np.ndarray, block: bool = True) -> None:
        """Enqueue float32 mono PCM. Blocks when queue is full (backpressure)."""
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        if samples.ndim != 1:
            samples = samples.reshape(-1)
        if self.cfg.audio.agc_enabled:
            samples, self._agc_ema = apply_agc(
                samples,
                target_rms=float(self.cfg.audio.agc_target_rms),
                max_gain=float(self.cfg.audio.agc_max_gain),
                min_rms=float(self.cfg.audio.agc_min_rms),
                ema=self._agc_ema,
            )
        t0 = time.monotonic()
        self._audio_q.put(samples.copy(), block=block)
        waited = (time.monotonic() - t0) * 1000.0
        self.stats.chunks_blocked_ms += waited
        self.stats.chunks_accepted += 1
        qsize = self._audio_q.qsize()
        if qsize > self.stats.queue_high_water:
            self.stats.queue_high_water = qsize

    def _emit(self, kind: str, text: str) -> None:
        evt = AsrEvent(
            type=kind,
            text=text,
            segment_id=self._segment_id,
            t_mono=time.monotonic(),
        )
        if kind == "partial":
            self.stats.partials += 1
        else:
            self.stats.finals += 1
        if self.on_event:
            self.on_event(evt)

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._audio_q.get(timeout=0.1)
            except Empty:
                continue
            if item is None:
                self._finish_stream()
                break
            t0 = time.monotonic()
            # Hold the same lock as reset_segment/force_finalize so barge cannot
            # replace the stream while decode is in native code. Emit outside the
            # lock — on_event may commit → force_finalize and would deadlock.
            partial_text: str | None = None
            final_text: str | None = None
            with self._lock:
                self._stream.accept_waveform(16000, item)
                self.stats.audio_seconds += len(item) / 16000.0
                while self._recognizer.is_ready(self._stream):
                    self._recognizer.decode_stream(self._stream)
                text = _result_text(self._recognizer, self._stream)
                if text and text != self._last_text:
                    self._last_text = text
                    partial_text = text
                if self._recognizer.is_endpoint(self._stream):
                    final = _result_text(self._recognizer, self._stream)
                    if final:
                        final_text = final
                    self._recognizer.reset(self._stream)
                    self._last_text = ""
                    self._segment_id = str(uuid4())
            if partial_text is not None:
                self._emit("partial", partial_text)
            if final_text is not None:
                self._emit("final", final_text)
            self.stats.process_seconds += time.monotonic() - t0

    def _finish_stream(self) -> None:
        t0 = time.monotonic()
        tail = np.zeros(int(16000 * 0.5), dtype=np.float32)
        final_text: str | None = None
        with self._lock:
            self._stream.accept_waveform(16000, tail)
            self._stream.input_finished()
            while self._recognizer.is_ready(self._stream):
                self._recognizer.decode_stream(self._stream)
            final = _result_text(self._recognizer, self._stream)
            if final:
                final_text = final
        if final_text is not None:
            self._emit("final", final_text)
        self.stats.process_seconds += time.monotonic() - t0
