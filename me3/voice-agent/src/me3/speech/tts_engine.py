"""Piper en_US-lessac-medium via piper-tts (OHF/rhasspy voice weights)."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Iterator
from uuid import uuid4

import numpy as np

from me3.config import AppConfig, load_config
from me3.speech.paths import tts_paths


@dataclass
class TtsChunk:
    utterance_id: str
    sequence: int
    pcm: np.ndarray  # float32 mono at native sample rate
    sample_rate: int
    done: bool = False


@dataclass
class TtsStats:
    utterances: int = 0
    cancelled: int = 0
    audio_seconds: float = 0.0
    process_seconds: float = 0.0

    @property
    def rtf(self) -> float | None:
        if self.audio_seconds <= 0:
            return None
        return self.process_seconds / self.audio_seconds


class PiperTtsEngine:
    """Synthesize text to PCM; supports cancel of the active utterance."""

    def __init__(self, cfg: AppConfig | None = None):
        from piper import PiperVoice

        self.cfg = cfg or load_config()
        paths = tts_paths(self.cfg)
        self.sample_rate = int(paths["sample_rate_hz"])
        self.stats = TtsStats()
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self._active_id: str | None = None
        self._voice = PiperVoice.load(str(paths["model"]))
        self.sample_rate = int(self._voice.config.sample_rate)

    def cancel(self, utterance_id: str | None = None) -> bool:
        with self._lock:
            if utterance_id is None or utterance_id == self._active_id:
                self._cancel.set()
                self.stats.cancelled += 1
                return True
        return False

    def synthesize(
        self,
        text: str,
        utterance_id: str | None = None,
    ) -> Iterator[TtsChunk]:
        """Generate float32 PCM chunks at Piper native rate. Yields a final done chunk."""
        uid = utterance_id or str(uuid4())
        with self._lock:
            self._active_id = uid
            self._cancel.clear()
        self.stats.utterances += 1
        t0 = time.monotonic()
        seq = 0
        rate = self.sample_rate
        total = 0
        for audio_chunk in self._voice.synthesize(text):
            if self._cancel.is_set():
                break
            pcm = np.asarray(audio_chunk.audio_float_array, dtype=np.float32)
            rate = int(audio_chunk.sample_rate) or rate
            total += len(pcm)
            yield TtsChunk(uid, seq, pcm, rate, done=False)
            seq += 1
        process_s = time.monotonic() - t0
        self.stats.process_seconds += process_s
        self.stats.audio_seconds += total / float(rate) if rate else 0.0
        self.sample_rate = rate
        yield TtsChunk(uid, seq, np.zeros(0, dtype=np.float32), rate, done=True)

    def synthesize_to_array(self, text: str) -> tuple[np.ndarray, int]:
        chunks = list(self.synthesize(text))
        parts = [c.pcm for c in chunks if len(c.pcm)]
        rate = chunks[0].sample_rate if chunks else self.sample_rate
        if not parts:
            return np.zeros(0, dtype=np.float32), rate
        return np.concatenate(parts), rate
