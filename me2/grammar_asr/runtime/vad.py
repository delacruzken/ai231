"""Lightweight energy-based VAD / endpointing for Pi streaming."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import numpy as np


@dataclass
class VADConfig:
    sample_rate: int = 16000
    frame_ms: int = 30
    start_ms: int = 120
    end_ms: int = 600
    max_utterance_ms: int = 5000
    energy_threshold: float = 0.01
    start_ratio: float = 1.5


class EnergyVAD:
    def __init__(self, cfg: Optional[VADConfig] = None):
        self.cfg = cfg or VADConfig()
        self.frame = int(self.cfg.sample_rate * self.cfg.frame_ms / 1000)
        self.start_frames = max(1, self.cfg.start_ms // self.cfg.frame_ms)
        self.end_frames = max(1, self.cfg.end_ms // self.cfg.frame_ms)
        self.max_frames = max(1, self.cfg.max_utterance_ms // self.cfg.frame_ms)
        self.reset()

    def reset(self):
        self.speaking = False
        self.speech_run = 0
        self.silence_run = 0
        self.buf: Deque[np.ndarray] = deque()
        self.noise_floor = self.cfg.energy_threshold

    def _energy(self, frame: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(frame)) + 1e-12))

    def push(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Push one frame; return completed utterance audio or None."""
        if len(frame) != self.frame:
            # resample length by pad/trim
            out = np.zeros(self.frame, dtype=np.float32)
            n = min(self.frame, len(frame))
            out[:n] = frame[:n]
            frame = out
        e = self._energy(frame)
        thr = max(self.cfg.energy_threshold, self.noise_floor * self.cfg.start_ratio)
        if not self.speaking:
            if e < thr:
                self.noise_floor = 0.95 * self.noise_floor + 0.05 * e
                return None
            self.speech_run += 1
            self.buf.append(frame.copy())
            if self.speech_run >= self.start_frames:
                self.speaking = True
                self.silence_run = 0
            return None

        self.buf.append(frame.copy())
        if e < thr:
            self.silence_run += 1
        else:
            self.silence_run = 0
        if self.silence_run >= self.end_frames or len(self.buf) >= self.max_frames:
            audio = np.concatenate(list(self.buf))
            self.reset()
            return audio
        return None
