"""Silero VAD (ONNX, CPU) — in-process, not an MCP service.

Frame API mirrors fd-benchmark: 512 samples at 16 kHz with 64-sample context.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from me3.config import AppConfig, load_config
from me3.speech.paths import vad_path

WINDOW = 512
CONTEXT = 64
SR = 16000

_session = None
_session_path: Path | None = None


def _get_session(model_file: Path, num_threads: int = 1):
    global _session, _session_path
    import onnxruntime as ort

    if _session is not None and _session_path == model_file:
        return _session
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = max(1, num_threads)
    opts.log_severity_level = 3
    _session = ort.InferenceSession(
        str(model_file),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    _session_path = model_file
    return _session


class SileroVAD:
    """Frame-level speech probability. Feed 512-sample float32 chunks at 16 kHz."""

    def __init__(
        self,
        model_file: Path | None = None,
        cfg: AppConfig | None = None,
        threshold: float | None = None,
    ):
        cfg = cfg or load_config()
        self.model_file = model_file or vad_path(cfg)
        self.threshold = (
            float(threshold) if threshold is not None else float(cfg.speech.vad_threshold)
        )
        self.window = int(cfg.speech.vad_window) or WINDOW
        self.session = _get_session(self.model_file, cfg.models.vad_threads)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, CONTEXT), dtype=np.float32)

    def prob(self, chunk: np.ndarray) -> float:
        if chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32)
        if chunk.shape[0] != self.window:
            padded = np.zeros(self.window, dtype=np.float32)
            n = min(self.window, chunk.shape[0])
            padded[:n] = chunk[:n]
            chunk = padded
        x = np.concatenate(
            [self._context, chunk.reshape(1, -1)],
            axis=1,
        )
        out, state = self.session.run(
            None,
            {
                "input": x,
                "state": self._state,
                "sr": np.array(SR, dtype=np.int64),
            },
        )
        self._state = state
        self._context = x[:, -CONTEXT:]
        return float(out[0][0])

    def is_speech(self, chunk: np.ndarray) -> bool:
        return self.prob(chunk) >= self.threshold


def iter_speech_masks(
    audio: np.ndarray,
    vad: SileroVAD | None = None,
) -> list[bool]:
    """Return per-window speech flags for a mono float32 waveform at 16 kHz."""
    vad = vad or SileroVAD()
    vad.reset()
    flags: list[bool] = []
    n = int(np.ceil(len(audio) / WINDOW))
    for i in range(n):
        chunk = audio[i * WINDOW : (i + 1) * WINDOW]
        flags.append(vad.is_speech(chunk))
    return flags
