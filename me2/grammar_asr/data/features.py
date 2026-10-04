"""Audio loading + 40-band log-mel features (training / eval)."""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import soundfile as sf
import torch

SAMPLE_RATE = 16000
N_MELS = 40
N_FFT = 512
HOP = 160
WIN = 512


def load_wav(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import torchaudio.transforms as T
        t = torch.from_numpy(audio).unsqueeze(0)
        audio = T.Resample(sr, SAMPLE_RATE)(t).squeeze(0).numpy()
    return audio.astype(np.float32)


def log_mel(audio: np.ndarray) -> np.ndarray:
    import torchaudio.transforms as T
    t = torch.from_numpy(audio).unsqueeze(0)
    mel = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP,
        win_length=WIN, n_mels=N_MELS,
    )
    spec = mel(t)
    logspec = torch.clamp(spec, min=1e-5).log()
    return logspec.squeeze(0).transpose(0, 1).numpy().astype(np.float32)


def normalize_feat(feat: np.ndarray) -> np.ndarray:
    mu = feat.mean(axis=0, keepdims=True)
    sd = feat.std(axis=0, keepdims=True) + 1e-8
    return ((feat - mu) / sd).astype(np.float32)


def wav_to_feat(path_or_audio: Union[str, Path, np.ndarray],
                max_seconds: float = 6.0) -> np.ndarray:
    if isinstance(path_or_audio, (str, Path)):
        audio = load_wav(Path(path_or_audio))
    else:
        audio = np.asarray(path_or_audio, dtype=np.float32)
    max_samples = int(max_seconds * SAMPLE_RATE)
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    elif len(audio) < max_samples:
        audio = np.pad(audio, (0, max_samples - len(audio)))
    return normalize_feat(log_mel(audio))
