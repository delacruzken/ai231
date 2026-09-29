"""
Audio loading + log-mel feature extraction for the ME2 grammar-ASR pipeline.

WAVs are 16 kHz mono 16-bit. We compute 40-band log-mel spectrograms with a
25 ms window / 10 ms hop (standard for compact ASR). Features are normalized
per-utterance (mean/std) which helps a small model generalize across the 100
speakers.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "AI231_src" / "MEX2" / "OptionB"
MANIFEST = DATA_DIR / "manifest.csv"

# ---- feature params --------------------------------------------------------
SAMPLE_RATE = 16000
N_MELS = 40
N_FFT = 512          # ~32 ms
HOP = 160            # 10 ms
WIN = 512


def load_wav(path: Path) -> np.ndarray:
    """Load a WAV as float32 mono at 16 kHz."""
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        # simple linear resample (rare; dataset is 16k)
        import torchaudio.transforms as T
        t = torch.from_numpy(audio).unsqueeze(0)
        audio = T.Resample(sr, SAMPLE_RATE)(t).squeeze(0).numpy()
    return audio


def log_mel(audio: np.ndarray) -> np.ndarray:
    """Compute 40-band log-mel features, shape (T, 40)."""
    import torchaudio
    import torchaudio.transforms as T
    t = torch.from_numpy(audio).unsqueeze(0)  # (1, N)
    mfcc = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP,
        win_length=WIN, n_mels=N_MELS,
    )
    spec = mfcc(t)                 # (1, 40, T)
    logspec = torch.clamp(spec, min=1e-5).log()
    feat = logspec.squeeze(0).transpose(0, 1).numpy()  # (T, 40)
    return feat.astype(np.float32)


def normalize_feat(feat: np.ndarray) -> np.ndarray:
    """Per-utterance mean/std normalization over time."""
    mu = feat.mean(axis=0, keepdims=True)
    sd = feat.std(axis=0, keepdims=True) + 1e-8
    return ((feat - mu) / sd).astype(np.float32)


# ---- manifest --------------------------------------------------------------
@dataclass
class Sample:
    path: Path
    intent: str
    transcript: str
    speaker: str
    split: str
    condition: str   # clean | noisy
    duration: float


def load_manifest(split: Optional[str] = None,
                  condition: Optional[str] = None) -> List[Sample]:
    """Load manifest rows, optionally filtered by split and/or condition."""
    samples: List[Sample] = []
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if split and row["split"] != split:
                continue
            fname = row["path"].rsplit("/", 1)[-1]
            row_cond = "noisy" if "_noisy.wav" in fname else "clean"
            if condition and row_cond != condition:
                continue
            samples.append(Sample(
                path=DATA_DIR / row["path"],
                intent=row["intent"],
                transcript=row["transcript"],
                speaker=row["speaker"],
                split=row["split"],
                condition=row_cond,
                duration=float(row["duration_sec"]),
            ))
    return samples


# ---- dataset ---------------------------------------------------------------
class OptionBDataset(Dataset):
    """Streams WAV -> log-mel features on the fly (no precompute needed)."""

    def __init__(self, samples: List[Sample], max_seconds: float = 6.0):
        self.samples = samples
        self.max_samples = int(max_seconds * SAMPLE_RATE)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        audio = load_wav(s.path)
        # trim / pad to max length
        if len(audio) > self.max_samples:
            audio = audio[:self.max_samples]
        elif len(audio) < self.max_samples:
            audio = np.pad(audio, (0, self.max_samples - len(audio)))
        feat = normalize_feat(log_mel(audio))          # (T, 40)
        T = feat.shape[0]
        return {
            "feat": torch.from_numpy(feat),
            "intent": s.intent,
            "transcript": s.transcript,
            "speaker": s.speaker,
            "condition": s.condition,
            "T": T,
        }


def make_dataloaders(batch_size: int = 32, num_workers: int = 0,
                     max_seconds: float = 6.0, collate_fn=None):
    train = OptionBDataset(load_manifest(split="train"), max_seconds)
    val = OptionBDataset(load_manifest(split="val"), max_seconds)
    test = OptionBDataset(load_manifest(split="test"), max_seconds)
    dl = lambda ds: DataLoader(ds, batch_size=batch_size, shuffle=(ds is train),
                               num_workers=num_workers, drop_last=False,
                               collate_fn=collate_fn)
    return dl(train), dl(val), dl(test), train, val, test


if __name__ == "__main__":
    import time
    for split in ("train", "val", "test"):
        for cond in (None, "clean", "noisy"):
            n = len(load_manifest(split=split, condition=cond))
            print(f"{split:5s} {str(cond):6s}: {n}")
    # feature sanity check on one file
    s = load_manifest(split="train")[0]
    t0 = time.time()
    a = load_wav(s.path)
    f = normalize_feat(log_mel(a))
    print(f"\nSample: {s.path.name}")
    print(f"  audio {len(a)} samples ({len(a)/16000:.2f}s), feat {f.shape}, "
          f"range [{f.min():.2f},{f.max():.2f}], took {time.time()-t0:.3f}s")
