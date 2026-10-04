#!/usr/bin/env python3
"""Create a tiny synthetic gold-shaped fixture for offline smoke tests."""
from __future__ import annotations

import csv
import math
import wave
from pathlib import Path

import numpy as np

from grammar_asr.schema import VARIATIONS

SR = 16000


def synth_wav(path: Path, seconds: float, seed: int):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    # Multi-tone + noise so mel features are non-degenerate
    freq = 120 + (seed % 20) * 17
    audio = 0.2 * np.sin(2 * math.pi * freq * t)
    audio += 0.05 * rng.standard_normal(len(t))
    audio = np.clip(audio, -1, 1)
    pcm = (audio * 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def write_split(root: Path, split: str, speakers: list[str], per_var: int = 1):
    split_dir = root / split
    audio_dir = split_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    i = 0
    for spk in speakers:
        for v in VARIATIONS:
            for k in range(per_var):
                fname = f"{spk}_{v.variation_id}_{k}.wav"
                synth_wav(audio_dir / fname, 1.2, seed=i + 1)
                rows.append({
                    "file": f"audio/{fname}",
                    "transcript": v.phrase,
                    "command": v.intent,
                    "variation": v.phrase,
                    "slot_value": v.slot_value,
                    "out_of_scope": 0,
                    "bucket": v.phrase,
                    "speaker_id": spk,
                    "source": "fixture",
                    "is_synthetic": 1,
                    "accent_group": "Synthetic",
                    "duration_s": 1.2,
                })
                i += 1
        # one OOS per speaker
        fname = f"{spk}_oos.wav"
        synth_wav(audio_dir / fname, 1.0, seed=10_000 + i)
        rows.append({
            "file": f"audio/{fname}",
            "transcript": "hello there",
            "command": "OUT_OF_SCOPE",
            "variation": "",
            "slot_value": "",
            "out_of_scope": 1,
            "bucket": "OUT_OF_SCOPE",
            "speaker_id": spk,
            "source": "fixture",
            "is_synthetic": 1,
            "accent_group": "Synthetic",
            "duration_s": 1.0,
        })
        i += 1
    fields = list(rows[0].keys())
    with open(split_dir / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main():
    root = Path(__file__).resolve().parents[2] / "data" / "gold"
    if root.exists():
        # only create if empty-ish
        pass
    n_train = write_split(root, "train", [f"tr{i}" for i in range(4)], per_var=1)
    n_test = write_split(root, "test", [f"te{i}" for i in range(2)], per_var=1)
    n_hold = write_split(root, "holdout", ["ho0"], per_var=1)
    # copy variations
    import shutil
    shutil.copy2(
        Path(__file__).resolve().parents[1] / "schema" / "variations.csv",
        root / "variations.csv",
    )
    print(f"Fixture at {root}: train={n_train} test={n_test} holdout={n_hold}")


if __name__ == "__main__":
    main()
