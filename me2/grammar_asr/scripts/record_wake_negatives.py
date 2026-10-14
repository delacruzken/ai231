#!/usr/bin/env python3
"""Record (or synthesize) Mac ambient hard-negatives for wake training.

Writes 1.5 s mono 16 kHz WAVs into me2/data/wake/kaki/negative/ as
``mac_ambient_{kind}_{NNN}.wav``.

Examples (from me2/, mic allowed in Terminal):

  python -m grammar_asr.scripts.record_wake_negatives --kind silence --seconds 90
  python -m grammar_asr.scripts.record_wake_negatives --kind typing --seconds 60
  python -m grammar_asr.scripts.record_wake_negatives --kind speech --seconds 60

Offline bootstrap (no mic) — fan/silence/clicks at MacBook-like levels:

  python -m grammar_asr.scripts.record_wake_negatives --synthesize --n-per-kind 40
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from grammar_asr.data.features import SAMPLE_RATE

ME2_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ME2_ROOT / "data" / "wake" / "kaki" / "negative"
CHUNK_S = 1.5
KINDS = ("silence", "typing", "noise", "speech")


def _next_index(out_dir: Path, kind: str) -> int:
    prefix = f"mac_ambient_{kind}_"
    nums = []
    for p in out_dir.glob(f"{prefix}*.wav"):
        stem = p.stem[len(prefix):]
        if stem.isdigit():
            nums.append(int(stem))
    return (max(nums) + 1) if nums else 1


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x.astype(np.float64))) + 1e-12))


def _write_chunk(out_dir: Path, kind: str, idx: int, audio: np.ndarray) -> Path:
    path = out_dir / f"mac_ambient_{kind}_{idx:03d}.wav"
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    n = int(CHUNK_S * SAMPLE_RATE)
    if len(audio) < n:
        audio = np.pad(audio, (0, n - len(audio)))
    else:
        audio = audio[:n]
    sf.write(str(path), audio, SAMPLE_RATE, subtype="PCM_16")
    return path


def record_kind(out_dir: Path, kind: str, seconds: float, device=None) -> int:
    import sounddevice as sd

    out_dir.mkdir(parents=True, exist_ok=True)
    n_chunks = max(1, int(round(seconds / CHUNK_S)))
    idx = _next_index(out_dir, kind)
    print(
        f"Recording {n_chunks} × {CHUNK_S}s as kind={kind!r} "
        f"(start index {idx:03d}). Ctrl+C to abort.",
        flush=True,
    )
    time.sleep(1.0)
    written = 0
    for i in range(n_chunks):
        audio = sd.rec(
            int(CHUNK_S * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=device,
        )
        sd.wait()
        mono = audio.reshape(-1)
        path = _write_chunk(out_dir, kind, idx + i, mono)
        print(f"  wrote {path.name}  rms={_rms(mono):.4f}", flush=True)
        written += 1
    print(f"Done: {written} clips → {out_dir}", flush=True)
    return written


def synthesize_kind(out_dir: Path, kind: str, n: int, rng: np.random.Generator) -> int:
    """MacBook-like ambient levels from live debug (rms ~0.02–0.08)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = _next_index(out_dir, kind)
    n_samp = int(CHUNK_S * SAMPLE_RATE)
    written = 0
    for i in range(n):
        t = np.arange(n_samp, dtype=np.float64) / SAMPLE_RATE
        if kind == "silence":
            # Laptop fan / HVAC: low pink-ish noise
            white = rng.normal(0, 1, n_samp)
            # crude pink via cumulative sum + high-pass
            pink = np.cumsum(white)
            pink = pink - pink.mean()
            pink /= np.max(np.abs(pink)) + 1e-9
            target_rms = float(rng.uniform(0.015, 0.035))
            audio = (pink * target_rms / (_rms(pink) + 1e-12)).astype(np.float32)
        elif kind == "typing":
            audio = rng.normal(0, 0.012, n_samp).astype(np.float32)
            n_clicks = int(rng.integers(3, 12))
            for _ in range(n_clicks):
                c0 = int(rng.integers(0, n_samp - 800))
                dur = int(rng.integers(80, 400))
                click = rng.normal(0, 1, dur).astype(np.float32)
                env = np.exp(-np.linspace(0, 8, dur)).astype(np.float32)
                amp = float(rng.uniform(0.08, 0.25))
                audio[c0:c0 + dur] += click * env * amp
        elif kind == "noise":
            # Broadband room noise + occasional bump
            audio = rng.normal(0, 0.025, n_samp).astype(np.float32)
            if rng.random() < 0.5:
                c0 = int(rng.integers(0, n_samp - 2000))
                bump = np.sin(2 * np.pi * float(rng.uniform(80, 200)) * t[:2000])
                audio[c0:c0 + 2000] += (0.04 * bump * np.hanning(2000)).astype(np.float32)
        else:  # speech — short non-kaki babble (modulated noise bands)
            audio = np.zeros(n_samp, dtype=np.float32)
            n_syll = int(rng.integers(2, 5))
            for _ in range(n_syll):
                c0 = int(rng.integers(0, max(1, n_samp - 4000)))
                dur = int(rng.integers(2000, 4000))
                dur = min(dur, n_samp - c0)
                f0 = float(rng.uniform(120, 280))
                carrier = np.sin(2 * np.pi * f0 * np.arange(dur) / SAMPLE_RATE)
                form = np.sin(2 * np.pi * float(rng.uniform(500, 1800)) * np.arange(dur) / SAMPLE_RATE)
                env = np.hanning(dur)
                amp = float(rng.uniform(0.04, 0.12))
                audio[c0:c0 + dur] += (amp * carrier * (0.5 + 0.5 * form) * env).astype(np.float32)
            audio += rng.normal(0, 0.008, n_samp).astype(np.float32)

        peak = float(np.max(np.abs(audio)) + 1e-9)
        if peak > 0.95:
            audio = (audio * (0.9 / peak)).astype(np.float32)
        path = _write_chunk(out_dir, kind, idx + i, audio)
        print(f"  synth {path.name}  rms={_rms(audio):.4f}", flush=True)
        written += 1
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--kind", choices=KINDS, default="silence")
    ap.add_argument("--seconds", type=float, default=90.0,
                    help="Recording duration (mic mode)")
    ap.add_argument("--device", default=None, help="sounddevice input device")
    ap.add_argument("--synthesize", action="store_true",
                    help="Generate Mac-like ambient WAVs without a mic")
    ap.add_argument("--n-per-kind", type=int, default=40,
                    help="Clips per kind when --synthesize")
    ap.add_argument("--seed", type=int, default=4)
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    if args.synthesize:
        rng = np.random.default_rng(args.seed)
        total = 0
        for kind in KINDS:
            print(f"=== synthesize {kind} × {args.n_per_kind} ===", flush=True)
            total += synthesize_kind(out_dir, kind, args.n_per_kind, rng)
        print(f"Synthesized {total} clips under {out_dir}", flush=True)
        return

    record_kind(out_dir, args.kind, args.seconds, device=args.device)


if __name__ == "__main__":
    main()
