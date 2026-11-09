"""Concurrent duplex harness: record while playing Piper TTS (Phase 2 gate)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from me3.config import ROOT, load_config, resolve_path
from me3.speech.device import AudioCapture, AudioPlayback, device_snapshot
from me3.speech.tts_engine import PiperTtsEngine

logger = logging.getLogger("me3.speech.device_check")


def run_duplex(
    seconds: float = 3.0,
    speak: str = "Device check. Recording while I speak.",
    out_dir: Path | None = None,
    cfg=None,
) -> dict:
    cfg = cfg or load_config()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = out_dir or (resolve_path(cfg.paths.artifacts_dir) / "device_phase2" / stamp)
    out_dir.mkdir(parents=True, exist_ok=True)

    capture = AudioCapture(cfg)
    playback = AudioPlayback(cfg)
    tts = PiperTtsEngine(cfg)

    capture.start()
    playback.start()
    gen = playback.flush()

    tts_samples, tts_rate = tts.synthesize_to_array(speak)
    recorded: list[np.ndarray] = []

    t0 = time.monotonic()
    playback.play(tts_samples, tts_rate, generation=gen)

    while time.monotonic() - t0 < seconds:
        chunk = capture.read_target_chunk(timeout=0.1)
        if chunk is not None and len(chunk):
            recorded.append(chunk)

    playback.wait_drain(timeout=10.0)
    # gather a little more after playback
    end = time.monotonic() + 0.3
    while time.monotonic() < end:
        chunk = capture.read_target_chunk(timeout=0.05)
        if chunk is not None and len(chunk):
            recorded.append(chunk)

    capture.stop()
    playback.stop()

    import soundfile as sf

    rec = np.concatenate(recorded) if recorded else np.zeros(0, dtype=np.float32)
    in_path = out_dir / "capture_16k.wav"
    out_path = out_dir / "tts_played.wav"
    if len(rec):
        sf.write(str(in_path), rec, 16000)
    sf.write(str(out_path), tts_samples, tts_rate)

    metrics = {
        "host": "laptop",
        "phase": 2,
        "seconds_requested": seconds,
        "capture_samples": int(len(rec)),
        "capture_seconds": float(len(rec) / 16000.0),
        "tts_seconds": float(len(tts_samples) / tts_rate) if tts_rate else 0.0,
        "capture_native_rate": capture.native_rate,
        "playback_native_rate": playback.native_rate,
        "capture_dropped": capture.dropped,
        "playback_underruns": playback.underruns,
        "devices": device_snapshot(cfg),
        "paths": {"capture": str(in_path) if len(rec) else None, "tts": str(out_path)},
        "gate": {
            "capture_nonempty": len(rec) > 1600,  # >0.1 s
            "tts_nonempty": len(tts_samples) > 1000,
            "duplex_ok": len(rec) > 1600 and len(tts_samples) > 1000,
        },
        "m1a_followup": "pending — set audio.*_device_hint to EMEET/M1A and re-run",
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    logger.info("duplex done gate=%s -> %s", metrics["gate"], out_dir)
    return metrics


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="ME3 Phase 2 duplex device check")
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--speak", default="Device check. Recording while I speak.")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args(argv)
    if args.list_devices:
        print(json.dumps(device_snapshot(), indent=2))
        return 0
    try:
        metrics = run_duplex(seconds=args.seconds, speak=args.speak, out_dir=args.out)
    except Exception as exc:
        logger.exception("device_check failed: %s", exc)
        print(json.dumps({"error": str(exc)}))
        return 1
    print(json.dumps({"gate": metrics["gate"], "paths": metrics["paths"]}, indent=2))
    return 0 if metrics["gate"]["duplex_ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
