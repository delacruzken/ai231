"""Recorded-input concurrency runner — Phase 1 exit-gate measurements.

Feeds a 16 kHz WAV through Silero VAD + Zipformer ASR while Piper TTS runs
concurrently in a sibling process (CPU contention). Writes artifacts + metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from me3.config import ROOT, load_config, resolve_path
from me3.speech.asr_engine import AsrEvent, StreamingAsrEngine
from me3.speech.paths import load_manifest
from me3.speech.tts_engine import PiperTtsEngine
from me3.speech.vad import WINDOW, SileroVAD

logger = logging.getLogger("me3.speech.runner")


def _load_wav_mono_16k(path: Path) -> np.ndarray:
    import soundfile as sf

    audio, sr = sf.read(str(path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != 16000:
        # linear resample at documented boundary
        n = int(len(audio) * 16000 / sr)
        x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n, endpoint=False)
        audio = np.interp(x_new, x_old, audio).astype(np.float32)
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > 1.0:
        audio = audio / peak
    return audio


def _process_rss_mb(pid: int) -> float | None:
    try:
        import psutil

        return psutil.Process(pid).memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def _cpu_percent(pid: int, interval: float = 0.2) -> float | None:
    try:
        import psutil

        p = psutil.Process(pid)
        p.cpu_percent(None)
        return p.cpu_percent(interval)
    except Exception:
        return None


def run_concurrency(
    wav_path: Path,
    speak_text: str,
    out_dir: Path,
    cfg=None,
) -> dict:
    cfg = cfg or load_config()
    out_dir.mkdir(parents=True, exist_ok=True)
    audio = _load_wav_mono_16k(wav_path)
    events: list[dict] = []

    def on_event(evt: AsrEvent) -> None:
        events.append(
            {
                "type": evt.type,
                "text": evt.text,
                "segment_id": evt.segment_id,
                "t_mono": evt.t_mono,
            }
        )
        logger.info("ASR %s: %s", evt.type, evt.text)

    asr = StreamingAsrEngine(cfg=cfg, on_event=on_event)
    tts = PiperTtsEngine(cfg=cfg)
    vad = SileroVAD(cfg=cfg)

    asr.start()
    t0 = time.monotonic()

    # Kick TTS on a worker while we stream ASR
    tts_result: dict = {}

    def tts_job():
        samples, rate = tts.synthesize_to_array(speak_text)
        tts_result["samples"] = samples
        tts_result["rate"] = rate
        tts_result["stats"] = {
            "rtf": tts.stats.rtf,
            "audio_seconds": tts.stats.audio_seconds,
            "process_seconds": tts.stats.process_seconds,
            "cancelled": tts.stats.cancelled,
        }

    import threading

    tts_thread = threading.Thread(target=tts_job, name="tts-concurrent", daemon=True)
    tts_thread.start()

    # Feed full WAV to ASR (feasibility measures recognizer RTF under TTS load).
    # VAD still runs for speech-ratio metrics; gating is Phase 2+/session policy.
    speech_frames = 0
    total_frames = 0
    for i in range(0, len(audio), WINDOW):
        chunk = audio[i : i + WINDOW]
        total_frames += 1
        if len(chunk) < WINDOW:
            pad = np.zeros(WINDOW, dtype=np.float32)
            pad[: len(chunk)] = chunk
            chunk = pad
        if vad.is_speech(chunk):
            speech_frames += 1
        asr.accept_pcm(chunk)

    asr.stop(drain=True)
    tts_thread.join(timeout=120)
    elapsed = time.monotonic() - t0

    # Save TTS wav
    tts_wav = out_dir / "tts_output.wav"
    if tts_result.get("samples") is not None and len(tts_result["samples"]):
        import soundfile as sf

        sf.write(str(tts_wav), tts_result["samples"], tts_result["rate"])

    # Save transcript
    finals = [e for e in events if e["type"] == "final"]
    partials = [e for e in events if e["type"] == "partial"]
    (out_dir / "transcript.json").write_text(
        json.dumps({"events": events, "finals": finals, "partials": partials}, indent=2)
        + "\n",
        encoding="utf-8",
    )

    try:
        import psutil

        proc = psutil.Process()
        cpu = proc.cpu_percent(0.3)
        rss = proc.memory_info().rss / (1024 * 1024)
        sys_cpu = psutil.cpu_percent(0.1)
    except Exception:
        cpu = rss = sys_cpu = None

    metrics = {
        "host": "laptop",
        "phase": 1,
        "wav": str(wav_path),
        "speak_text": speak_text,
        "wall_elapsed_s": elapsed,
        "input_audio_s": len(audio) / 16000.0,
        "vad_frames": total_frames,
        "vad_speech_frames": speech_frames,
        "vad_speech_ratio": (speech_frames / total_frames) if total_frames else 0.0,
        "asr": {
            "rtf": asr.stats.rtf,
            "audio_seconds": asr.stats.audio_seconds,
            "process_seconds": asr.stats.process_seconds,
            "queue_max": asr.queue_max,
            "queue_high_water": asr.stats.queue_high_water,
            "chunks_blocked_ms": asr.stats.chunks_blocked_ms,
            "partials": asr.stats.partials,
            "finals": asr.stats.finals,
            "final_texts": [f["text"] for f in finals],
        },
        "tts": tts_result.get("stats", {}),
        "tts_wav": str(tts_wav) if tts_wav.exists() else None,
        "resources": {
            "runner_cpu_percent": cpu,
            "runner_rss_mb": rss,
            "system_cpu_percent": sys_cpu,
        },
        "manifest": load_manifest(cfg),
        "config_hash": {
            "asr_threads": cfg.models.asr_threads,
            "tts_threads": cfg.models.tts_threads,
            "vad_threads": cfg.models.vad_threads,
            "asr_queue_max": cfg.speech.asr_queue_max,
        },
        "gate": {
            "asr_rtf_target": 0.5,
            "asr_rtf": asr.stats.rtf,
            "asr_kept_up": (asr.stats.rtf is not None and asr.stats.rtf <= 1.0),
            "queue_bounded": asr.stats.queue_high_water <= asr.queue_max,
            "tts_produced_audio": bool(tts_result.get("samples") is not None and len(tts_result.get("samples", []))),
        },
        "pi_remeasure": "pending — follow-up before claiming Pi readiness",
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "done RTF=%s queue_hw=%s tts_audio_s=%s -> %s",
        asr.stats.rtf,
        asr.stats.queue_high_water,
        tts.stats.audio_seconds,
        out_dir,
    )
    return metrics


def ensure_fixture_wav(
    path: Path,
    text: str = (
        "Please set a timer for five minutes. "
        "Also remind me to check the oven when it rings."
    ),
) -> Path:
    """Create a short 16 kHz fixture by synthesizing with Piper then resampling."""
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    import soundfile as sf

    tts = PiperTtsEngine()
    samples, rate = tts.synthesize_to_array(text)
    if rate != 16000:
        n = int(len(samples) * 16000 / rate)
        x_old = np.linspace(0.0, 1.0, num=len(samples), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n, endpoint=False)
        samples = np.interp(x_new, x_old, samples).astype(np.float32)
        rate = 16000
    sf.write(str(path), samples, rate)
    (path.with_suffix(".txt")).write_text(text + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = load_config()
    parser = argparse.ArgumentParser(description="ME3 Phase 1 speech concurrency runner")
    parser.add_argument("--wav", type=Path, default=None, help="16 kHz mono WAV (created if missing)")
    parser.add_argument(
        "--speak",
        default=(
            "I am synthesizing this sentence while the recognizer processes recorded speech. "
            "Concurrent load is the Phase one feasibility gate."
        ),
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    fixture = args.wav or (ROOT / "tests" / "fixtures" / "audio" / "timer_five_minutes.wav")
    ensure_fixture_wav(fixture)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out or (resolve_path(cfg.paths.artifacts_dir) / "speech_phase1" / stamp)
    metrics = run_concurrency(fixture, args.speak, out_dir, cfg=cfg)
    print(json.dumps({"out": str(out_dir), "gate": metrics["gate"], "asr_rtf": metrics["asr"]["rtf"]}, indent=2))
    ok = metrics["gate"]["asr_kept_up"] and metrics["gate"]["queue_bounded"] and metrics["gate"]["tts_produced_audio"]
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
