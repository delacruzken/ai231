"""Stepwise barge-in diagnostic for M1A / laptop.

Phases:
  1) Self-test — synthetic loud frames must trip the detector (no mic needed)
  2) Live probe — play TTS, measure quiet RMS, then wait for you to speak over it
  3) Verdict — whether interrupt fired and playback flushed

Usage:
  python -m me3.speech.barge_check
  python -m me3.speech.barge_check --seconds 12 --speak-after 3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from me3.config import load_config, resolve_path
from me3.speech.device import AudioCapture, AudioPlayback, device_snapshot
from me3.speech.tts_engine import PiperTtsEngine
from me3.speech.vad import WINDOW, SileroVAD

logger = logging.getLogger("me3.speech.barge_check")


@dataclass
class FrameHit:
    t_ms: float
    rms: float
    ema: float
    energy_hit: bool
    vad_prob: float
    vad_hit: bool
    held_ms: float


@dataclass
class BargeReport:
    phase: str
    ok: bool
    detail: str
    metrics: dict = field(default_factory=dict)


def _rms(frame: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(frame.astype(np.float32))) + 1e-12))


class BargeDetector:
    """Mirrors SessionController._watch_barge_in scoring (without session side effects)."""

    def __init__(self, cfg, vad: SileroVAD | None):
        self.cfg = cfg
        self.vad = vad
        self.buf = np.zeros(0, dtype=np.float32)
        self.held_ms = 0.0
        self.ema = 0.0
        self.ema_ready = False
        self.hits: list[FrameHit] = []

        s = cfg.session
        self.thr = float(getattr(s, "barge_in_vad_threshold", 0.32) or 0.32)
        self.hold_ms = int(getattr(s, "barge_in_hold_ms", 80) or 80)
        self.ratio = float(getattr(s, "barge_in_energy_ratio", 1.55) or 1.55)
        self.floor = float(getattr(s, "barge_in_energy_floor", 0.005) or 0.005)

    def reset(self) -> None:
        self.buf = np.zeros(0, dtype=np.float32)
        self.held_ms = 0.0
        self.ema = 0.0
        self.ema_ready = False
        self.hits.clear()

    def feed(self, chunk: np.ndarray, *, t0: float) -> bool:
        """Return True once held speech exceeds hold_ms."""
        self.buf = np.concatenate([self.buf, chunk.astype(np.float32, copy=False)])
        fired = False
        while len(self.buf) >= WINDOW:
            frame = self.buf[:WINDOW]
            self.buf = self.buf[WINDOW:]
            frame_ms = (WINDOW / 16000.0) * 1000.0
            rms = _rms(frame)

            if not self.ema_ready:
                self.ema = min(rms, self.floor)
                self.ema_ready = True
            else:
                baseline = max(self.ema, 1e-6)
                if rms < max(baseline * 1.25, self.floor * 0.8):
                    self.ema = 0.90 * self.ema + 0.10 * rms

            aec_like = self.ema < max(self.floor * 0.8, 0.004)
            if aec_like:
                energy_hit = rms >= max(self.floor * 0.5, 0.002)
            else:
                energy_hit = rms >= max(self.floor, self.ema * self.ratio)

            vad_prob = 0.0
            vad_hit = False
            if self.vad is not None:
                try:
                    peak = float(np.max(np.abs(frame))) if frame.size else 0.0
                    vad_frame = frame
                    if 1e-5 < peak < 0.08:
                        vad_frame = np.clip(frame * (0.12 / peak), -1.0, 1.0).astype(np.float32)
                    vad_prob = float(self.vad.prob(vad_frame))
                    vad_hit = vad_prob >= self.thr
                except Exception:
                    vad_hit = False

            if energy_hit or vad_hit:
                self.held_ms += frame_ms
            else:
                self.held_ms = max(0.0, self.held_ms - frame_ms * 0.6)

            self.hits.append(
                FrameHit(
                    t_ms=(time.monotonic() - t0) * 1000.0,
                    rms=rms,
                    ema=self.ema,
                    energy_hit=energy_hit,
                    vad_prob=vad_prob,
                    vad_hit=vad_hit,
                    held_ms=self.held_ms,
                )
            )
            if self.held_ms >= self.hold_ms:
                fired = True
                break
        return fired


def phase_self_test(cfg) -> BargeReport:
    """Prove detector logic trips on synthetic speech without hardware."""
    det = BargeDetector(cfg, vad=None)
    quiet = (np.random.randn(WINDOW).astype(np.float32) * 0.0005).astype(np.float32)
    loud = (np.random.randn(WINDOW).astype(np.float32) * 0.04).astype(np.float32)
    t0 = time.monotonic()
    det.feed(quiet, t0=t0)
    fired = False
    for _ in range(6):
        if det.feed(loud, t0=t0):
            fired = True
            break
    return BargeReport(
        phase="1_self_test",
        ok=fired,
        detail="synthetic loud frames tripped detector" if fired else "detector did not trip on synthetic audio",
        metrics={
            "hold_ms": det.hold_ms,
            "floor": det.floor,
            "ratio": det.ratio,
            "frames": len(det.hits),
            "last": asdict(det.hits[-1]) if det.hits else None,
        },
    )


def phase_live(cfg, *, seconds: float, speak_after: float, speak: str) -> BargeReport:
    """Play TTS and wait for a real barge-in from the mic."""
    capture = AudioCapture(cfg)
    playback = AudioPlayback(cfg)
    tts = PiperTtsEngine(cfg)
    try:
        vad = SileroVAD(cfg=cfg)
    except Exception as exc:
        logger.warning("VAD unavailable: %s — energy-only barge", exc)
        vad = None

    det = BargeDetector(cfg, vad)
    capture.start()
    playback.start()
    gen = playback.flush()

    samples, rate = tts.synthesize_to_array(speak)
    # Stretch by looping if needed so there is time to barge.
    need = max(seconds + 1.0, speak_after + 4.0)
    audio = samples
    while len(audio) / float(rate) < need:
        audio = np.concatenate([audio, samples])

    print(
        f"\n>>> Playing TTS on device {playback.device}. "
        f"Stay quiet for {speak_after:.0f}s (baseline), then SPEAK OVER the voice.\n"
    )
    t0 = time.monotonic()
    playback.play(audio, rate, generation=gen)

    quiet_rms: list[float] = []
    speech_rms: list[float] = []
    fired_at: float | None = None
    stopped = False

    while time.monotonic() - t0 < seconds:
        chunk = capture.read_target_chunk(timeout=0.05)
        if chunk is None or not len(chunk):
            if playback.queue_seconds() < 0.04 and fired_at is None:
                # TTS ended before barge — refill once so user still has a chance.
                gen = playback.flush()
                playback.play(audio, rate, generation=gen)
            continue

        elapsed = time.monotonic() - t0
        rms = _rms(chunk)
        if elapsed < speak_after:
            quiet_rms.append(rms)
        else:
            speech_rms.append(rms)

        if fired_at is None and det.feed(chunk, t0=t0):
            fired_at = elapsed
            gen = playback.flush()
            stopped = playback.queue_seconds() < 0.08
            print(f"\n*** BARGE DETECTED at t={fired_at:.2f}s — flushed playback ***\n")
            break

        # Live meter every ~0.4s
        if det.hits and int(elapsed * 5) != int((elapsed - 0.05) * 5):
            h = det.hits[-1]
            print(
                f"t={elapsed:5.2f}s  rms={h.rms:.4f}  ema={h.ema:.4f}  "
                f"E={'Y' if h.energy_hit else 'n'}  V={h.vad_prob:.2f}/{'Y' if h.vad_hit else 'n'}  "
                f"held={h.held_ms:5.0f}ms",
                flush=True,
            )

    capture.stop()
    playback.stop()

    q = float(np.median(quiet_rms)) if quiet_rms else 0.0
    s = float(np.median(speech_rms)) if speech_rms else 0.0
    energy_hits = sum(1 for h in det.hits if h.energy_hit)
    vad_hits = sum(1 for h in det.hits if h.vad_hit)

    if fired_at is not None and stopped:
        ok, detail = True, f"barge fired at {fired_at:.2f}s and playback stopped"
    elif fired_at is not None:
        ok, detail = False, f"detector fired at {fired_at:.2f}s but playback still had audio"
    elif s < max(det.floor * 0.5, 0.002):
        ok, detail = (
            False,
            f"mic speech RMS too low (median {s:.4f} < floor path). "
            "Raise M1A input gain in macOS Sound settings, speak closer/louder.",
        )
    elif energy_hits == 0 and vad_hits == 0:
        ok, detail = (
            False,
            f"no energy/VAD hits. quiet_rms={q:.4f} speech_rms={s:.4f} "
            f"floor={det.floor} thr={det.thr}. Thresholds may still be too high.",
        )
    else:
        ok, detail = (
            False,
            f"hits seen (E={energy_hits}, V={vad_hits}) but held<{det.hold_ms}ms — "
            "speak more continuously, or lower barge_in_hold_ms.",
        )

    return BargeReport(
        phase="2_live_probe",
        ok=ok,
        detail=detail,
        metrics={
            "devices": device_snapshot(cfg),
            "quiet_rms_median": q,
            "speech_rms_median": s,
            "energy_hits": energy_hits,
            "vad_hits": vad_hits,
            "fired_at_s": fired_at,
            "playback_stopped": stopped,
            "thresholds": {
                "vad": det.thr,
                "hold_ms": det.hold_ms,
                "ratio": det.ratio,
                "floor": det.floor,
            },
            "hit_tail": [asdict(h) for h in det.hits[-12:]],
        },
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="ME3 barge-in diagnostic")
    parser.add_argument("--seconds", type=float, default=14.0, help="total live probe duration")
    parser.add_argument("--speak-after", type=float, default=3.0, help="quiet baseline seconds before you talk")
    parser.add_argument(
        "--speak",
        default=(
            "This is a barge in test. I will keep talking so you can interrupt me. "
            "Please wait for the countdown, then say stop stop stop clearly into the speakerphone."
        ),
    )
    parser.add_argument("--skip-live", action="store_true", help="only run synthetic self-test")
    args = parser.parse_args(argv)

    cfg = load_config()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = resolve_path(cfg.paths.artifacts_dir) / "barge_check" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    reports: list[BargeReport] = []

    print("=== Phase 1: synthetic self-test (no mic) ===")
    r1 = phase_self_test(cfg)
    reports.append(r1)
    print(f"[{'PASS' if r1.ok else 'FAIL'}] {r1.detail}")
    if not r1.ok:
        _write(out_dir, reports)
        return 2

    if args.skip_live:
        _write(out_dir, reports)
        return 0

    print("=== Phase 2: live M1A/laptop probe ===")
    print("Tips: session UI can stay closed; this opens the mic/speaker directly.")
    print(f"Devices: {json.dumps(device_snapshot(cfg)['hints'])}")
    try:
        r2 = phase_live(cfg, seconds=args.seconds, speak_after=args.speak_after, speak=args.speak)
    except Exception as exc:
        logger.exception("live probe failed")
        r2 = BargeReport(phase="2_live_probe", ok=False, detail=f"exception: {exc}")
    reports.append(r2)
    print(f"[{'PASS' if r2.ok else 'FAIL'}] {r2.detail}")

    path = _write(out_dir, reports)
    print(f"\nWrote {path}")
    print("\nNext:")
    if not r2.ok:
        print("  - If speech_rms is tiny: raise macOS Sound → Input gain for EMEET")
        print("  - If E/V hits but held short: speak continuously 'stop stop stop'")
        print("  - If Phase 1 pass + Phase 2 fail: hardware levels, not detector math")
    else:
        print("  - Detector+stop OK. If UI still fails, barge isn't reached (state/gate/pump).")
        print("  - Restart --serve and watch logs for: barge-in hard stop")
    return 0 if all(r.ok for r in reports) else 3


def _write(out_dir: Path, reports: list[BargeReport]) -> Path:
    path = out_dir / "report.json"
    path.write_text(
        json.dumps([asdict(r) for r in reports], indent=2) + "\n",
        encoding="utf-8",
    )
    return path


if __name__ == "__main__":
    sys.exit(main())
