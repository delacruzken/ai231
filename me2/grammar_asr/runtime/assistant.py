"""Always-listening Pi assistant: wake -> command -> reject/dispatch -> sleep."""
from __future__ import annotations

import argparse
import json
import queue
import time
from pathlib import Path
from typing import Optional

import numpy as np

from ..data.features import SAMPLE_RATE, normalize_feat, log_mel
from ..decode.grammar_trie import build_grammar
from ..decode.semantic import RejectGate, decode_model_output, SemanticPrediction
from ..schema import OUT_OF_SCOPE
from .actions import MockDispatcher, HttpDemoDispatcher
from .logging_jsonl import BenchmarkLogger
from .vad import EnergyVAD, VADConfig
from .console import C, paint, tag, score_vs_thr, conf_vs_thr

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None


class OnnxSession:
    def __init__(self, path: Path):
        if ort is None:
            raise ImportError("onnxruntime is required for Pi inference")
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(
            str(path), opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.sess.get_inputs()[0].name
        outs = [o.name for o in self.sess.get_outputs()]
        self.output_names = outs

    def run(self, feat: np.ndarray) -> dict:
        x = feat[None].astype(np.float32)
        ys = self.sess.run(self.output_names, {self.input_name: x})
        # Map by name; support single-output CTC and multi-output heads.
        if len(self.output_names) == 1:
            return {"ctc_logits": ys[0]}
        out = {}
        for name, y in zip(self.output_names, ys):
            out[name] = y
        # Normalize common aliases
        if "logits" in out and "ctc_logits" not in out:
            out["ctc_logits"] = out["logits"]
        if "intent_logits" not in out:
            for k, v in list(out.items()):
                if "intent" in k:
                    out["intent_logits"] = v
        return out


class VoiceAssistant:
    STATE_SLEEP = "sleep"
    STATE_LISTEN = "listen"

    def __init__(
        self,
        command_model: Path,
        wake_model: Optional[Path] = None,
        reject_gate: Optional[Path] = None,
        wake_phrase_norm: str = "kaki",
        wake_threshold: float = 0.85,
        student_id: str = "me2",
        log_dir: Optional[str] = None,
        char_reward: float = 0.15,
        prefer: str = "auto",
        debug: bool = True,
        min_wake_rms: float = 0.040,
        min_wake_peak_rms: float = 0.11,
        min_wake_ms: float = 200.0,
        max_wake_ms: float = 1600.0,
        min_cmd_rms: float = 0.035,
        demo_ui: bool = False,
        demo_url: str = "http://127.0.0.1:8765",
    ):
        self.command = OnnxSession(command_model)
        self.wake = OnnxSession(wake_model) if wake_model else None
        self.gate = RejectGate.load(reject_gate) if reject_gate else RejectGate(
            min_confidence=0.35, min_margin=0.05
        )
        self.trie = build_grammar()
        # Sleep-time VAD is stricter so quiet room noise does not form utterances.
        self.vad = EnergyVAD(VADConfig(
            energy_threshold=0.025,
            start_ratio=2.5,
            start_ms=180,
            end_ms=450,
            max_utterance_ms=4500,
        ))
        self.state = self.STATE_SLEEP
        self.wake_phrase_norm = wake_phrase_norm
        self.wake_threshold = wake_threshold
        self.min_wake_rms = min_wake_rms
        self.min_wake_peak_rms = min_wake_peak_rms
        self.min_wake_ms = min_wake_ms
        self.max_wake_ms = max_wake_ms
        self.min_cmd_rms = min_cmd_rms
        self.listen_timeout_s = 4.0
        self._listen_started: float | None = None
        self.char_reward = char_reward
        self.prefer = prefer
        self.debug = debug
        self.logger = BenchmarkLogger(student_id=student_id, log_dir=log_dir)
        if demo_ui:
            self.actions = HttpDemoDispatcher(base_url=demo_url, verbose=True)
        else:
            self.actions = MockDispatcher(verbose=True)
        self.config = {
            "command_model": str(command_model),
            "wake_model": str(wake_model) if wake_model else None,
            "wake_threshold": wake_threshold,
            "min_wake_rms": min_wake_rms,
            "min_wake_peak_rms": min_wake_peak_rms,
            "min_wake_ms": min_wake_ms,
            "max_wake_ms": max_wake_ms,
            "min_cmd_rms": min_cmd_rms,
            "demo_ui": bool(demo_ui),
            "demo_url": demo_url if demo_ui else None,
            "listen_timeout_s": self.listen_timeout_s,
            "prefer": prefer,
            "debug": debug,
            "reject_gate": self.gate.to_dict(),
        }

    def _dbg(self, msg: str) -> None:
        if self.debug:
            print(msg, flush=True)

    def _audio_stats(self, audio: np.ndarray) -> tuple[float, float, float]:
        ms = len(audio) / SAMPLE_RATE * 1000.0
        x = audio.astype(np.float64)
        rms = float(np.sqrt(np.mean(np.square(x)) + 1e-12))
        # Peak over 30 ms frames — speech spikes; steady fan noise does not.
        frame = max(1, int(SAMPLE_RATE * 0.03))
        if len(x) < frame:
            peak = float(np.max(np.abs(x))) if len(x) else 0.0
        else:
            n = (len(x) // frame) * frame
            frames = x[:n].reshape(-1, frame)
            peak = float(np.max(np.sqrt(np.mean(np.square(frames), axis=1) + 1e-12)))
        return ms, rms, peak

    def _active_crop(self, audio: np.ndarray,
                     *, rel_thr: float = 0.35,
                     pad_ms: float = 80.0) -> np.ndarray:
        """Keep the loud contiguous region so trailing silence does not
        inflate wake duration (real 'kaki' was being rejected as >1.8 s)."""
        x = audio.astype(np.float32)
        frame = max(1, int(SAMPLE_RATE * 0.03))
        if len(x) < frame * 2:
            return x
        n = (len(x) // frame) * frame
        energies = np.sqrt(
            np.mean(np.square(x[:n].reshape(-1, frame)), axis=1) + 1e-12
        )
        thr = max(float(energies.max()) * rel_thr, self.min_wake_rms * 0.8)
        on = np.where(energies >= thr)[0]
        if len(on) == 0:
            return x
        pad = max(1, int(pad_ms / 30.0))
        i0 = max(0, int(on[0]) - pad)
        i1 = min(len(energies), int(on[-1]) + pad + 1)
        return x[i0 * frame: i1 * frame]

    def _feat(self, audio: np.ndarray) -> np.ndarray:
        return normalize_feat(log_mel(audio.astype(np.float32)))

    def _predict_command(self, audio: np.ndarray) -> tuple[SemanticPrediction, float, float]:
        t0 = time.perf_counter()
        feat = self._feat(audio)
        outputs = self.command.run(feat)
        # ORT returns numpy; wrap into decode_model_output format
        pred = decode_model_output(outputs, trie=self.trie,
                                   char_reward=self.char_reward,
                                   prefer=self.prefer)
        infer_ms = (time.perf_counter() - t0) * 1000
        audio_ms = len(audio) / SAMPLE_RATE * 1000
        return pred, infer_ms, audio_ms

    def _reject_reasons(self, pred: SemanticPrediction) -> list[str]:
        reasons: list[str] = []
        if pred.intent == OUT_OF_SCOPE:
            reasons.append("model-oos")
        thr = float(self.gate.intent_thresholds.get(
            pred.intent, self.gate.min_confidence
        ))
        if pred.confidence < thr:
            reasons.append(f"conf {pred.confidence:.3f}<{thr:.3f}")
        if pred.margin < self.gate.min_margin:
            reasons.append(f"margin {pred.margin:.3f}<{self.gate.min_margin:.3f}")
        if pred.score < self.gate.min_score:
            reasons.append(f"score {pred.score:.3f}<{self.gate.min_score:.3f}")
        return reasons

    def _wake_score(self, audio: np.ndarray) -> float:
        if self.wake is None:
            # Fallback: grammar/CTC decode looking for wake phrase keywords
            # using the command model scores is weak; use energy+keyword proxy.
            # Prefer providing a wake ONNX. Here we use a simple energy gate.
            return 1.0 if float(np.sqrt(np.mean(audio ** 2))) > 0.02 else 0.0
        feat = self._feat(audio)
        out = self.wake.run(feat)
        logits = out.get("wake_logits") or out.get("logits") or next(iter(out.values()))
        if logits.ndim == 2:
            logits = logits[0]
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        return float(probs[1] if len(probs) > 1 else probs[0])

    def handle_utterance(self, audio: np.ndarray) -> Optional[dict]:
        ms, rms, peak = self._audio_stats(audio)
        self._dbg(
            f"{tag('vad', 'vad')} state={self.state} audio={ms:.0f}ms "
            f"rms={rms:.4f} peak={peak:.4f} frames={len(audio)}"
        )

        if self.state == self.STATE_SLEEP:
            # Score/gate on the active crop so trailing silence does not
            # make a real "kaki" look like a 2 s utterance.
            crop = self._active_crop(audio)
            cms, crms, cpeak = self._audio_stats(crop)
            if (
                cms < self.min_wake_ms
                or cms > self.max_wake_ms
                or crms < self.min_wake_rms
                or cpeak < self.min_wake_peak_rms
            ):
                self._dbg(
                    f"{tag('wake', 'wake_skip')} skip energy "
                    f"buf={ms:.0f}ms active={cms:.0f}ms "
                    f"rms={crms:.4f} peak={cpeak:.4f} "
                    f"(need {self.min_wake_ms:.0f}–{self.max_wake_ms:.0f}ms, "
                    f"rms>={self.min_wake_rms:.3f}, "
                    f"peak>={self.min_wake_peak_rms:.3f})"
                )
                return None
            score = self._wake_score(crop)
            hit = score >= self.wake_threshold
            scored = score_vs_thr(score, self.wake_threshold)
            if hit:
                self.logger.wake(score)
                print(
                    f"{tag('wake', 'wake_hit')} HIT {scored} "
                    f"active={cms:.0f}ms rms={crms:.4f} peak={cpeak:.4f}",
                    flush=True,
                )
                self.state = self.STATE_LISTEN
                self._listen_started = time.perf_counter()
                print(
                    f"{tag('listen', 'listen')} say a command after "
                    f"'{self.wake_phrase_norm}'…",
                    flush=True,
                )
                # Alexa-like: beep + listening UI (browser handles chime/TTS).
                self.actions.notify("wake", phrase=self.wake_phrase_norm)
            else:
                self._dbg(
                    f"{tag('wake', 'wake_miss')} miss {scored} "
                    f"active={cms:.0f}ms rms={crms:.4f} peak={cpeak:.4f}"
                )
            return None

        # Listening for a command — ignore near-silence leftovers after wake.
        # OOS on residual noise is expected; do not even decode quiet clips.
        if self._listen_started is not None:
            waited = time.perf_counter() - self._listen_started
            if waited > self.listen_timeout_s:
                self._dbg(
                    f"{tag('sleep', 'listen_timeout')} no clear command in "
                    f"{waited:.1f}s — back to wake"
                )
                self.state = self.STATE_SLEEP
                self._listen_started = None
                self.actions.notify("listen_end", reason="timeout")
                self.vad.reset()
                return None
        if rms < self.min_cmd_rms or peak < self.min_wake_peak_rms:
            self._dbg(
                f"{tag('cmd', 'vad')} skip quiet leftover "
                f"audio={ms:.0f}ms rms={rms:.4f} peak={peak:.4f} "
                f"(still listening for a real command)"
            )
            return None
        raw, infer_ms, audio_ms = self._predict_command(audio)
        reasons = self._reject_reasons(raw)
        pred = self.gate.apply(raw)
        record = pred.to_log_dict(infer_ms, audio_ms)
        record["model"] = self.config["command_model"]
        record["raw_intent"] = raw.intent
        record["raw_variation"] = raw.variation
        self.logger.log(record)

        gate_thr = float(self.gate.intent_thresholds.get(
            raw.intent, self.gate.min_confidence
        ))
        conf_part = conf_vs_thr(raw.confidence, gate_thr)
        reject_why = ""
        if pred.rejected:
            reject_why = " | " + (", ".join(reasons) if reasons else "gate")

        kind = "cmd_rej" if pred.rejected else "cmd_ok"
        out_intent = record["intent"]
        out_col = paint(out_intent, C.BRIGHT_RED) if pred.rejected else paint(
            out_intent, C.BRIGHT_GREEN
        )
        print(
            f"{tag('cmd', kind)} raw={raw.intent}/{raw.slot or '-'} "
            f"var={raw.variation!r} → out={out_col} "
            f"conf {conf_part} margin={raw.margin:.3f} "
            f"src={raw.source} rej={pred.rejected}{reject_why} "
            f"infer={infer_ms:.0f}ms audio={audio_ms:.0f}ms",
            flush=True,
        )
        if not pred.rejected and pred.intent != OUT_OF_SCOPE:
            self.actions.dispatch(pred.intent, pred.slot)
        else:
            # Wrong / OOS: keep the room, tell the UI we didn't understand.
            self.actions.notify("reject", intent=raw.intent)
        self._dbg(f"{tag('sleep', 'sleep')} back to wake listening")
        self.state = self.STATE_SLEEP
        self._listen_started = None
        # Drop in-flight VAD so leftover command audio cannot sticky-lock wake.
        self.vad.reset()
        return record

    @staticmethod
    def _resample_linear(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
        if src_sr == dst_sr or len(audio) == 0:
            return audio.astype(np.float32, copy=False)
        n_out = int(round(len(audio) * dst_sr / src_sr))
        if n_out < 1:
            return np.zeros(0, dtype=np.float32)
        x = np.arange(len(audio), dtype=np.float64)
        xi = np.linspace(0.0, len(audio) - 1, n_out)
        return np.interp(xi, x, audio.astype(np.float64)).astype(np.float32)

    def _pick_mic(self, sd, device: Optional[int]) -> tuple[int, int, int]:
        """Return (device_index, capture_sr, capture_blocksize) for ~30 ms frames."""
        if device is None:
            # Prefer first input-capable device; fall back to default input.
            try:
                device = int(sd.default.device[0])
            except Exception:
                device = None
            if device is None or device < 0:
                for i, info in enumerate(sd.query_devices()):
                    if int(info.get("max_input_channels") or 0) > 0:
                        device = i
                        break
        if device is None or device < 0:
            raise RuntimeError(
                "No input microphone found. Plug in a USB mic and run: "
                "python -c \"import sounddevice as sd; print(sd.query_devices())\""
            )
        info = sd.query_devices(device)
        native = int(round(float(info.get("default_samplerate") or SAMPLE_RATE)))
        capture_sr = SAMPLE_RATE
        for cand in (SAMPLE_RATE, native, 44100, 48000):
            try:
                sd.check_input_settings(
                    device=device, samplerate=cand, channels=1, dtype="float32"
                )
                capture_sr = int(cand)
                break
            except Exception:
                continue
        else:
            raise RuntimeError(
                f"Mic {device!r} does not support 16/44.1/48 kHz input"
            )
        # ~30 ms at capture rate → resamples cleanly to VAD frame at 16 kHz.
        capture_block = max(1, int(round(capture_sr * self.vad.cfg.frame_ms / 1000.0)))
        return int(device), capture_sr, capture_block

    def run_mic(self, device: Optional[int] = None) -> None:
        import sounddevice as sd
        print(f"Assistant listening. Log: {self.logger.path}", flush=True)
        print(json.dumps(self.config, indent=2), flush=True)
        dev, capture_sr, capture_block = self._pick_mic(sd, device)
        self._dbg(
            f"{tag('ready', 'ready')} sleeping — say '{self.wake_phrase_norm}' "
            f"(wake thr={self.wake_threshold:.2f}; mic={dev} @ {capture_sr} Hz → {SAMPLE_RATE} Hz)"
        )
        # Callback + queue: stream.read() can block forever on flaky USB mics.
        audio_q: queue.Queue = queue.Queue(maxsize=64)

        def _callback(indata, frames, time_info, status) -> None:  # noqa: ARG001
            if status and self.debug:
                print(f"{tag('mic', 'vad')} {status}", flush=True)
            try:
                audio_q.put_nowait(indata.copy())
            except queue.Full:
                try:
                    audio_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    audio_q.put_nowait(indata.copy())
                except queue.Full:
                    pass

        def _drain_q() -> None:
            while True:
                try:
                    audio_q.get_nowait()
                except queue.Empty:
                    break

        try:
            with sd.InputStream(
                samplerate=capture_sr,
                channels=1,
                dtype="float32",
                blocksize=capture_block,
                device=dev,
                callback=_callback,
            ):
                last_frame_t = time.perf_counter()
                last_beat_t = last_frame_t
                while True:
                    try:
                        data = audio_q.get(timeout=1.0)
                        last_frame_t = time.perf_counter()
                    except queue.Empty:
                        now = time.perf_counter()
                        if now - last_frame_t > 3.0:
                            self._dbg(
                                f"{tag('mic', 'vad')} no mic frames for "
                                f"{now - last_frame_t:.1f}s — is the USB mic OK?"
                            )
                            last_frame_t = now
                        if now - last_beat_t > 8.0:
                            self._dbg(
                                f"{tag('ready', 'ready')} still sleeping — say "
                                f"'{self.wake_phrase_norm}'"
                            )
                            last_beat_t = now
                        continue
                    mono = np.asarray(data).reshape(-1)
                    if capture_sr != SAMPLE_RATE:
                        mono = self._resample_linear(mono, capture_sr, SAMPLE_RATE)
                    frame = self.vad.frame
                    if len(mono) < frame:
                        mono = np.pad(mono, (0, frame - len(mono)))
                    elif len(mono) > frame:
                        mono = mono[:frame]
                    utt = self.vad.push(mono)
                    if utt is not None and len(utt) > SAMPLE_RATE * 0.2:
                        self.handle_utterance(utt)
                        # Stale audio captured during decode would false-wake.
                        _drain_q()
                        self.vad.reset()
        except KeyboardInterrupt:
            print("\nStopped.", flush=True)
        finally:
            self.logger.close()

    def run_wav(self, paths) -> None:
        import soundfile as sf
        for p in paths:
            audio, sr = sf.read(p, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != SAMPLE_RATE:
                # linear resample
                n = int(len(audio) * SAMPLE_RATE / sr)
                audio = np.interp(
                    np.linspace(0, len(audio) - 1, n),
                    np.arange(len(audio)), audio,
                ).astype(np.float32)
            # Force listen state for offline wav tests of commands
            self.state = self.STATE_LISTEN
            self.handle_utterance(audio)
        self.logger.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="Command ONNX path")
    ap.add_argument("--wake-model", default=None)
    ap.add_argument("--reject-gate", default=None)
    ap.add_argument("--mic", action="store_true")
    ap.add_argument("--device", type=int, default=None,
                    help="sounddevice input device index (default: first mic)")
    ap.add_argument("--wav", nargs="*", default=None)
    ap.add_argument("--student-id", default="me2")
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--prefer", default="auto")
    ap.add_argument("--wake-threshold", type=float, default=0.82)
    ap.add_argument("--wake-phrase", default="kaki",
                    help="Expected wake phrase label (training/docs; ONNX is binary)")
    ap.add_argument("--min-wake-rms", type=float, default=0.040,
                    help="Ignore wake candidates quieter than this mean RMS "
                         "(blocks room-noise false wakes)")
    ap.add_argument("--min-wake-peak-rms", type=float, default=0.09,
                    help="Require a speech-like peak frame RMS "
                         "(Pi USB often ~0.09–0.15; use 0.08 if too deaf)")
    ap.add_argument("--min-wake-ms", type=float, default=200.0,
                    help="Ignore wake candidates shorter than this (active crop)")
    ap.add_argument("--max-wake-ms", type=float, default=1600.0,
                    help="Ignore wake candidates longer than this (active crop)")
    ap.add_argument("--min-cmd-rms", type=float, default=0.035,
                    help="Ignore quiet command leftovers after wake")
    ap.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True,
                    help="Print VAD/wake/command diagnostics (default: on)")
    ap.add_argument("--demo-ui", action="store_true",
                    help="POST wake/command events to the Tamagotchi localhost demo")
    ap.add_argument("--demo-url", default="http://127.0.0.1:8765",
                    help="Tamagotchi demo base URL (default http://127.0.0.1:8765)")
    args = ap.parse_args(argv)

    asst = VoiceAssistant(
        command_model=Path(args.model),
        wake_model=Path(args.wake_model) if args.wake_model else None,
        reject_gate=Path(args.reject_gate) if args.reject_gate else None,
        student_id=args.student_id,
        log_dir=args.log_dir,
        prefer=args.prefer,
        wake_phrase_norm=str(args.wake_phrase).strip().lower(),
        wake_threshold=args.wake_threshold,
        min_wake_rms=float(args.min_wake_rms),
        min_wake_peak_rms=float(args.min_wake_peak_rms),
        min_wake_ms=float(args.min_wake_ms),
        max_wake_ms=float(args.max_wake_ms),
        min_cmd_rms=float(args.min_cmd_rms),
        debug=bool(args.debug),
        demo_ui=bool(args.demo_ui),
        demo_url=str(args.demo_url),
    )
    if args.mic:
        asst.run_mic(device=args.device)
    elif args.wav:
        asst.run_wav(args.wav)
    else:
        ap.error("Pass --mic or --wav")


if __name__ == "__main__":
    main()
