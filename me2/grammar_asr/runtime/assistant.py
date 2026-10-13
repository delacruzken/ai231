"""Always-listening Pi assistant: wake -> command -> reject/dispatch -> sleep."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

from ..data.features import SAMPLE_RATE, normalize_feat, log_mel
from ..decode.grammar_trie import build_grammar
from ..decode.semantic import RejectGate, decode_model_output, SemanticPrediction
from ..schema import OUT_OF_SCOPE
from .actions import MockDispatcher
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
        min_wake_rms: float = 0.025,
        min_wake_ms: float = 400.0,
        min_cmd_rms: float = 0.012,
    ):
        self.command = OnnxSession(command_model)
        self.wake = OnnxSession(wake_model) if wake_model else None
        self.gate = RejectGate.load(reject_gate) if reject_gate else RejectGate(
            min_confidence=0.35, min_margin=0.05
        )
        self.trie = build_grammar()
        # Sleep-time VAD is stricter so quiet room noise does not form utterances.
        self.vad = EnergyVAD(VADConfig(
            energy_threshold=0.015,
            start_ratio=2.0,
            start_ms=150,
            end_ms=500,
            max_utterance_ms=5000,
        ))
        self.state = self.STATE_SLEEP
        self.wake_phrase_norm = wake_phrase_norm
        self.wake_threshold = wake_threshold
        self.min_wake_rms = min_wake_rms
        self.min_wake_ms = min_wake_ms
        self.min_cmd_rms = min_cmd_rms
        self.char_reward = char_reward
        self.prefer = prefer
        self.debug = debug
        self.logger = BenchmarkLogger(student_id=student_id, log_dir=log_dir)
        self.actions = MockDispatcher(verbose=True)
        self.config = {
            "command_model": str(command_model),
            "wake_model": str(wake_model) if wake_model else None,
            "wake_threshold": wake_threshold,
            "min_wake_rms": min_wake_rms,
            "min_wake_ms": min_wake_ms,
            "prefer": prefer,
            "debug": debug,
            "reject_gate": self.gate.to_dict(),
        }

    def _dbg(self, msg: str) -> None:
        if self.debug:
            print(msg, flush=True)

    def _audio_stats(self, audio: np.ndarray) -> tuple[float, float]:
        ms = len(audio) / SAMPLE_RATE * 1000.0
        rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float64))) + 1e-12))
        return ms, rms

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
        ms, rms = self._audio_stats(audio)
        self._dbg(
            f"{tag('vad', 'vad')} state={self.state} audio={ms:.0f}ms "
            f"rms={rms:.4f} frames={len(audio)}"
        )

        if self.state == self.STATE_SLEEP:
            if ms < self.min_wake_ms or rms < self.min_wake_rms:
                self._dbg(
                    f"{tag('wake', 'wake_skip')} skip energy "
                    f"audio={ms:.0f}ms rms={rms:.4f} "
                    f"(need>={self.min_wake_ms:.0f}ms & rms>={self.min_wake_rms:.3f})"
                )
                return None
            score = self._wake_score(audio)
            hit = score >= self.wake_threshold
            scored = score_vs_thr(score, self.wake_threshold)
            if hit:
                self.logger.wake(score)
                print(
                    f"{tag('wake', 'wake_hit')} HIT {scored} "
                    f"audio={ms:.0f}ms rms={rms:.4f}",
                    flush=True,
                )
                self.state = self.STATE_LISTEN
                print(
                    f"{tag('listen', 'listen')} say a command after "
                    f"'{self.wake_phrase_norm}'…",
                    flush=True,
                )
            else:
                self._dbg(
                    f"{tag('wake', 'wake_miss')} miss {scored} "
                    f"audio={ms:.0f}ms rms={rms:.4f}"
                )
            return None

        # Listening for a command — ignore near-silence leftovers after wake.
        if rms < self.min_cmd_rms:
            self._dbg(
                f"{tag('cmd', 'vad')} skip quiet leftover "
                f"audio={ms:.0f}ms rms={rms:.4f} (still listening)"
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
        self._dbg(f"{tag('sleep', 'sleep')} back to wake listening")
        self.state = self.STATE_SLEEP
        return record

    def run_mic(self) -> None:
        import sounddevice as sd
        print(f"Assistant listening. Log: {self.logger.path}", flush=True)
        print(json.dumps(self.config, indent=2), flush=True)
        self._dbg(
            f"{tag('ready', 'ready')} sleeping — say '{self.wake_phrase_norm}' "
            f"(wake thr={self.wake_threshold:.2f})"
        )
        frame = self.vad.frame
        try:
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                dtype="float32", blocksize=frame) as stream:
                while True:
                    data, _ = stream.read(frame)
                    mono = data.reshape(-1)
                    utt = self.vad.push(mono)
                    if utt is not None and len(utt) > SAMPLE_RATE * 0.2:
                        self.handle_utterance(utt)
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
    ap.add_argument("--wav", nargs="*", default=None)
    ap.add_argument("--student-id", default="me2")
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--prefer", default="auto")
    ap.add_argument("--wake-threshold", type=float, default=0.85)
    ap.add_argument("--wake-phrase", default="kaki",
                    help="Expected wake phrase label (training/docs; ONNX is binary)")
    ap.add_argument("--min-wake-rms", type=float, default=0.025,
                    help="Ignore wake candidates quieter than this RMS "
                         "(blocks room-noise false wakes)")
    ap.add_argument("--min-wake-ms", type=float, default=400.0,
                    help="Ignore wake candidates shorter than this")
    ap.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True,
                    help="Print VAD/wake/command diagnostics (default: on)")
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
        min_wake_ms=float(args.min_wake_ms),
        debug=bool(args.debug),
    )
    if args.mic:
        asst.run_mic()
    elif args.wav:
        asst.run_wav(args.wav)
    else:
        ap.error("Pass --mic or --wav")


if __name__ == "__main__":
    main()
