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
        wake_phrase_norm: str = "hey kiwi",
        wake_threshold: float = 0.7,
        student_id: str = "me2",
        log_dir: Optional[str] = None,
        char_reward: float = 0.15,
        prefer: str = "auto",
    ):
        self.command = OnnxSession(command_model)
        self.wake = OnnxSession(wake_model) if wake_model else None
        self.gate = RejectGate.load(reject_gate) if reject_gate else RejectGate(
            min_confidence=0.35, min_margin=0.05
        )
        self.trie = build_grammar()
        self.vad = EnergyVAD()
        self.state = self.STATE_SLEEP
        self.wake_phrase_norm = wake_phrase_norm
        self.wake_threshold = wake_threshold
        self.char_reward = char_reward
        self.prefer = prefer
        self.logger = BenchmarkLogger(student_id=student_id, log_dir=log_dir)
        self.actions = MockDispatcher(verbose=True)
        self.config = {
            "command_model": str(command_model),
            "wake_model": str(wake_model) if wake_model else None,
            "wake_threshold": wake_threshold,
            "prefer": prefer,
            "reject_gate": self.gate.to_dict(),
        }

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
        pred = self.gate.apply(pred)
        infer_ms = (time.perf_counter() - t0) * 1000
        audio_ms = len(audio) / SAMPLE_RATE * 1000
        return pred, infer_ms, audio_ms

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
        if self.state == self.STATE_SLEEP:
            score = self._wake_score(audio)
            if score >= self.wake_threshold:
                self.logger.wake(score)
                print(f"[wake] score={score:.3f}")
                self.state = self.STATE_LISTEN
            return None

        # Listening for a command
        pred, infer_ms, audio_ms = self._predict_command(audio)
        record = pred.to_log_dict(infer_ms, audio_ms)
        record["model"] = self.config["command_model"]
        self.logger.log(record)
        print(
            f"[cmd] intent={record['intent']} slot={record.get('slot','')} "
            f"rej={pred.rejected} {infer_ms:.0f}ms"
        )
        if not pred.rejected and pred.intent != OUT_OF_SCOPE:
            self.actions.dispatch(pred.intent, pred.slot)
        self.state = self.STATE_SLEEP
        return record

    def run_mic(self) -> None:
        import sounddevice as sd
        print(f"Assistant listening. Log: {self.logger.path}")
        print(json.dumps(self.config, indent=2))
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
            print("\nStopped.")
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
    ap.add_argument("--wake-threshold", type=float, default=0.7)
    args = ap.parse_args(argv)

    asst = VoiceAssistant(
        command_model=Path(args.model),
        wake_model=Path(args.wake_model) if args.wake_model else None,
        reject_gate=Path(args.reject_gate) if args.reject_gate else None,
        student_id=args.student_id,
        log_dir=args.log_dir,
        prefer=args.prefer,
        wake_threshold=args.wake_threshold,
    )
    if args.mic:
        asst.run_mic()
    elif args.wav:
        asst.run_wav(args.wav)
    else:
        ap.error("Pass --mic or --wav")


if __name__ == "__main__":
    main()
