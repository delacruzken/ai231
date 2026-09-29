"""Standalone, torch-free inference for the RPi5.

Pipeline:  WAV/mic -> NumPy 40-band log-mel -> ONNX Runtime encoder
           -> log-softmax -> grammar forced-alignment decode -> intent.

Dependencies: numpy, onnxruntime, soundfile.  NO torch / torchaudio.

Usage:
  python -m grammar_asr.infer_rpi --wav path/to/cmd.wav
  python -m grammar_asr.infer_rpi --wav a.wav b.wav c.wav
  python -m grammar_asr.infer_rpi --mic          # live microphone loop
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

# Allow running either as a package (python -m grammar_asr.infer_rpi) or as a
# flat script on the RPi (python infer_rpi.py) where the files sit side-by-side.
try:
    from .mel_numpy import wav_to_feat, SAMPLE_RATE
    from .grammar import build_grammar
    try:
        from .decode_rpi import score_all_words      # torch-free (preferred)
    except ImportError:
        from .decode_grammar import score_all_words
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from mel_numpy import wav_to_feat, SAMPLE_RATE
    from grammar import build_grammar
    try:
        from decode_rpi import score_all_words
    except ImportError:
        from decode_grammar import score_all_words

DEFAULT_MODEL = Path(__file__).resolve().parent / "runs" / "low30" / "model.onnx"


class GrammarASR:
    """Tiny on-device spoken-command recognizer (PocketSphinx-style)."""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL,
                 char_reward: float = 0.15):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(model_path), opts,
                                         providers=["CPUExecutionProvider"])
        self.trie = build_grammar()
        self.char_reward = char_reward

    def _log_probs(self, feat: np.ndarray) -> np.ndarray:
        x = feat[None].astype(np.float32)
        logits = self.sess.run(None, {"feat": x})[0][0]      # (T', C)
        m = logits.max(-1, keepdims=True)
        return logits - (m + np.log(np.exp(logits - m).sum(-1, keepdims=True)))

    def decode_features(self, feat: np.ndarray) -> dict:
        lp = self._log_probs(feat)
        res = score_all_words(lp, self.trie, length_norm=True,
                              char_reward=self.char_reward)
        if not res:
            return {"intent": None, "text": "", "score": -1e9}
        r = res[0]
        return {"intent": r.intent, "text": r.transcript, "score": r.score}

    def decode_wav(self, path: str | Path) -> dict:
        feat = wav_to_feat(str(path))
        return self.decode_features(feat)

    def decode_audio(self, audio: np.ndarray) -> dict:
        """audio: float32 mono @ 16 kHz."""
        from .mel_numpy import normalize_feat, log_mel
        feat = normalize_feat(log_mel(audio.astype(np.float32)))
        return self.decode_features(feat)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wav", nargs="*", help="WAV file(s) to decode")
    ap.add_argument("--mic", action="store_true", help="live microphone loop")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--seconds", type=float, default=3.0,
                    help="capture length per utterance in --mic mode")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    asr = GrammarASR(args.model)

    if args.wav:
        for p in args.wav:
            t0 = time.time()
            r = asr.decode_wav(p)
            dt = (time.time() - t0) * 1000
            print(f"{Path(p).name:34s} -> {r['intent']}  "
                  f"(\"{r['text']}\")  {dt:.0f} ms")
        return

    if args.mic:
        _mic_loop(asr, args.seconds)
        return

    ap.print_help()


def _mic_loop(asr: GrammarASR, seconds: float):
    import sounddevice as sd
    print(f"Listening on microphone ({seconds:.1f}s captures). Ctrl-C to stop.")
    print("-" * 50)
    frame = int(seconds * SAMPLE_RATE)
    try:
        while True:
            t0 = time.time()
            audio = sd.rec(frame, samplerate=SAMPLE_RATE, channels=1,
                           dtype="float32")
            sd.wait()
            mono = audio.mean(axis=1)
            r = asr.decode_audio(mono)
            dt = (time.time() - t0) * 1000
            intent = r["intent"] or "(no match)"
            print(f"[{time.strftime('%H:%M:%S')}] {intent:16s} "
                  f"\"{r['text']}\"  ({dt:.0f} ms incl. capture)")
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
