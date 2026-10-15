"""Always-listening Pi assistant: wake -> command -> reject/dispatch -> sleep."""
from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

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
        # Dual endpointing: snappy for wake, patient for commands so mid-phrase
        # pauses do not cut "turn on the lights" in half.
        self._vad_wake = dict(
            energy_threshold=0.018,
            start_ratio=2.0,
            start_ms=90,
            end_ms=280,
            max_utterance_ms=2000,
        )
        self._vad_listen = dict(
            energy_threshold=0.016,
            start_ratio=1.8,
            start_ms=120,
            end_ms=850,  # wait ~0.85s of silence before decoding
            max_utterance_ms=6000,
        )
        self.vad = EnergyVAD(VADConfig(**self._vad_wake))
        self.state = self.STATE_SLEEP
        self.wake_phrase_norm = wake_phrase_norm
        self.wake_threshold = wake_threshold
        self.min_wake_rms = min_wake_rms
        self.min_wake_peak_rms = min_wake_peak_rms
        self.min_wake_ms = min_wake_ms
        self.max_wake_ms = max_wake_ms
        self.min_cmd_rms = min_cmd_rms
        self.listen_timeout_s = 7.0
        self.min_cmd_ms = 450.0  # ignore tiny mid-pause fragments while listening
        self._listen_started: float | None = None
        self._mute_until = 0.0
        self._mute_reason = ""
        self.char_reward = char_reward
        self.prefer = prefer
        self.debug = debug
        self.demo_ui = bool(demo_ui)
        self.demo_url = (demo_url or "http://127.0.0.1:8765").rstrip("/")
        # Cached after first successful mic open — reconnect must not re-probe
        # (check_input_settings + ALSA close on a dead USB stream can take minutes).
        self._mic_cache: Optional[Tuple[int, int, int, str]] = None
        self.logger = BenchmarkLogger(student_id=student_id, log_dir=log_dir)
        if demo_ui:
            self.actions = HttpDemoDispatcher(base_url=self.demo_url, verbose=True)
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
            "demo_url": self.demo_url if demo_ui else None,
            "listen_timeout_s": self.listen_timeout_s,
            "prefer": prefer,
            "debug": debug,
            "reject_gate": self.gate.to_dict(),
        }

    def _use_wake_vad(self) -> None:
        self.vad.configure(**self._vad_wake)
        self.vad.reset()

    def _use_listen_vad(self) -> None:
        self.vad.configure(**self._vad_listen)
        self.vad.reset()

    @staticmethod
    def _estimate_speech_s(text: str) -> float:
        # Generous vs typical speechSynthesis (often slower than natural talk).
        words = max(1, len(str(text or "").split()))
        return float(min(14.0, max(2.2, words * 0.55 + 1.0)))

    def _mute_for(self, seconds: float, reason: str, *, quiet: bool = False) -> None:
        until = time.perf_counter() + max(0.0, float(seconds))
        extended = until > self._mute_until + 0.05
        if until > self._mute_until:
            self._mute_until = until
            self._mute_reason = reason
            if extended and not quiet:
                self._dbg(
                    f"{tag('mute', 'vad')} hold {seconds:.2f}s ({reason}) — "
                    f"ignoring mic until speakers finish"
                )

    def _speaker_busy(self) -> bool:
        """True while local mute or demo UI reports TTS/chime playing."""
        now = time.perf_counter()
        if now < self._mute_until:
            return True
        if not self.demo_ui:
            return False
        try:
            import urllib.request
            with urllib.request.urlopen(
                f"{self.demo_url}/audio-gate", timeout=0.15
            ) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if bool(data.get("busy")):
                rem = float(data.get("remaining_ms") or 0.0) / 1000.0
                # Mirror remote gate locally so we don't hammer GET every frame.
                self._mute_for(max(0.05, rem), "demo_audio_gate", quiet=True)
                return True
        except Exception:
            return False
        return False

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

        # Speakers still playing (wake chime / reject cue / TTS) — do not
        # treat loopback as kaki or as a command.
        if self._speaker_busy():
            if self.state == self.STATE_LISTEN:
                # Restart listen window after speakers finish.
                self._listen_started = time.perf_counter()
            self._dbg(
                f"{tag('mute', 'vad')} skip loopback "
                f"({self._mute_reason or 'speakers'}) "
                f"audio={ms:.0f}ms"
            )
            self.vad.reset()
            return None

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
                self._use_listen_vad()
                print(
                    f"{tag('listen', 'listen')} say a command after "
                    f"'{self.wake_phrase_norm}'…",
                    flush=True,
                )
                # Alexa-like: beep + listening UI (browser handles chime/TTS).
                # Mute command decode until the chime finishes (mic hears it).
                self._mute_for(0.70, "wake_chime")
                self.actions.notify("wake", phrase=self.wake_phrase_norm)
                # Duck Spotify so the command can be heard over music.
                try:
                    from .spotify import maybe_duck_for_listen
                    maybe_duck_for_listen(verbose=self.debug, fraction=0.20)
                except Exception:
                    pass
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
                self._mute_for(0.55, "listen_timeout_cue")
                self.actions.notify("listen_end", reason="timeout")
                try:
                    from .spotify import maybe_restore_after_duck
                    maybe_restore_after_duck(verbose=self.debug, delay_s=0.55)
                except Exception:
                    pass
                self._use_wake_vad()
                return None
        # Mid-phrase pause can still emit a short fragment — keep listening.
        if ms < self.min_cmd_ms:
            self._dbg(
                f"{tag('cmd', 'vad')} skip short fragment "
                f"audio={ms:.0f}ms < {self.min_cmd_ms:.0f}ms "
                f"(still listening — finish your command)"
            )
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
        unduck_delay = 0.0
        if not pred.rejected and pred.intent != OUT_OF_SCOPE:
            spoken = self.actions.dispatch(pred.intent, pred.slot)
            # Wait out TTS so confirmations are not heard as a wake/cmd.
            tts_s = self._estimate_speech_s(spoken) + 0.6
            self._mute_for(tts_s, "tts_confirm")
            # Keep Spotify ducked while the den speaks (weather, time, etc.).
            # Extra leeway so unduck doesn't overlap the end of long TTS.
            # PLAY_MUSIC already restored inside Spotify handle_intent.
            if pred.intent not in ("PLAY_MUSIC", "PAUSE", "STOP"):
                unduck_delay = tts_s + 1.25
            else:
                unduck_delay = 0.0  # already restored (or N/A)
        else:
            # Wrong / OOS: keep the room, tell the UI we didn't understand.
            self.actions.notify("reject", intent=raw.intent)
            # Also absorb the rest of the user's unfinished phrase.
            self._mute_for(1.4, "reject_cue_plus_tail")
            unduck_delay = 1.4
        try:
            from .spotify import maybe_restore_after_duck
            if unduck_delay > 0:
                maybe_restore_after_duck(verbose=self.debug, delay_s=unduck_delay)
            elif pred.intent not in ("PLAY_MUSIC", "PAUSE", "STOP"):
                maybe_restore_after_duck(verbose=self.debug, delay_s=0.0)
        except Exception:
            pass
        self._dbg(f"{tag('sleep', 'sleep')} back to wake listening")
        self.state = self.STATE_SLEEP
        self._listen_started = None
        # Drop in-flight VAD so leftover command audio cannot sticky-lock wake.
        self._use_wake_vad()
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
            # Prefer USB PnP / USB Audio over built-in laptop mics when present.
            prefer_substrings = (
                "usb pnp", "usb audio", "usb microphone", "usb mic",
            )
            skip_substrings = (
                "macbook", "built-in", "iphone", "continuity", "aggregate",
            )
            inputs = []
            for i, info in enumerate(sd.query_devices()):
                if int(info.get("max_input_channels") or 0) <= 0:
                    continue
                name = str(info.get("name") or "")
                inputs.append((i, name))
            chosen = None
            for i, name in inputs:
                low = name.lower()
                if any(s in low for s in skip_substrings):
                    continue
                if any(s in low for s in prefer_substrings):
                    chosen = i
                    break
            if chosen is None:
                try:
                    chosen = int(sd.default.device[0])
                except Exception:
                    chosen = None
            if chosen is None or chosen < 0:
                for i, name in inputs:
                    low = name.lower()
                    if any(s in low for s in skip_substrings):
                        continue
                    chosen = i
                    break
            if chosen is None and inputs:
                chosen = inputs[0][0]
            device = chosen
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
        # USB mics on Pi often overflow then silently stop delivering frames.
        # Outer loop reopens capture instantly (cached device; abort+async close).
        reconnect_n = 0
        try:
            while True:
                try:
                    self._run_mic_session(sd, device, reconnect_n)
                    # Clean stall return — reopen immediately, no backoff.
                    reconnect_n = max(1, reconnect_n + 1)
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    reconnect_n += 1
                    self._mic_cache = None  # force re-probe after hard failure
                    wait = min(2.0, 0.25 * reconnect_n)
                    print(
                        f"{tag('mic', 'vad')} mic session error ({exc}); "
                        f"reopening in {wait:.1f}s…",
                        flush=True,
                    )
                    time.sleep(wait)
        except KeyboardInterrupt:
            print("\nStopped.", flush=True)
        finally:
            self.logger.close()

    @staticmethod
    def _close_stream_fast(stream, *, timeout_s: float = 0.35) -> None:
        """Abort+close without blocking for minutes on a wedged USB/ALSA device."""
        def _worker() -> None:
            try:
                stream.abort()
            except Exception:
                try:
                    stream.stop()
                except Exception:
                    pass
            try:
                stream.close()
            except Exception:
                pass

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(timeout=timeout_s)
        if t.is_alive():
            print(
                f"{tag('mic', 'vad')} stream close still hanging — "
                f"continuing without waiting",
                flush=True,
            )

    def _run_mic_session(
        self, sd, device: Optional[int], reconnect_n: int
    ) -> None:
        """One InputStream lifetime. Returns to reopen; raises to retry/backoff."""
        if self._mic_cache is not None and reconnect_n > 0:
            dev, capture_sr, capture_block, mic_name = self._mic_cache
        else:
            dev, capture_sr, capture_block = self._pick_mic(sd, device)
            # 3× block = ~90ms — fewer USB overflows on Pi than 30ms frames.
            capture_block = max(capture_block, capture_block * 3)
            mic_name = str(sd.query_devices(dev).get("name") or "?")
            self._mic_cache = (dev, capture_sr, capture_block, mic_name)

        if reconnect_n == 0:
            self._dbg(
                f"{tag('ready', 'ready')} sleeping — say '{self.wake_phrase_norm}' "
                f"(wake thr={self.wake_threshold:.2f}; "
                f"mic={dev} {mic_name!r} @ {capture_sr} Hz → {SAMPLE_RATE} Hz, "
                f"block={capture_block})"
            )
        else:
            print(
                f"{tag('mic', 'vad')} reopening mic={dev} {mic_name!r} "
                f"@ {capture_sr} Hz (#{reconnect_n})",
                flush=True,
            )

        audio_q: queue.Queue = queue.Queue(maxsize=256)
        overflow_n = 0

        def _callback(indata, frames, time_info, status) -> None:  # noqa: ARG001
            nonlocal overflow_n
            if status:
                overflow_n += 1
                if self.debug and (overflow_n <= 3 or overflow_n % 50 == 0):
                    print(
                        f"{tag('mic', 'vad')} {status} (x{overflow_n})",
                        flush=True,
                    )
                while audio_q.qsize() > 4:
                    try:
                        audio_q.get_nowait()
                    except queue.Empty:
                        break
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

        stream = sd.InputStream(
            samplerate=capture_sr,
            channels=1,
            dtype="float32",
            blocksize=capture_block,
            device=dev,
            latency="high",
            callback=_callback,
        )
        stream.start()
        last_frame_t = time.perf_counter()
        last_beat_t = last_frame_t
        last_level_t = last_frame_t
        level_peak = 0.0
        level_n = 0
        tail = np.zeros(0, dtype=np.float32)
        try:
            while True:
                try:
                    data = audio_q.get(timeout=0.5)
                    last_frame_t = time.perf_counter()
                except queue.Empty:
                    now = time.perf_counter()
                    gap = now - last_frame_t
                    # Fail fast: ~1.5s of silence from a live USB mic ⇒ reopen.
                    if gap > 1.5:
                        print(
                            f"{tag('mic', 'vad')} no frames {gap:.1f}s — "
                            f"reopening capture now",
                            flush=True,
                        )
                        self.vad.reset()
                        return
                    if now - last_beat_t > 8.0:
                        self._dbg(
                            f"{tag('ready', 'ready')} still sleeping — say "
                            f"'{self.wake_phrase_norm}'"
                        )
                        last_beat_t = now
                    continue
                mono = np.asarray(data, dtype=np.float32).reshape(-1)
                if capture_sr != SAMPLE_RATE:
                    mono = self._resample_linear(mono, capture_sr, SAMPLE_RATE)
                level_peak = max(
                    level_peak,
                    float(np.max(np.abs(mono))) if len(mono) else 0.0,
                )
                level_n += 1
                now = time.perf_counter()
                if now - last_level_t >= 1.0:
                    self._dbg(
                        f"{tag('mic', 'vad')} level peak={level_peak:.4f} "
                        f"blocks={level_n}"
                        + (
                            " — SILENT (check Mic privacy + Sound Input)"
                            if level_peak < 0.002
                            else ""
                        )
                    )
                    level_peak = 0.0
                    level_n = 0
                    last_level_t = now
                # Feed VAD in exact frame-sized chunks (capture block may be larger).
                frame = self.vad.frame
                buf = np.concatenate([tail, mono]) if len(tail) else mono
                offset = 0
                while offset + frame <= len(buf):
                    chunk = buf[offset:offset + frame]
                    offset += frame
                    utt = self.vad.push(chunk)
                    if utt is not None and len(utt) > SAMPLE_RATE * 0.2:
                        self.handle_utterance(utt)
                        _drain_q()
                        self.vad.reset()
                        tail = np.zeros(0, dtype=np.float32)
                        # Drop remainder of this buf — it's pre-decode audio.
                        offset = len(buf)
                        break
                else:
                    tail = np.asarray(buf[offset:], dtype=np.float32).copy()
        finally:
            self._close_stream_fast(stream)

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
