# Deploying the Grammar-ASR VCM to the Raspberry Pi 5

Standalone, on-device, real-time, no LLM, no cloud (Tasks 5–8).
Model: `model.onnx` (**7.76 MB fp32, single self-contained file**, ~1.93M params).
Inference: **NumPy mel + ONNX Runtime + grammar forced-alignment** — no torch.

Validated on the dev machine (torch-free flat-script path, no data dir):
**97% command accuracy (100-sample clean test), ~42 ms p50 infer+decode.**
Per-intent smoke test: 15/15 distinct intents + 5/5 weak intents correct.

---

## 0. What ships to the Pi (the `dist/` folder, ~7.8 MB total)

```
dist/
  model.onnx              7.76 MB  single-file ONNX (weights inlined)
  mel_numpy.py                   torch-free 40-band log-mel
  grammar.py                     trie + static grammar fallback
  grammar_commands.json          93 command strings (no manifest needed)
  decode_rpi.py                  torch-free forced-alignment decoder
  infer_rpi.py                   CLI: --wav / --mic
  requirements-rpi.txt           numpy, onnxruntime, soundfile, sounddevice
```

Everything is self-contained — **no dataset, no manifest, no torch**. Copy the
whole `dist/` folder to the Pi.

## 1. Flash the Pi 5

1. Raspberry Pi Imager → **Raspberry Pi OS 64-bit** (Bookworm, Lite is fine).
2. Boot, then:
   ```bash
   sudo apt update && sudo apt full-upgrade -y
   sudo apt install -y python3-pip python3-venv portaudio19-dev libsndfile1
   ```
   (`portaudio19-dev` + `libsndfile1` are the system libs `sounddevice`/
   `soundfile` need for the microphone.)

## 2. Python env + deps

```bash
mkdir ~/me2 && cd ~/me2
# (copy dist/ here as ./me2)
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-rpi.txt
```

> **onnxruntime on arm64:** Microsoft publishes official aarch64 Linux wheels
> for Python 3.9–3.12. Bookworm ships Python 3.11 → `pip install onnxruntime`
> works out of the box.

## 3. Run it

```bash
# decode a saved WAV
python infer_rpi.py --model model.onnx --wav /path/to/command.wav

# live microphone (3 s captures, prints intent per utterance)
python infer_rpi.py --model model.onnx --mic --seconds 3
```

Expected: intent printed in < 150 ms after you stop speaking (capture time
excluded). Weak/confusable intents (PAUSE, LIGHT_OFF, VOLUME_DOWN, NEXT, STOP)
are the known soft spots — see the accuracy table in the main README.

## 4. (Optional) int8 — smaller + faster CPU

The fp32 model (7.76 MB) already fits easily on a 4 GB Pi 5. To shrink further:
```bash
python -m grammar_asr.export_onnx \
    --ckpt runs/low30/best.pt --out model.onnx --model low --quantize
```
(Quantization needs `onnxruntime` with the quantization extra. Optional — the
fp32 model already meets the < 10 MB / real-time spec.)

---

## Notes / gotchas found during packaging

- **ONNX external data:** torch exports weights to a sibling `model.onnx.data`.
  `export_onnx.py` now inlines it so the shipped `.onnx` is one file. If you
  re-export and forget, the Pi will fail with "External data path validation".
- **Sample rate:** some dataset WAVs are 12 kHz. `mel_numpy.wav_to_feat`
  resamples to 16 kHz automatically (linear). Mic input is captured at 16 kHz.
- **Grammar is static:** `grammar_commands.json` (93 commands) ships with the
  model, so the Pi never needs the dataset manifest. To add a command, edit the
  JSON (and retrain only if you add a new *character*).
