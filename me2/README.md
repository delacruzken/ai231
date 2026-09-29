# ME2 — Voice Command Model (VCM) on Raspberry Pi 5

**Goal:** A tiny, real-time, fully **on-device** voice command system that recognizes
smart-home commands from the microphone — **no LLM, no cloud, no internet at inference**.

| | |
|---|---|
| **Hardware** | Raspberry Pi 5 (8 GB) — must also run on RPi 4 GB |
| **Spec** | `ME2 - Specifications.pdf` (in this folder) |
| **Dataset** | `AI231 ME2 - Datasets.pdf`, `Dataset Schema - Option B.pdf` |
| **Status** | 🟢 **Core model complete & deployable** — RPi5 live-mic demo is the remaining task |
| **Last updated** | Sep 29, 2026 |

---

## Where we are standing (TL;DR)

We built a **grammar-constrained ASR** system — *not* a one-shot intent classifier.
It replicates **Anthony Navarez's** approach (AI231, 9/22): predict characters with a
tiny CTC model, then decode against a **grammar trie** so the system can only ever
emit one of the 93 known commands. This is how **PocketSphinx** works.

**Result:** a **1.93 M-param** model (7.76 MB fp32, ~0.5 MB int8) that hits
**80.8% clean / 85.4% noisy command accuracy** at **~2 ms** inference latency,
and a **self-contained 7.4 MB `dist/` folder** ready to copy onto the Pi.

| Metric | Value |
|---|---|
| Clean command accuracy | **80.76%** (726/899) |
| Noisy command accuracy | **85.43%** (768/899) |
| CER (clean) | 1.90% |
| Latency p50 / p95 | 2.0 / 2.3 ms |
| Model size | 1.93 M params · 7.76 MB fp32 · ~0.5 MB int8 |
| Torch-free deploy path | **97%** on 100-sample clean test, ~42 ms p50 |

**Remaining:** deploy to the physical RPi5 + real-microphone demo (Tasks 5–8),
and optionally lift the weak/confusable short intents.

---

## Why this design (vs. a classifier)

| Property | One-shot classifier | Grammar-ASR (ours) |
|---|---|---|
| Model size | needs ~5–20 M params for 19 classes | **1.93 M params** (learns 27 chars, not 19 patterns) |
| Robustness | mis-hear → wrong class | trie prunes invalid paths |
| New commands | retrain | add a string to the trie |
| Footprint (int8) | several MB | **< 1 MB** (matches Anthony's RPi claim) |

The two-stage design trades a small amount of accuracy for a much smaller model
and the ability to add commands without retraining.

---

## How it works

```
WAV (16 kHz mono)
  → 40-band log-mel (25 ms window / 10 ms hop)
  → 1-D residual CNN (5 blocks, stride-2 stem only → 2× downsample)
  → Linear → 28 classes (26 letters + space + CTC blank)
  → [decode]
       greedy CTC  → raw text  (diagnostic baseline)
       trie forced-alignment  → one of 93 command strings  (the real system)
```

**Grammar:** 93 unique command strings (13 fixed intents × 3 variants + 6 slot
intents × 9 slot values), built from the dataset manifest. Digits are spelled out
("10 seconds" → "ten seconds") so the ASR target and the trie share one representation.

**Decoder:** each candidate command is scored by **CTC forced alignment** plus a small
per-character reward (`char_reward=0.15`). This is what makes long, specific commands
win fairly against short ones (a naive beam would always collapse to "stop"/"time").

---

## Folder layout

```
me2/
├── README.md                 ← you are here (overview + status)
├── PROGRESS.md               ← dated log of what was done and when
├── ME2-PLAN.md               ← original step-by-step execution plan
├── TEAM-PROGRESS.md          ← team-wide synthesis (both groups, decisions, owners)
├── ME2 - Specifications.pdf  ← the assignment spec
├── AI231 ME2 - Datasets.pdf  ← dataset overview
├── Dataset Schema - Option B.pdf
├── data/                     ← Mark's OptionB dataset (1.7 GB, git-ignored)
└── grammar_asr/              ← THE CODE
    ├── README.md             ← deep-dive: architecture, results, lessons
    ├── DEPLOY-RPI5.md        ← step-by-step Pi deployment guide
    ├── grammar.py            ← alphabet, number→words, trie, manifest loader
    ├── features.py           ← WAV→log-mel, dataset, dataloaders
    ├── asr_model.py          ← CTCEncoder (base), ctc_loss, greedy_decode
    ├── asr_model_res.py      ← residual variant (1.56 M)
    ├── asr_model_low.py      ← LOW variant (1.93 M, the production model)
    ├── train_asr.py          ← base training loop
    ├── train_asr_res.py      ← residual training loop
    ├── train_asr_low.py      ← low training loop
    ├── decode_grammar.py     ← trie forced-alignment decoder + cmd-accuracy eval
    ├── benchmark.py          ← 3-tier benchmark (clean/noisy/latency) → JSON+MD
    ├── export_onnx.py        ← ONNX fp32 + int8 export (weights inlined)
    ├── mel_numpy.py          ← torch-free log-mel (for the Pi)
    ├── decode_rpi.py         ← torch-free forced-alignment decoder (for the Pi)
    ├── infer_rpi.py          ← Pi CLI: --wav / --mic
    ├── grammar_commands.json ← 93 command strings (ships with the model)
    ├── requirements-rpi.txt  ← Pi deps: numpy, onnxruntime, soundfile, sounddevice
    ├── dist/                 ← ★ SELF-CONTAINED SHIP FOLDER (7.4 MB) → copy to Pi
    └── runs/                 ← checkpoints + benchmark outputs (git-ignored)
        ├── base/  base30/  res30/  low30/
        └── low30_bench.{md,json}
```

---

## Quick start (dev machine, Apple Silicon)

```bash
cd ai231/me2
source .venv/bin/activate

# 1. Train the production (low) model — ~60 s/epoch on MPS
python -m grammar_asr.train_asr_low --epochs 30 --tag low30

# 2. End-to-end command accuracy (grammar decode)
python -m grammar_asr.decode_grammar \
    --ckpt grammar_asr/runs/low30/best.pt \
    --split test --condition clean --mode grammar --beam 32

# 3. Full 3-tier benchmark
python -m grammar_asr.benchmark --ckpt grammar_asr/runs/low30/best.pt --device mps

# 4. Export the self-contained Pi bundle
python -m grammar_asr.export_onnx --ckpt grammar_asr/runs/low30/best.pt --model low
```

## Deploy to the RPi5 (the remaining task)

Copy the **`grammar_asr/dist/`** folder to the Pi and follow
[`grammar_asr/DEPLOY-RPI5.md`](grammar_asr/DEPLOY-RPI5.md). Short version:

```bash
# on the Pi, after flashing Bookworm 64-bit:
sudo apt install -y python3-pip python3-venv portaudio19-dev libsndfile1
mkdir ~/me2 && cd ~/me2            # (dist/ contents live here)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-rpi.txt

python infer_rpi.py --model model.onnx --wav /path/to/cmd.wav   # WAV smoke test
python infer_rpi.py --model model.onnx --mic --seconds 3        # LIVE MIC DEMO
```

---

## Results detail

### Per-intent accuracy (clean, low30)

| Intent | Acc | | Intent | Acc |
|---|---|---|---|---|
| TEMPERATURE | 100% | | MESSAGE | 100% |
| BRIGHTNESS | 100% | | ALARM | 97.7% |
| TIMER | 95.5% | | LIST_REMINDERS | 93.1% |
| CREATE_REMINDER | 92.2% | | WEATHER | 80.8% |
| COLOR | 77.8% | | CALL | 74.1% |
| TIME | 74.1% | | PLAY_MUSIC | 62.1% |
| VOLUME_UP | 60.0% | | STOP | 59.3% |
| NEXT | 53.3% | | LIGHT_ON | 40.7% |
| LIGHT_OFF | 40.0% | | PAUSE | 39.3% |
| VOLUME_DOWN | 23.1% | | | |

**Strong:** the long, specific commands (temperature, brightness, reminders, alarm,
timer, message). **Weak:** the short, confusable ones (PAUSE, LIGHT_OFF, VOLUME_DOWN,
NEXT, STOP) — these are the natural next optimization target.

### Model comparison (why "low" won)

| Model | Params | Downsample | Test CER | Cmd acc |
|---|---|---|---|---|
| base (4-layer CNN) | 276 K | 16× | 29.8% | 0% |
| res (4-block residual) | 1.56 M | 16× | 10.8% | 0.5% |
| **low (5-block residual)** | **1.93 M** | **2×** | **1.8%** | **80.8%** |

### Three hard-won design lessons

1. **Temporal downsampling must be low.** A 16× CNN (T′≈25) makes long commands
   (>25 chars) *structurally impossible* under CTC (need T′ ≥ L), so the decoder
   collapses to short words → 0%. A 2× stem (T′≈98) fixes it → 90%+.
2. **Forced alignment beats naive beam search.** A beam carrying a blank-stay
   hypothesis compounds its score each frame, so short commands always win.
   Scoring each grammar word via CTC forced alignment + a per-char reward is what
   makes long, specific commands win fairly.
3. **The ASR front-end is the bottleneck.** The 276 K base (CER 30%) gave 0% command
   accuracy — the grammar had nothing to lock onto. The 1.93 M low model (CER 1.8%)
   gives 81%. Capacity belongs in the encoder, not the decoder.

---

## Caveats

- **Synthetic data** (Chatterbox TTS, 100 speakers). Defensible for a tiny-VCM
  baseline; consider mixing real GSC/SLURP audio for a robustness comparison.
- Confirm **license** of Chatterbox + any LibriSpeech/SilencioPH references.
- The NumPy mel on the Pi is functionally (not bit-) identical to the torch mel;
  the 97% torch-free test confirms the model decodes correctly through it.

## See also

- [`PROGRESS.md`](PROGRESS.md) — dated changelog of all work to date
- [`grammar_asr/README.md`](grammar_asr/README.md) — full technical deep-dive
- [`grammar_asr/DEPLOY-RPI5.md`](grammar_asr/DEPLOY-RPI5.md) — Pi deployment guide
- [`ME2-PLAN.md`](ME2-PLAN.md) — the original execution plan
- [`TEAM-PROGRESS.md`](TEAM-PROGRESS.md) — class-wide status & decisions
