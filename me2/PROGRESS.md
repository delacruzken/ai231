# ME2 — Progress Log

Dated record of what was done, when, and what's next. Newest at the bottom.

---

## Sep 22 — Approach chosen

- **Anthony Navarez** proposed the grammar-based route (Telegram, 9/22): instead of
  one-shot classification, predict per character/phoneme and search a grammar
  structure (trie) to iteratively form the command. Reference implementation:
  **PocketSphinx**. Reported good latency + **< 1 MB** footprint on RPi 4 GB, trained
  on Sir Mark's synthetic dataset, robust even to his own voice.
- Decision: replicate this approach rather than the originally-planned one-shot
  MicroCNN/TC-ResNet classifier.

## Sep 24 — Data + pipeline built

- **Dataset cloned & verified:** Mark Macalalad's **OptionB** (Chatterbox TTS,
  100 speakers) — **17,986 active WAVs** (8,993 clean + 8,993 noisy @ ~30 dB SNR),
  16 kHz mono, 19 intents, 93 unique command strings. Speaker-split train/val/test
  (s1–80 / s81–90 / s91–100) → **no leakage**. Cloned to `data/AI231_src/MEX2/OptionB/`.
- **Full pipeline written** in `grammar_asr/`:
  - `grammar.py` — char alphabet, number→words, trie, manifest loader
  - `features.py` — 40-band log-mel, per-utterance normalization, dataloaders
  - `asr_model.py` / `_res.py` / `_low.py` — three CTC encoder variants
  - `decode_grammar.py` — trie-constrained **forced-alignment** decoder
  - `train_asr*.py`, `benchmark.py`, `export_onnx.py`
- **Trained three models** to compare capacity + downsampling:
  - base (276 K, 16×) → CER 29.8%, **0%** cmd acc
  - res (1.56 M, 16×) → CER 10.8%, **0.5%** cmd acc
  - **low (1.93 M, 2×)** → CER 1.8%, **80.8%** cmd acc ← selected
- **Diagnosed the two structural bugs** that caused the 0% models:
  1. 16× temporal downsampling made long commands impossible under CTC.
  2. Naive beam search collapsed to short commands; switched to forced alignment.
- Wrote `ME2-PLAN.md`, `TEAM-PROGRESS.md`, `grammar_asr/README.md`.

## Sep 25 — Benchmarked + finalized the production model

- Ran the **3-tier benchmark** on `low30` (30 epochs, MPS):
  - Clean **80.76%** / Noisy **85.43%** command accuracy
  - CER 1.90%, WER 5.26%, latency p50 2.0 ms / p95 2.3 ms
  - Per-intent table produced (see `runs/low30_bench.md`).
- Identified **weak intents** (short/confusable): PAUSE 39%, LIGHT_OFF 40%,
  LIGHT_ON 41%, VOLUME_DOWN 23%, NEXT 53%, STOP 59%.
- Confirmed **deployment feasibility** on the RPi budget (1.93 M params,
  7.7 MB fp32 / ~0.5 MB int8, ~2 ms latency).

## Sep 29 — Made it genuinely deployable (torch-free, self-contained)

Found the model was a *training/benchmark* artifact, not a *deployable* one. Fixed
four blockers:

| Blocker | Fix |
|---|---|
| ONNX never exported | Exported `low30/best.pt` → `model.onnx` (fixed a bug: export used the wrong model class) |
| ONNX split into 2 files (`.onnx` + `.onnx.data`) | Inlined weights → **single 7.76 MB file** |
| Feature extraction needed `torchaudio` | Wrote `mel_numpy.py` — pure-NumPy log-mel, **no torch** |
| Decoder + grammar needed torch / the dataset manifest | Wrote torch-free `decode_rpi.py`; bundled static `grammar_commands.json` (93 commands) |

- Packaged a **self-contained `dist/` folder (7.4 MB total)** — no dataset, no
  manifest, no torch.
- **Validated the Pi-style path** (torch-free, no data dir): **97% command accuracy**
  on a 100-sample clean test, **~42 ms p50** infer+decode; 15/15 distinct intents and
  5/5 weak intents correct.
- Wrote `DEPLOY-RPI5.md` (step-by-step) and `deploy_rpi.sh` (one-command Mac→Pi).

---

## Next (open tasks)

- [ ] **RPi5 deployment + real-microphone demo** (Tasks 5–8: real-time, standalone,
      on-device, no LLM). Hardware is in hand; follow `grammar_asr/DEPLOY-RPI5.md`.
- [ ] Lift the **weak/confusable short intents** (PAUSE, LIGHT_OFF, VOLUME_DOWN, NEXT,
      STOP) — candidates: targeted augmentation, a slightly larger encoder, or
      per-intent reweighting.
- [ ] Optional: **mix in real GSC/SLURP audio** for a robustness comparison in the report.
- [ ] Confirm **dataset/TTS licenses** (Chatterbox, LibriSpeech/SilencioPH refs).
- [ ] Write up the final ME2 report.

## Known caveats

- Data is **synthetic** (Chatterbox TTS). Fine for a tiny-VCM baseline; real-audio
  mix recommended before final claims.
- NumPy mel (Pi) is functionally, not bit-, identical to the torch mel — validated
  by the 97% torch-free test.
