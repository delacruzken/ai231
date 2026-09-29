# ME2 — Grammar-Constrained Spoken Command Recognition

Replicates Anthony Navarez's approach (AI231, 9/22): instead of one-shot
intent classification, **predict characters with a tiny CTC ASR model, then
decode against a grammar trie** so the system can only ever output one of the
19 known commands. PocketSphinx-style.

## Why this design

| Property | One-shot classifier | Grammar-ASR (this) |
|---|---|---|
| Model size | needs ~5-20M params for 19 classes | **276K params** (learns 27 chars, not 19 patterns) |
| Robustness | mis-hear → wrong class | trie prunes invalid paths |
| New commands | retrain | just add to the trie |
| Footprint (int8) | several MB | **< 1 MB** (matches Anthony's RPi claim) |

## Architecture

```
WAV (16 kHz)
  -> 40-band log-mel (25 ms / 10 ms hop)
  -> 1-D CNN x4 (stride 2)  ->  T/16 frames
  -> Linear -> 28 classes (26 letters + space + CTC blank)
  -> [decode]
       greedy CTC  ->  raw text  (baseline)
       trie beam   ->  one of 93 command strings  (real system)
```

**Grammar:** 93 unique command strings (13 fixed intents × 3 variants +
6 slot intents × 9 slot-values), built from `manifest.csv`. Digits are
spelled out ("10 seconds" → "ten seconds") so the ASR target and the trie
share one representation.

## Files

| File | Purpose |
|---|---|
| `grammar.py` | character alphabet, number→words, trie, manifest loader |
| `features.py` | WAV → log-mel, manifest, `OptionBDataset`, dataloaders |
| `asr_model.py` | `CTCEncoder` (1-D CNN), `ctc_loss`, `greedy_decode` |
| `train_asr.py` | train loop (AdamW + cosine), CER/WER eval, saves `best.pt` |
| `decode_grammar.py` | trie-constrained beam decoder + command-accuracy eval |
| `benchmark.py` | 3-tier benchmark (clean / noisy / latency) → JSON + MD |
| `export_onnx.py` | ONNX fp32 + int8 export for RPi deployment |
| `runs/` | checkpoints + benchmark outputs |

## Data

Mark Macalalad's **OptionB** dataset (Chatterbox TTS, 100 speakers):
- 17,986 active WAVs (8,993 clean + 8,993 noisy @ ~30 dB SNR), 16 kHz mono
- Split **by speaker** (no leakage): train s1–s80, val s81–s90, test s91–s100
- Cloned to `../data/AI231_src/MEX2/OptionB/`

## Quick start

```bash
cd ai231/me2
source .venv/bin/activate

# 1. train (MPS on Apple Silicon; ~60 s/epoch)
python -m grammar_asr.train_asr --epochs 30 --tag base30

# 2. end-to-end command accuracy (grammar decode)
python -m grammar_asr.decode_grammar \
    --ckpt grammar_asr/runs/base30/best.pt \
    --split test --condition clean --mode grammar --beam 32

# 3. full 3-tier benchmark
python -m grammar_asr.benchmark \
    --ckpt grammar_asr/runs/base30/best.pt --device mps

# 4. export for RPi
python -m grammar_asr.export_onnx --ckpt grammar_asr/runs/base30/best.pt --quantize
```

## Status

- [x] Dataset cloned + verified (17,986 WAVs, manifest intact)
- [x] Grammar trie (93 commands, digits spelled out)
- [x] Feature pipeline (40-band log-mel, per-utt normalized)
- [x] CTC encoder (3 variants: base 276K, res 1.56M, **low 1.93M**) + training loops
- [x] Grammar-constrained **forced-alignment** decoder (PocketSphinx-style)
- [x] 3-tier benchmark + ONNX export
- [x] **Final results** (see below)
- [ ] RPi 5 deployment + real-mic demo
- [ ] Optional: mix in real GSC/SLURP audio for robustness comparison

## Results (low30 model, 30 epochs, MPS)

| Metric | Value |
|---|---|
| **Clean command accuracy** | **80.76%** (726/899) |
| **Noisy command accuracy** | **85.43%** (768/899) |
| CER (clean) | 1.90% |
| WER (clean) | 5.26% |
| Latency p50 / p95 | 2.0 / 2.3 ms |
| Model size | 1.93M params (7.7 MB fp32, ~0.5 MB int8) |

Strong intents: TEMPERATURE 100%, MESSAGE 100%, BRIGHTNESS 100%, ALARM 97.7%,
TIMER 95.5%, LIST_REMINDERS 93.1%, CREATE_REMINDER 92.2%.
Weak intents (confusable short commands): PAUSE 39%, LIGHT_OFF 40%,
LIGHT_ON 41%, VOLUME_DOWN 23%, NEXT 53%, STOP 59%.

### Key design lessons (the hard-won ones)

1. **Temporal downsampling must be low.** A 16×-downsample CNN (T'≈25) makes
   long commands (>25 chars) *structurally impossible* under CTC (need T'≥L),
   so the decoder collapses to short words ("time"/"stop") → 0% accuracy.
   Using only a 2× stem (T'≈T/2≈98) fixes it → 90%+.
2. **Forced alignment beats naive beam search.** A beam that carries a
   blank-stay hypothesis compounds its score each frame, so short commands
   always win. Scoring each grammar word via CTC forced alignment + a small
   per-char reward (`char_reward=0.15`) is what makes long, specific commands
   win fairly.
3. **The ASR front-end must be good.** The 276K base model (CER 30%) gave 0%
   command accuracy — the grammar has nothing to lock onto. The 1.93M low
   model (CER 1.8%) gives 81%. More capacity in the encoder, not the decoder.

### Model comparison (test CER)

| Model | Params | Downsample | Test CER | Cmd acc |
|---|---|---|---|---|
| base (4-layer CNN) | 276K | 16× | 29.8% | 0% |
| res (4-block residual) | 1.56M | 16× | 10.8% | 0.5% |
| **low (5-block residual)** | **1.93M** | **2×** | **1.8%** | **80.8%** |

## Notes / caveats

- Data is **synthetic** (Chatterbox TTS). Defensible for a tiny-VCM baseline;
  consider mixing real GSC/SLURP for a robustness comparison in the report.
- Confirm **license** of Chatterbox + LibriSpeech/SilencioPH references.
- The 10-epoch model (CER 33%) gives 0% command accuracy because the grammar
  has nothing to lock onto — the two-stage design needs a decent ASR front-end.
  Longer training is the fix, not a bigger model.
