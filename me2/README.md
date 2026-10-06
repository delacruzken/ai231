# ME2 — Voice Command Model (VCM)

Tiny, always-on, **on-device** voice commands for Raspberry Pi.  
No cloud, no LLM, no pretrained speech weights — train from scratch.

| | |
|---|---|
| **Dataset** | Course gold set on Hugging Face: [`ai231-me2-voice-commands`](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands) (DOI `10.57967/hf/10723`) |
| **Benchmark** | Class VCM harness: [`vcm-benchmark`](https://github.com/airimonda/vcm-benchmark) |
| **Labels** | 19 intents + `OUT_OF_SCOPE`, 93 variations, 6 slots × 3 values |
| **Hardware** | Train on DGX A100 · validate on Raspberry Pi 4/5 |
| **Status** | Rehauled for the final class contract (Oct 2026) |

## Architecture tracks

Three from-scratch models share the same data/eval contract:

| Track | Config | Role |
|---|---|---|
| **A. CTC baseline** | `configs/ctc_gold.yaml` | Low-downsample character CTC + grammar forced alignment (~1.9M) |
| **B. Intent CRNN** | `configs/intent_gold.yaml` | Depthwise-CRNN + intent/slot heads (~0.2–0.4M) |
| **C. Hybrid** | `configs/hybrid_gold.yaml` | Shared encoder with CTC + intent + slot + OOS |

Selection rule: maximize **real-speaker joint command accuracy** subject to a
false-accept ceiling and Pi size/latency budget. Synthetic-heavy overall
accuracy alone is not enough.

```
Mic → VAD → wake gate → command model → reject gate → JSONL log → mock action
```

## Quick start (dev)

```bash
cd me2
python3 -m venv .venv && source .venv/bin/activate
pip install -r grammar_asr/requirements.txt

# Unit tests / schema contract
pytest grammar_asr/tests -q

# Dataset audit (local gold or --hf)
python -m grammar_asr.scripts.audit_dataset --data-root /data/ai231
# python -m grammar_asr.scripts.audit_dataset --hf

# Smoke train (downloads HF if needed)
python -m grammar_asr.train.train --config grammar_asr/configs/smoke.yaml --smoke
```

## DGX A100 training

```bash
export ME2_DATA_ROOT=/data/ai231
export ME2_VENV=$HOME/.venvs/me2

# Interactive
python -m grammar_asr.train.train --config grammar_asr/configs/intent_gold.yaml --device cuda

# Slurm
sbatch grammar_asr/scripts/train_slurm.sh grammar_asr/configs/intent_gold.yaml
sbatch grammar_asr/scripts/train_slurm.sh grammar_asr/configs/ctc_gold.yaml
sbatch grammar_asr/scripts/train_slurm.sh grammar_asr/configs/hybrid_gold.yaml

# Optional Filipino-supplement ablation (document license caveats)
sbatch grammar_asr/scripts/train_slurm.sh grammar_asr/configs/intent_fil_ablation.yaml

# Compare runs
python -m grammar_asr.scripts.compare_runs grammar_asr/runs/*/summary.json \
  --max-far 0.10 --max-params 500000 --out grammar_asr/runs/selection.json
```

Each run writes under `grammar_asr/runs/<tag>/`:

- `config.resolved.json`, `history.json`, `best.pt`, `last.pt`
- `reject_gate.json`, `val_metrics.json`, `test_metrics.json`, `summary.json`

**Never train or early-stop on `holdout`.** Validation is a speaker-disjoint
fold carved from `train` only.

## Export + Pi bundle

```bash
python -m grammar_asr.export.export_onnx \
  --ckpt grammar_asr/runs/intent_gold_s1/best.pt \
  --model intent --out grammar_asr/runs/intent_gold_s1/model.onnx --quantize

python -m grammar_asr.export.build_bundle \
  --model grammar_asr/runs/intent_gold_s1/model.onnx \
  --reject-gate grammar_asr/runs/intent_gold_s1/reject_gate.json \
  --summary grammar_asr/runs/intent_gold_s1/summary.json
```

Copy `grammar_asr/build/pi_bundle/` to the Pi, then:

```bash
pip install -r requirements-rpi.txt
python assistant.py --model model.onnx --reject-gate reject_gate.json --mic
```

Logs go to `~/vcm_benchmark/*.log` with `intent`, `slot`, `infer_ms`, `audio_ms`.

Full official evaluation: see [`grammar_asr/scripts/setup_benchmark.md`](grammar_asr/scripts/setup_benchmark.md).

## Repository layout

```
me2/
├── README.md
├── grammar_asr/
│   ├── schema/           # variations.csv, licenses, canonical mappings
│   ├── data/             # gold loader, features, speaker-val split
│   ├── models/           # ctc_low, intent_crnn, hybrid, wake
│   ├── decode/           # grammar trie, forced align, reject gate
│   ├── train/            # trainer, losses, semantic evaluator
│   ├── runtime/          # Pi assistant (VAD, wake FSM, JSONL, actions)
│   ├── export/           # ONNX + generated Pi bundle
│   ├── configs/          # smoke + gold experiment YAMLs
│   ├── scripts/          # audit, slurm, compare, benchmark notes
│   ├── tests/
│   ├── runs/             # checkpoints + metrics (gitignored)
│   ├── build/            # generated bundles (gitignored)
│   └── HISTORY.md        # pre-rehaul Option B baseline
└── data/                 # local datasets (gitignored)
```

## Data policy

- Gold `train` / `test` / `holdout` membership is immutable.
- Optional train-only: `synthetic_negatives`, `supplemental_fil`, permitted
  `supplemental_synth`, `numerals`.
- `tune_oos` is for threshold calibration only — never training.
- Combined dataset is **research/education only**; see
  [`grammar_asr/schema/LICENSE-DATA.md`](grammar_asr/schema/LICENSE-DATA.md).

## Historical note

The older Option-B-only CTC stack (≈81% intent-only accuracy on synthetic test)
is summarized in [`grammar_asr/HISTORY.md`](grammar_asr/HISTORY.md).
