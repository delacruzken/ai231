# ME2 lean cleanup inventory

Scoped to AI231 ME2 files under `me2/` only.

| Removed | Replacement / reason |
| --- | --- |
| `me2/ai231/`, `me2/ai222/` | Historical Telegram HTML exports; decisions captured in README + schema |
| `PROGRESS.md`, `KNOWLEDGE-TRANSFER.md`, `TEAM-PROGRESS.md`, `ME2-PLAN.md` | Superseded by current README / HISTORY |
| `Dataset Schema - Option B.pdf` | Canonical `schema/variations.csv` |
| Old CTC trainers/models (`asr_model*.py`, `train_asr*.py`) | `models/` + `train/train.py` |
| Old Option-B loader/decoder (`features.py`, `grammar.py`, `decode_grammar.py`, `decode_rpi.py`, `benchmark.py`, `infer_rpi.py`) | `data/`, `decode/`, `train/evaluate.py`, `runtime/assistant.py` |
| `export_onnx.py`, `deploy_rpi.sh`, `dist/` | `export/export_onnx.py`, `export/build_bundle.py` → `build/` |
| `debug_scores*.py`, `slurm_probe*.sh` | Replaced by tests + `scripts/train_slurm.sh` |

| Kept | Reason |
| --- | --- |
| `ME2 - Specifications.pdf`, `AI231 ME2 - Datasets.pdf` | Assignment citations |
| `mel_numpy.py` | Torch-free Pi features |
| `HISTORY.md` | Pre-rehaul Option B baseline metrics |
