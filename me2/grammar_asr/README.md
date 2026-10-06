# `grammar_asr` package

Implementation of the ME2 voice-command system. See the top-level
[`../README.md`](../README.md) for the full workflow.

| Module | Purpose |
|---|---|
| `schema/` | Canonical 93 variations + licensing notes |
| `data/` | Gold dataset loader (local / DGX / HF) |
| `models/` | CTC, intent CRNN, hybrid, wake |
| `decode/` | Grammar forced-align + reject gate |
| `train/` | Config-driven A100/CPU trainer + evaluator |
| `runtime/` | Pi assistant (VAD, wake, JSONL, actions) |
| `export/` | ONNX + generated Pi bundle |
| `configs/` | Experiment YAMLs |
| `scripts/` | Audit, Slurm, compare, benchmark notes |
| `tests/` | Contract / model / decode / ONNX tests |
| `HISTORY.md` | Pre-rehaul Option B baseline |

```bash
cd me2
pytest grammar_asr/tests -q
python -m grammar_asr.train.train --config grammar_asr/configs/smoke_local.yaml
```
