# ME2 submission pack

| Field | Value |
| --- | --- |
| **GitHub repository** | [`delacruzken/ai231`](https://github.com/delacruzken/ai231) · project folder [`me2/`](https://github.com/delacruzken/ai231/tree/main/me2) |
| **One-command reproduce** | `bash me2/reproduce.sh` |
| **Dataset** | [`airimonda/ai231-me2-voice-commands`](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands) · DOI [`10.57967/hf/10723`](https://doi.org/10.57967/hf/10723) · research/education (mixed upstream; see `grammar_asr/schema/LICENSE-DATA.md`) |
| **A100 cluster** | UP DGX · path `/mnt/jfs_hpc/kbdelacruz/ai231` · **1× A100** · CUDA · config `intent_hf_v1.yaml` · **seed 1** · epoch-sum wall **~11.6 min** (80 epochs, best @ 40) · wake `wake_kaki_v6` seed **5** (MPS) |
| **Model weights** | [`me2/artifacts/submission/`](https://github.com/delacruzken/ai231/tree/main/me2/artifacts/submission) — `intent_hf_v1/command.onnx` + `best.pt` + `train.log`; `wake_kaki_v6/kaki.onnx` + `best.pt` |

## Reviewer checklist

| # | Item | Status | Evidence |
| --- | --- | --- | --- |
| 1 | Repo public, one-command reproduction | **done** | Public repo · `bash me2/reproduce.sh` |
| 2 | Dataset licensed and citable (DOI) | **done** | HF + DOI above |
| 3 | Training logs + final checkpoint committed | **done** | `artifacts/submission/intent_hf_v1/{train.log,best.pt,summary.json}` |
| 4 | Pi 4 latency reproduced by the posted script | **partial** | Live `infer_ms` ~6 ms p50 / ~12 ms p90 (`artifacts/submission/latency_live.json`). Formal `vcm-benchmark` suite still open — script: `bash me2/grammar_asr/scripts/run_pi_benchmark.sh` |
| 5 | Held-out test set, unseen speakers | **done** | HF gold test (n=4443) · speaker-disjoint val · real subset n=824 |
| 6 | Baseline of comparable size compared | **done** | `ctc_hf_s0` (1.93M) vs `intent_hf_v1` (0.515M) on same HF gold — real cmd **23.2%** vs **70.1%** (`artifacts/submission/baseline_ctc_hf_s0/summary.json`) |

## Shipped metrics (intent_hf_v1)

- Test intent / command: **94.2% / 93.5%** (n=4443)
- Real speech intent / command: **72.7% / 70.1%** (n=824)
- Wake val (kaki v6): **98.5%** balanced · FAR 3.0% · thr 0.637

## Reproduce variants

```bash
bash me2/reproduce.sh                  # tests + artifact + ONNX load
REPRODUCE_HF=1 bash me2/reproduce.sh   # + HF dataset audit (network)
REPRODUCE_SMOKE=1 bash me2/reproduce.sh  # + tiny CPU train smoke
```
