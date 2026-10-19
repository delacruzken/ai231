# ME2 submission artifacts

Tracked copies of shipped models / logs (the working `grammar_asr/runs/` tree stays gitignored).

| Path | Contents |
| --- | --- |
| `submission/intent_hf_v1/` | Command CRNN: `best.pt`, `command.onnx`, `command_int8.onnx`, `train.log`, `summary.json`, metrics, reject gates |
| `submission/wake_kaki_v6/` | Wake keyword spotter: `best.pt`, `kaki.onnx`, metrics |
| `submission/baseline_ctc_hf_s0/` | CTC baseline `summary.json` (comparable-task baseline) |
| `vcm_benchmark/` | Drop official Pi `vcm-benchmark` reports here after a run |

Public URLs (repo `delacruzken/ai231`, folder `me2/`):

- https://github.com/delacruzken/ai231/tree/main/me2/artifacts/submission
- Intent ONNX: `me2/artifacts/submission/intent_hf_v1/command.onnx`
- Wake ONNX: `me2/artifacts/submission/wake_kaki_v6/kaki.onnx`
