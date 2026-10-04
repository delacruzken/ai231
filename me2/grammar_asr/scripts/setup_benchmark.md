# Official Pi benchmark (`vcm-benchmark`)

Use the class harness: https://github.com/airimonda/vcm-benchmark

## Before you start

1. Deploy the generated bundle from `grammar_asr/build/pi_bundle/` to the Pi.
2. On the Pi:
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements-rpi.txt
   mkdir -p ~/vcm_benchmark
   python assistant.py --model model.onnx \
     --reject-gate reject_gate.json \
     --wake-model wake.onnx \
     --mic --student-id YOUR_ID
   ```
3. Confirm JSONL lines contain `intent`, `slot` (when slotted), `infer_ms`, `audio_ms`.

## Laptop side

```bash
git clone https://github.com/airimonda/vcm-benchmark
cd vcm-benchmark
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python benchmark.py   # follow prompts: SSH, wake recordings, mic check
```

- Full run: ~196 holdout clips + 10 no-wake trials (~60 min).
- Quick run: 1 clip/variation + OOS + no-wake (~32 min).
- Collect `runs/<id>/report.md`, `metrics.json`, `trials.csv`, `pi_metrics.csv`,
  `pi_specs.json`, `config.json` into this repo under
  `me2/grammar_asr/runs/vcm_benchmark/<id>/` after the run.

## Notes

- Do not train or tune on holdout.
- Record Pi model (4/8 GB), mic device, and software versions honestly.
- Prefer live mic; standardized speaker playback is the accepted reproducible mode.
