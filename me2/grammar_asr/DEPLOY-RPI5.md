# Deploy ME2 VCM on Raspberry Pi 5

## 1. Build the bundle on the training machine

```bash
python -m grammar_asr.export.export_onnx \
  --ckpt grammar_asr/runs/<tag>/best.pt \
  --model intent \
  --out grammar_asr/runs/<tag>/model.onnx

python -m grammar_asr.export.build_bundle \
  --model grammar_asr/runs/<tag>/model.onnx \
  --reject-gate grammar_asr/runs/<tag>/reject_gate.json \
  --summary grammar_asr/runs/<tag>/summary.json
```

## 2. Copy to the Pi

```bash
rsync -avz grammar_asr/build/pi_bundle/ pi@raspberrypi.local:~/me2/
```

## 3. Install runtime deps

```bash
sudo apt install -y python3-pip python3-venv portaudio19-dev libsndfile1
cd ~/me2
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-rpi.txt
mkdir -p ~/vcm_benchmark
```

## 4. Run

Run from `~/me2` with the venv active (the bundle is a **flat** layout; do not
use `python -m`). Live `[wake]` / `[cmd]` / `[action]` lines print in the SSH
terminal.

```bash
cd ~/me2 && source .venv/bin/activate

# WAV smoke
python assistant.py --model model.onnx --reject-gate reject_gate.json --wav cmd.wav

# Always-on mic (wake → command → sleep)
python assistant.py --model model.onnx --reject-gate reject_gate.json \
  --wake-model wake.onnx --prefer intent_head --wake-threshold 0.70 \
  --mic --debug --student-id YOUR_ID
```

Each decision appends one JSON line with `intent`, `slot`, `infer_ms`, `audio_ms`.

## 5. Official benchmark

Follow [`scripts/setup_benchmark.md`](scripts/setup_benchmark.md).
Copy resulting `runs/<id>/` artifacts back into
`me2/grammar_asr/runs/vcm_benchmark/<id>/`.
