#!/usr/bin/env bash
# One-shot: copy dist/ to the Pi and bootstrap the Python env.
# Usage:  ./deploy_rpi.sh  pi@<PI_IP>
# The Pi must already be booted, on the same network, and you must be able to
# ssh into it (default user: pi). Set PI_PASS only if you use password auth.
set -euo pipefail

PI="${1:-pi@raspberrypi.local}"
SRC="$(cd "$(dirname "$0")" && pwd)/dist"

echo "==> Source: $SRC"
echo "==> Target: $PI:~/me2/"

# 1. Transfer the self-contained folder (no dataset, no torch)
rsync -avz --progress "$SRC/" "$PI:~/me2/"

# 2. Bootstrap the venv + deps on the Pi
ssh "$PI" bash -s <<'REMOTE'
set -euo pipefail
sudo apt-get update -y
sudo apt-get install -y python3-pip python3-venv portaudio19-dev libsndfile1
cd ~/me2
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-rpi.txt
echo "================ SETUP DONE ================"
echo "Smoke test:  source ~/me2/.venv/bin/activate"
echo "             python infer_rpi.py --model model.onnx --wav /path/to/cmd.wav"
echo "Live mic:    python infer_rpi.py --model model.onnx --mic --seconds 3"
REMOTE
