#!/usr/bin/env bash
# One-command ME2 reproduction (tests + shipped artifact check).
#
#   bash me2/reproduce.sh
#
# From repo root or from me2/. Optional:
#   REPRODUCE_SMOKE=1 bash me2/reproduce.sh   # also run a tiny train smoke
#   REPRODUCE_HF=1 bash me2/reproduce.sh      # also audit HF dataset (needs net)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "==> ME2 reproduce · $(pwd)"

if [[ ! -d .venv ]]; then
  echo "==> Creating .venv"
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Installing grammar_asr requirements"
python -m pip install -q --upgrade pip
python -m pip install -q -r grammar_asr/requirements.txt

echo "==> Unit tests / schema contract"
python -m pytest grammar_asr/tests -q

ART="artifacts/submission"
echo "==> Checking shipped artifacts under $ART"
need=(
  "$ART/intent_hf_v1/command.onnx"
  "$ART/intent_hf_v1/best.pt"
  "$ART/intent_hf_v1/summary.json"
  "$ART/intent_hf_v1/train.log"
  "$ART/intent_hf_v1/reject_gate_pi.json"
  "$ART/wake_kaki_v6/kaki.onnx"
  "$ART/wake_kaki_v6/best.pt"
  "$ART/baseline_ctc_hf_s0/summary.json"
)
missing=0
for f in "${need[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "  MISSING: $f"
    missing=1
  else
    echo "  ok  $f"
  fi
done
if [[ "$missing" -ne 0 ]]; then
  echo "ERROR: submission artifacts missing." >&2
  exit 1
fi

echo "==> ONNX load smoke (command + wake)"
python - <<'PY'
from pathlib import Path
import onnxruntime as ort

root = Path("artifacts/submission")
for rel in (
    "intent_hf_v1/command.onnx",
    "wake_kaki_v6/kaki.onnx",
):
    p = root / rel
    sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    print(f"  ok  {p} inputs={[i.name for i in sess.get_inputs()]}")
PY

if [[ "${REPRODUCE_HF:-0}" == "1" ]]; then
  echo "==> HF dataset audit"
  python -m grammar_asr.scripts.audit_dataset --hf
fi

if [[ "${REPRODUCE_SMOKE:-0}" == "1" ]]; then
  echo "==> Tiny train smoke (CPU)"
  python -m grammar_asr.train.train \
    --config grammar_asr/configs/smoke_local.yaml \
    --smoke --device cpu
fi

echo
echo "OK — ME2 reproduce passed."
echo "Shipped weights:  $ROOT/$ART/"
echo "Pi benchmark:     bash grammar_asr/scripts/run_pi_benchmark.sh --help"
echo "Den UI (optional): python -m tamagotchi_demo.server   # then open http://127.0.0.1:8765"
