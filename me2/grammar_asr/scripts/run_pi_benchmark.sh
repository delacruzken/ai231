#!/usr/bin/env bash
# Official Pi latency / accuracy path (class vcm-benchmark).
#
# Prerequisites:
#   1. Pi has the bundle from artifacts (or grammar_asr/build/pi_bundle/)
#   2. Laptop can SSH to the Pi
#   3. Clone https://github.com/airimonda/vcm-benchmark beside this repo (or set VCM_BENCH)
#
# Usage:
#   bash me2/grammar_asr/scripts/run_pi_benchmark.sh --help
#   bash me2/grammar_asr/scripts/run_pi_benchmark.sh --prepare-pi
#   bash me2/grammar_asr/scripts/run_pi_benchmark.sh --collect
#
# After a full run, copy report.md / metrics.json / pi_metrics.csv into:
#   me2/artifacts/vcm_benchmark/<run_id>/
set -euo pipefail

ME2_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ART_CMD="$ME2_ROOT/artifacts/submission/intent_hf_v1"
ART_WAKE="$ME2_ROOT/artifacts/submission/wake_kaki_v6"
OUT_DIR="$ME2_ROOT/artifacts/vcm_benchmark"
VCM_BENCH="${VCM_BENCH:-$ME2_ROOT/../vcm-benchmark}"
PI_HOST="${PI_HOST:-}"
PI_DIR="${PI_DIR:-~/me2_pi_bundle}"

usage() {
  cat <<EOF
ME2 Pi benchmark helper

Env:
  PI_HOST     SSH target, e.g. kaki@192.168.1.20
  PI_DIR      Remote bundle dir (default: ~/me2_pi_bundle)
  VCM_BENCH   Path to cloned airimonda/vcm-benchmark
              (default: $ME2_ROOT/../vcm-benchmark)

Commands:
  --help         This text
  --prepare-pi   rsync shipped ONNX + reject gate to the Pi (needs PI_HOST)
  --collect      Copy latest vcm-benchmark run into artifacts/vcm_benchmark/
  (no flag)      Print the official steps (does not start the ~60 min run)

See also: grammar_asr/scripts/setup_benchmark.md
EOF
}

prepare_pi() {
  if [[ -z "$PI_HOST" ]]; then
    echo "Set PI_HOST=user@pi-address" >&2
    exit 1
  fi
  echo "==> Syncing shipped models to $PI_HOST:$PI_DIR"
  ssh "$PI_HOST" "mkdir -p $PI_DIR"
  rsync -avz \
    "$ART_CMD/command.onnx" \
    "$ART_CMD/reject_gate_pi.json" \
    "$ART_WAKE/kaki.onnx" \
    "$PI_HOST:$PI_DIR/"
  ssh "$PI_HOST" "cd $PI_DIR && mv -f reject_gate_pi.json reject_gate.json && mv -f kaki.onnx wake.onnx && ls -lh"
  cat <<EOF

On the Pi, run (from your deployed assistant bundle):

  python assistant.py --model command.onnx \\
    --reject-gate reject_gate.json \\
    --wake-model wake.onnx --mic --student-id YOUR_ID

Then on the laptop:

  cd "$VCM_BENCH"
  python benchmark.py   # SSH + holdout trials

When finished: bash $0 --collect
EOF
}

collect() {
  mkdir -p "$OUT_DIR"
  if [[ ! -d "$VCM_BENCH/runs" ]]; then
    echo "No runs under $VCM_BENCH/runs — clone/run vcm-benchmark first." >&2
    exit 1
  fi
  latest="$(ls -1dt "$VCM_BENCH"/runs/*/ 2>/dev/null | head -1 || true)"
  if [[ -z "$latest" ]]; then
    echo "No run folders found." >&2
    exit 1
  fi
  id="$(basename "$latest")"
  dest="$OUT_DIR/$id"
  mkdir -p "$dest"
  rsync -a "$latest" "$dest/"
  echo "==> Copied $latest → $dest"
  ls -la "$dest"
}

case "${1:-}" in
  --help|-h) usage ;;
  --prepare-pi) prepare_pi ;;
  --collect) collect ;;
  "")
    usage
    echo
    echo "Full write-up: $ME2_ROOT/grammar_asr/scripts/setup_benchmark.md"
    echo "Drop finished reports in: $OUT_DIR/<run_id>/"
    ;;
  *)
    echo "Unknown option: $1" >&2
    usage
    exit 1
    ;;
esac
