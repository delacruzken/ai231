#!/usr/bin/env bash
# 60s live mic debug — run in your own Terminal (macOS mic permission).
# Uses a process group so Ctrl+C / timeout actually stops the mic loop.
set -euo pipefail
cd "$(dirname "$0")/../.."
ME2="$(pwd)"
cd "$ME2"

PY="${ME2_PYTHON:-}"
if [[ -z "$PY" ]]; then
  for cand in .venv/bin/python ../.venv/bin/python; do
    if [[ -x "$cand" ]] && "$cand" -c 'import sounddevice' 2>/dev/null; then
      PY=$cand
      break
    fi
  done
  if [[ -z "${PY:-}" ]]; then
    if [[ -x .venv/bin/python ]]; then PY=.venv/bin/python
    elif [[ -x ../.venv/bin/python ]]; then PY=../.venv/bin/python
    else PY=python3
    fi
  fi
fi

mkdir -p "$HOME/vcm_benchmark"
STAMP=$(date +%Y%m%d-%H%M%S)
OUT="$HOME/vcm_benchmark/live_debug_${STAMP}.console.log"
JSONL_DIR="$HOME/vcm_benchmark"

# Kill any leftover assistants from earlier runs
pkill -f 'grammar_asr.runtime.assistant' 2>/dev/null || true

echo "Python: $PY"
"$PY" -c 'import sounddevice' 2>/dev/null || {
  echo "Installing sounddevice into this venv…"
  "$PY" -m pip install -q sounddevice
}
echo "Console log: $OUT"
echo "Speak 'kaki' then a command a few times. Auto-stops in 60s (Ctrl+C also stops)."
echo

ASSIST_PID=""
TEE_PID=""
cleanup() {
  # Prefer the assistant PID; fall back to pkill (setsid is Linux-only).
  if [[ -n "${ASSIST_PID:-}" ]]; then
    kill -INT "$ASSIST_PID" 2>/dev/null || true
    sleep 0.3
    kill -TERM "$ASSIST_PID" 2>/dev/null || true
    kill -KILL "$ASSIST_PID" 2>/dev/null || true
  fi
  if [[ -n "${TEE_PID:-}" ]]; then
    kill -TERM "$TEE_PID" 2>/dev/null || true
  fi
  pkill -f 'grammar_asr.runtime.assistant' 2>/dev/null || true
}
trap cleanup EXIT INT TERM

DEMO_ARGS=()
if [[ "${DEMO_UI:-}" == "1" || "${DEMO_UI:-}" == "true" ]]; then
  DEMO_ARGS+=(--demo-ui --demo-url "${DEMO_URL:-http://127.0.0.1:8765}")
  echo "Demo UI: ${DEMO_URL:-http://127.0.0.1:8765} (start tamagotchi_demo/server.py first)"
fi

set +e
# Portable launch (no setsid — missing on macOS). Track PIDs for cleanup.
: >"$OUT"
"$PY" -u -m grammar_asr.runtime.assistant \
  --model grammar_asr/runs/intent_hf_v1/command.onnx \
  --wake-model grammar_asr/runs/wake_kaki_v5/kaki.onnx \
  --reject-gate grammar_asr/runs/intent_hf_v1/reject_gate_live.json \
  --prefer intent_head \
  --wake-threshold 0.70 \
  --min-wake-rms 0.030 \
  --min-wake-peak-rms 0.08 \
  --max-wake-ms 1400 \
  --min-cmd-rms 0.040 \
  --wake-phrase kaki \
  --mic --debug \
  --student-id live_debug \
  --log-dir "$JSONL_DIR" \
  "${DEMO_ARGS[@]}" \
  >"$OUT" 2>&1 &
ASSIST_PID=$!
# Mirror console while assistant runs
tail -n +1 -f "$OUT" &
TEE_PID=$!
# Wait up to 60s; Ctrl+C hits trap → cleanup
for _ in $(seq 1 60); do
  if ! kill -0 "$ASSIST_PID" 2>/dev/null; then break; fi
  sleep 1
done
cleanup
trap - EXIT INT TERM
wait "$ASSIST_PID" 2>/dev/null
wait "$TEE_PID" 2>/dev/null
set -e

echo
echo "======== 60s summary ========"
"$PY" - <<PY
from pathlib import Path
import collections
text = Path("$OUT").read_text(errors="replace")
c = collections.Counter()
for line in text.splitlines():
    if "[wake] HIT" in line: c["wake_HIT"] += 1
    elif "[wake] miss" in line: c["wake_miss"] += 1
    elif "skip energy" in line: c["wake_skip"] += 1
    elif "[cmd]" in line and "skip quiet" not in line: c["cmd"] += 1
    elif "skip quiet leftover" in line: c["cmd_skip_quiet"] += 1
    elif "listen timeout" in line: c["listen_timeout"] += 1
    elif "[vad]" in line: c["vad"] += 1
print(dict(c))
print("Log:", "$OUT")
PY
echo "If lines still appear after this, run: pkill -9 -f grammar_asr.runtime.assistant"
