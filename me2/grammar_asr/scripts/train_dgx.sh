#!/usr/bin/env bash
#SBATCH --job-name=me2-vcm
#SBATCH --partition=uperdfi-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=me2/grammar_asr/runs/slurm-%j.out
#SBATCH --error=me2/grammar_asr/runs/slurm-%j.err
#
# DGX A100 launcher for the re-hauled ME2 (grammar_asr package).
# Runs either directly (the AI DGX has no active Slurm controller) or via sbatch.
#
# Usage (from repo root, i.e. the dir containing me2/):
#   CUDA_VISIBLE_DEVICES=2 bash me2/grammar_asr/scripts/train_dgx.sh \
#     me2/grammar_asr/configs/intent_gold_dgx.yaml
#   sbatch me2/grammar_asr/scripts/train_dgx.sh me2/grammar_asr/configs/intent_gold_dgx.yaml
#   CONFIG=me2/grammar_asr/configs/ctc_gold.yaml sbatch me2/grammar_asr/scripts/train_dgx.sh
#
# Env overrides:
#   ME2_DATA_ROOT  gold dataset root (default: <repo>/me2/data/gold)
#   ME2_WORK_ROOT  writable DGX storage (default: $HOME, else /mnt/jfs_hpc/$USER)
#   ME2_VENV       project venv (default: <work-root>/.venvs/me2)
#   ME2_HF_CACHE   writable Hugging Face datasets cache
set -euo pipefail

CONFIG="${1:-${CONFIG:-me2/grammar_asr/configs/intent_gold_dgx.yaml}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

if [[ "$CONFIG" = /* ]]; then
  CONFIG_PATH="$CONFIG"
elif [[ -f "$REPO_ROOT/$CONFIG" ]]; then
  CONFIG_PATH="$REPO_ROOT/$CONFIG"
elif [[ -f "$REPO_ROOT/me2/$CONFIG" ]]; then
  CONFIG_PATH="$REPO_ROOT/me2/$CONFIG"
else
  echo "Config not found: $CONFIG" >&2
  exit 2
fi

export ME2_DATA_ROOT="${ME2_DATA_ROOT:-$REPO_ROOT/me2/data/gold}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

if [[ -z "${ME2_WORK_ROOT:-}" ]]; then
  if [[ -d "$HOME" && -w "$HOME" ]]; then
    ME2_WORK_ROOT="$HOME"
  else
    ME2_WORK_ROOT="/mnt/jfs_hpc/$USER"
  fi
fi
mkdir -p "$ME2_WORK_ROOT"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ME2_WORK_ROOT/.cache}"
export HF_DATASETS_CACHE="${ME2_HF_CACHE:-$ME2_WORK_ROOT/hf-cache}"
export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$XDG_CACHE_HOME/triton}"
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$TRITON_CACHE_DIR"

VENV="${ME2_VENV:-$ME2_WORK_ROOT/.venvs/me2}"
if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -U pip
  "$VENV/bin/pip" install -r "$REPO_ROOT/me2/grammar_asr/requirements.txt"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

cd "$REPO_ROOT/me2"
mkdir -p grammar_asr/runs
echo "host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-all} config=$CONFIG_PATH data=$ME2_DATA_ROOT"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY

USE_HF="$(python - "$CONFIG_PATH" <<'PY'
import sys, yaml
with open(sys.argv[1], encoding="utf-8") as f:
    print("true" if yaml.safe_load(f).get("data", {}).get("use_hf") else "false")
PY
)"
AUDIT_ARGS=(--data-root "$ME2_DATA_ROOT")
if [[ "$USE_HF" == true ]]; then
  AUDIT_ARGS=(--hf)
fi
python -m grammar_asr.scripts.audit_dataset "${AUDIT_ARGS[@]}" \
  --out "grammar_asr/runs/dataset_audit_${SLURM_JOB_ID:-local}.json" || true

python -m grammar_asr.train.train --config "$CONFIG_PATH" --device cuda
echo "DONE $CONFIG_PATH"
