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
#
# Usage (from repo root, i.e. the dir containing me2/):
#   sbatch me2/grammar_asr/scripts/train_dgx.sh me2/grammar_asr/configs/kiwi_gold_dgx.yaml
#   CONFIG=me2/grammar_asr/configs/ctc_gold.yaml sbatch me2/grammar_asr/scripts/train_dgx.sh
#
# Env overrides:
#   ME2_DATA_ROOT  gold dataset root (default: <repo>/me2/data/gold)
#   ME2_VENV       project venv (default: $HOME/.venvs/me2)
set -euo pipefail

CONFIG="${1:-${CONFIG:-me2/grammar_asr/configs/kiwi_gold_dgx.yaml}}"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"

export ME2_DATA_ROOT="${ME2_DATA_ROOT:-$REPO_ROOT/me2/data/gold}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

VENV="${ME2_VENV:-$HOME/.venvs/me2}"
if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -U pip
  "$VENV/bin/pip" install -r me2/grammar_asr/requirements.txt
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

mkdir -p me2/grammar_asr/runs
echo "host=$(hostname) cuda=$CUDA_VISIBLE_DEVICES config=$CONFIG data=$ME2_DATA_ROOT"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY

python -m grammar_asr.scripts.audit_dataset --data-root "$ME2_DATA_ROOT" \
  --out "me2/grammar_asr/runs/dataset_audit_${SLURM_JOB_ID:-local}.json" || true

python -m grammar_asr.train.train --config "$CONFIG" --device cuda
echo "DONE $CONFIG"
