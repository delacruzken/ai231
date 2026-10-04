#!/usr/bin/env bash
#SBATCH --job-name=me2-vcm
#SBATCH --partition=uperdfi-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=me2/grammar_asr/runs/slurm-%j.out
#SBATCH --error=me2/grammar_asr/runs/slurm-%j.err
#
# Usage (from repo root):
#   sbatch me2/grammar_asr/scripts/train_slurm.sh me2/grammar_asr/configs/kiwi_gold.yaml
#   CONFIG=me2/grammar_asr/configs/ctc_gold.yaml sbatch me2/grammar_asr/scripts/train_slurm.sh
set -euo pipefail

CONFIG="${1:-${CONFIG:-me2/grammar_asr/configs/kiwi_gold.yaml}}"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"

export ME2_DATA_ROOT="${ME2_DATA_ROOT:-/data/ai231}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# Prefer an existing project venv; otherwise create one under $HOME.
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
