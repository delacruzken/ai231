#!/usr/bin/env bash
# Launch a versioned ME2 training run on DGX inside tmux, with logs.
#
# Usage (from repo root, after SSH ControlMaster is up):
#   bash me2/grammar_asr/scripts/launch_experiment.sh me2/grammar_asr/configs/intent_hf_v1.yaml 2
#
# Args:
#   $1  config path (default: intent_hf_v1.yaml)
#   $2  CUDA device index (default: 2)
set -euo pipefail

CONFIG="${1:-me2/grammar_asr/configs/intent_hf_v1.yaml}"
GPU="${2:-2}"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
TAG="$(python3 - "$ROOT/$CONFIG" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["tag"])
PY
)"
SESSION="me2-${TAG}"
REMOTE_ROOT="/mnt/jfs_hpc/kbdelacruz/ai231"
LOG="$REMOTE_ROOT/me2/grammar_asr/runs/${TAG}.train.log"
EXIT_FILE="$REMOTE_ROOT/me2/grammar_asr/runs/${TAG}.exit"

echo "Syncing grammar_asr + config to DGX…"
rsync -az -e 'ssh -S '"$ROOT/.dgx-ssh.sock"' -o ControlMaster=no -o BatchMode=yes' \
  "$ROOT/me2/grammar_asr/" \
  "kbdelacruz@n002.ai.internal:$REMOTE_ROOT/me2/grammar_asr/"

ssh -S "$ROOT/.dgx-ssh.sock" -o ControlMaster=no -o BatchMode=yes kbdelacruz@n002.ai.internal \
  "set -e
   SESSION='$SESSION'
   ROOT='$REMOTE_ROOT'
   LOG='$LOG'
   EXIT_FILE='$EXIT_FILE'
   CONFIG='$CONFIG'
   GPU='$GPU'
   TAG='$TAG'
   tmux has-session -t \"\$SESSION\" 2>/dev/null && { echo \"tmux session \$SESSION already exists\"; exit 1; }
   mkdir -p \"\$ROOT/me2/grammar_asr/runs\"
   rm -f \"\$EXIT_FILE\"
   tmux new-session -d -s \"\$SESSION\" \"
     cd \$ROOT &&
     export CUDA_VISIBLE_DEVICES=\$GPU
     export ME2_WORK_ROOT=/mnt/jfs_hpc/kbdelacruz
     export ME2_HF_CACHE=/mnt/jfs_hpc/kbdelacruz/hf-cache
     export XDG_CACHE_HOME=/mnt/jfs_hpc/kbdelacruz/.cache
     export TRITON_CACHE_DIR=/mnt/jfs_hpc/kbdelacruz/.cache/triton
     set -o pipefail
     bash me2/grammar_asr/scripts/train_dgx.sh \$CONFIG 2>&1 | tee \$LOG
     echo \\\$? > \$EXIT_FILE
   \"
   echo STARTED session=\$SESSION tag=\$TAG gpu=\$GPU
   echo LOG=\$LOG
   sleep 2
   tail -n 30 \$LOG || true
"

echo
echo "Monitor:"
echo "  ssh -S $ROOT/.dgx-ssh.sock kbdelacruz@n002.ai.internal \"tail -f $LOG\""
echo "  python -m grammar_asr.scripts.list_experiments --rank   # after finish + rsync"
