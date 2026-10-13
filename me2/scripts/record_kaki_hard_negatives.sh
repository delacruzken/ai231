#!/usr/bin/env bash
# Record curated HARD negatives for kaki wake (never overwrites existing files).
#
# Usage:
#   bash me2/scripts/record_kaki_hard_negatives.sh              # speaker=hdc, full list ×1
#   bash me2/scripts/record_kaki_hard_negatives.sh hdc 2        # say each word twice
#   bash me2/scripts/record_kaki_hard_negatives.sh alice 1
#
# Words rotate from a fixed hard-negative list (good, cookie, khaki, …).
# Output: me2/data/wake/kaki/negative/<speaker>_hard_NNN.wav

set -euo pipefail

SPEAKER="${1:-hdc}"
REPEATS="${2:-1}"
MIC="${MIC:-:0}"
SECONDS_PER_CLIP="${SECONDS_PER_CLIP:-1.5}"

SPEAKER="$(echo "$SPEAKER" | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9_-')"
if [[ -z "$SPEAKER" ]]; then
  echo "Invalid speaker id" >&2
  exit 1
fi
if ! [[ "$REPEATS" =~ ^[0-9]+$ ]] || [[ "$REPEATS" -lt 1 ]]; then
  echo "REPEATS must be a positive integer" >&2
  exit 1
fi

# Priority false-accept confusers for "kaki"
WORDS=(
  hi
  hello
  hey
  good
  cookie
  coffee
  khaki
  cocky
  okay
  ok
  rocky
  lucky
  hockey
  copy
  packing
  talking
  boss
  pause
  because
  thankyou
  yeah
  what
  yes
  no
)

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$ROOT/data/wake/kaki/negative"
mkdir -p "$OUT_DIR"
PREFIX="${SPEAKER}_hard"

next_index() {
  local max=0 f base num
  shopt -s nullglob
  for f in "$OUT_DIR"/${PREFIX}_*.wav; do
    base="$(basename "$f" .wav)"
    num="${base##*_}"
    if [[ "$num" =~ ^[0-9]+$ ]] && (( 10#$num > max )); then
      max=$((10#$num))
    fi
  done
  echo $((max + 1))
}

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found. Install with: brew install ffmpeg" >&2
  exit 1
fi

TOTAL=$(( ${#WORDS[@]} * REPEATS ))
IDX="$(next_index)"
echo "Speaker : $SPEAKER"
echo "Output  : $OUT_DIR/${PREFIX}_*.wav"
echo "Takes   : $TOTAL  (${#WORDS[@]} words × $REPEATS)"
echo "Mic     : $MIC (${SECONDS_PER_CLIP}s, 16 kHz mono)"
echo
echo "Existing negatives are kept. Press Enter for each word (Ctrl+C to stop)."
echo

n=0
for ((r = 1; r <= REPEATS; r++)); do
  for word in "${WORDS[@]}"; do
    n=$((n + 1))
    file="$(printf '%s/%s_%03d.wav' "$OUT_DIR" "$PREFIX" "$IDX")"
    if [[ -e "$file" ]]; then
      echo "Refusing to overwrite: $file" >&2
      exit 1
    fi
    echo "[$n/$TOTAL] Enter, then say:  >>>  $word  <<<"
    read -r
    ffmpeg -y -hide_banner -loglevel error \
      -f avfoundation -i "$MIC" \
      -t "$SECONDS_PER_CLIP" -ac 1 -ar 16000 -c:a pcm_s16le \
      "$file"
    # sidecar label for auditing (optional)
    echo "$word" > "${file%.wav}.txt"
    echo "  saved $(basename "$file")  ($word)"
    IDX=$((IDX + 1))
  done
done

echo
echo "Hard negatives added. Total kaki/negative: $(ls "$OUT_DIR"/*.wav 2>/dev/null | wc -l | tr -d ' ')"
echo "Tell the agent when done to retrain wake_kaki_v3."
