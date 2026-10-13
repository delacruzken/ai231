#!/usr/bin/env bash
# Record additional kaki wake clips without overwriting existing WAVs.
#
# Usage (from anywhere):
#   bash me2/scripts/record_kaki_wake.sh                 # positives, speaker=guest
#   bash me2/scripts/record_kaki_wake.sh alice 30        # 30 positives for alice
#   bash me2/scripts/record_kaki_wake.sh bob 20 negative # 20 hard-negatives for bob
#
# Files land in:
#   me2/data/wake/kaki/positive/<speaker>_NNN.wav
#   me2/data/wake/kaki/negative/<speaker>_neg_NNN.wav
# Existing files are never replaced; numbering continues from the highest index.

set -euo pipefail

SPEAKER="${1:-guest}"
COUNT="${2:-20}"
KIND="${3:-positive}"   # positive | negative
MIC="${MIC:-:0}"        # ffmpeg avfoundation "video:audio"; audio-only = :0
SECONDS_PER_CLIP="${SECONDS_PER_CLIP:-1.5}"

# sanitize speaker id for filenames
SPEAKER="$(echo "$SPEAKER" | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9_-')"
if [[ -z "$SPEAKER" ]]; then
  echo "Invalid speaker id" >&2
  exit 1
fi
if ! [[ "$COUNT" =~ ^[0-9]+$ ]] || [[ "$COUNT" -lt 1 ]]; then
  echo "COUNT must be a positive integer" >&2
  exit 1
fi
if [[ "$KIND" != "positive" && "$KIND" != "negative" ]]; then
  echo "KIND must be positive or negative" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$ROOT/data/wake/kaki/$KIND"
mkdir -p "$OUT_DIR"

if [[ "$KIND" == "positive" ]]; then
  PREFIX="${SPEAKER}"
  PROMPT="say KAKI"
else
  PREFIX="${SPEAKER}_neg"
  PROMPT="say a NON-wake word (e.g. cookie, coffee, khaki, boss, pause) or make noise"
fi

# Next free index for this prefix (never overwrite).
next_index() {
  local max=0
  local f base num
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

START="$(next_index)"
END=$((START + COUNT - 1))
echo "Speaker : $SPEAKER"
echo "Kind    : $KIND"
echo "Output  : $OUT_DIR"
echo "Clips   : ${PREFIX}_$(printf '%03d' "$START").wav .. ${PREFIX}_$(printf '%03d' "$END").wav"
echo "Mic     : $MIC  (${SECONDS_PER_CLIP}s, 16 kHz mono)"
echo
echo "Existing files will NOT be replaced."
echo "Press Enter for each take (Ctrl+C to stop early)."
echo

for ((i = START; i <= END; i++)); do
  file="$(printf '%s/%s_%03d.wav' "$OUT_DIR" "$PREFIX" "$i")"
  if [[ -e "$file" ]]; then
    echo "Refusing to overwrite: $file" >&2
    exit 1
  fi
  n=$((i - START + 1))
  echo "[$n/$COUNT] Enter then ${PROMPT}"
  read -r
  ffmpeg -y -hide_banner -loglevel error \
    -f avfoundation -i "$MIC" \
    -t "$SECONDS_PER_CLIP" -ac 1 -ar 16000 -c:a pcm_s16le \
    "$file"
  echo "  saved $(basename "$file")"
done

echo
echo "Done. Positives: $(ls "$ROOT/data/wake/kaki/positive"/*.wav 2>/dev/null | wc -l | tr -d ' ')"
echo "       Negatives: $(ls "$ROOT/data/wake/kaki/negative"/*.wav 2>/dev/null | wc -l | tr -d ' ')"
echo
echo "Retrain later with BOTH speaker dirs still under kaki/positive and kaki/negative."
