#!/usr/bin/env bash
# Download / vendor pinned speech models into voice-agent/models/
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS="$ROOT/models"
TMP="$MODELS/.tmp"
FD_BENCH_SILERO="$ROOT/../fd-benchmark/models/silero_vad.onnx"
FD_BENCH_LICENSE="$ROOT/../fd-benchmark/models/SILERO_LICENSE"

# Conversational streaming Zipformer2 (Kroko) — default ASR
ASR_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06.tar.bz2"
ASR_DIR="sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06"
TTS_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-lessac-medium.tar.bz2"
TTS_DIR="vits-piper-en_US-lessac-medium"

mkdir -p "$MODELS" "$TMP"
cd "$MODELS"

fetch_archive() {
  local url="$1" name="$2" dir="$3"
  if [[ -d "$MODELS/$dir" ]]; then
    echo "[ok] $dir already present"
    return 0
  fi
  echo "[fetch] $name from $url"
  local archive="$TMP/$name"
  curl -L --fail --retry 3 -o "$archive" "$url"
  tar -xjf "$archive" -C "$MODELS"
  rm -f "$archive"
  echo "[ok] extracted $dir"
}

fetch_archive "$ASR_URL" "asr.tar.bz2" "$ASR_DIR"
fetch_archive "$TTS_URL" "tts.tar.bz2" "$TTS_DIR"

if [[ ! -f "$MODELS/silero_vad.onnx" ]]; then
  if [[ -f "$FD_BENCH_SILERO" ]]; then
    echo "[copy] silero_vad.onnx from fd-benchmark"
    cp "$FD_BENCH_SILERO" "$MODELS/silero_vad.onnx"
    [[ -f "$FD_BENCH_LICENSE" ]] && cp "$FD_BENCH_LICENSE" "$MODELS/SILERO_LICENSE"
  else
    echo "[fetch] silero_vad.onnx from sherpa-onnx release"
    curl -L --fail --retry 3 -o "$MODELS/silero_vad.onnx" \
      "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad_v5.onnx"
  fi
else
  echo "[ok] silero_vad.onnx already present"
fi

pick_component() {
  # Prefer int8 variant when present; else plain <stem>.onnx / epoch-named files.
  local dir="$1" stem="$2"
  local c
  for c in \
    "$dir/${stem}.int8.onnx" \
    "$dir/${stem}.onnx" \
    "$dir/${stem}-epoch-99-avg-1.int8.onnx" \
    "$dir/${stem}-epoch-99-avg-1.onnx"
  do
    if [[ -f "$MODELS/$c" ]]; then
      echo "$c"
      return 0
    fi
  done
  echo "ERROR: missing $stem onnx under $dir" >&2
  return 1
}

ENCODER="$(pick_component "$ASR_DIR" encoder)"
DECODER="$(pick_component "$ASR_DIR" decoder)"
JOINER="$(pick_component "$ASR_DIR" joiner)"
TOKENS="$ASR_DIR/tokens.txt"
if [[ ! -f "$MODELS/$TOKENS" ]]; then
  echo "ERROR: missing $TOKENS" >&2
  exit 1
fi

TTS_ONNX_PATH="$(find "$MODELS/$TTS_DIR" -name 'en_US-lessac-medium.onnx' | head -n1 || true)"
if [[ -z "$TTS_ONNX_PATH" || ! -f "$TTS_ONNX_PATH" ]]; then
  echo "ERROR: Piper onnx not found under $TTS_DIR" >&2
  exit 1
fi
TTS_ONNX="${TTS_ONNX_PATH#$MODELS/}"

export ME3_MODELS="$MODELS"
export ME3_ASR_DIR="$ASR_DIR"
export ME3_ENCODER="$ENCODER"
export ME3_DECODER="$DECODER"
export ME3_JOINER="$JOINER"
export ME3_TOKENS="$TOKENS"
export ME3_TTS_ONNX="$TTS_ONNX"
export ME3_ASR_URL="$ASR_URL"
export ME3_TTS_URL="$TTS_URL"

python3 <<'PY'
import hashlib, json, os, pathlib, time

root = pathlib.Path(os.environ["ME3_MODELS"])
asr_dir = os.environ["ME3_ASR_DIR"]
encoder = os.environ["ME3_ENCODER"]
decoder = os.environ["ME3_DECODER"]
joiner = os.environ["ME3_JOINER"]
tokens = os.environ["ME3_TOKENS"]
tts_onnx = os.environ["ME3_TTS_ONNX"]
rels = [encoder, decoder, joiner, tokens, tts_onnx, "silero_vad.onnx"]
entries = []
for rel in rels:
    p = root / rel
    if not p.is_file():
        raise SystemExit(f"missing {rel}")
    entries.append({
        "path": rel,
        "bytes": p.stat().st_size,
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
    })

data_dir = None
for cand in (root / "vits-piper-en_US-lessac-medium" / "espeak-ng-data",):
    if cand.is_dir():
        data_dir = str(cand.relative_to(root))
        break
if data_dir is None:
    found = list((root / "vits-piper-en_US-lessac-medium").rglob("espeak-ng-data"))
    if found:
        data_dir = str(found[0].relative_to(root))

tokens_tts = None
for cand in (root / "vits-piper-en_US-lessac-medium").rglob("tokens.txt"):
    tokens_tts = str(cand.relative_to(root))
    break

doc = {
    "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "asr": {
        "id": asr_dir,
        "url": os.environ["ME3_ASR_URL"],
        "encoder": encoder,
        "decoder": decoder,
        "joiner": joiner,
        "tokens": tokens,
        "note": "Kroko EN streaming Zipformer2; leave sherpa model_type empty for auto-detect",
    },
    "tts": {
        "id": "en_US-lessac-medium",
        "url": os.environ["ME3_TTS_URL"],
        "runtime": "piper-tts (PiperVoice; same ONNX weights)",
        "model": tts_onnx,
        "tokens": tokens_tts,
        "data_dir": data_dir,
        "sample_rate_hz": 22050,
        "note": "Prefer piper-tts for synthesis; sherpa-onnx>=2.0 deprecated vits data_dir",
    },
    "vad": {
        "id": "silero-vad-v5",
        "path": "silero_vad.onnx",
        "sample_rate_hz": 16000,
        "window_samples": 512,
    },
    "files": entries,
}
out = root / "MANIFEST.json"
out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
print(f"[ok] wrote {out}")
PY

rm -rf "$TMP"
echo "[done] models ready under $MODELS"
