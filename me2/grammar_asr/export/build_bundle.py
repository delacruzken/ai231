"""Generate a self-contained Pi deploy bundle from canonical package sources."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
BUILD_DIR = PACKAGE / "build"


def build_bundle(model_onnx: Path, reject_gate: Path | None,
                 out_dir: Path | None = None, wake_onnx: Path | None = None,
                 summary: Path | None = None) -> Path:
    out_dir = out_dir or (BUILD_DIR / "pi_bundle")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # Runtime modules (canonical sources)
    runtime_files = [
        PACKAGE / "runtime" / "assistant.py",
        PACKAGE / "runtime" / "vad.py",
        PACKAGE / "runtime" / "actions.py",
        PACKAGE / "runtime" / "logging_jsonl.py",
        PACKAGE / "mel_numpy.py",
        PACKAGE / "requirements-rpi.txt",
    ]
    # Minimal decode/schema copies for torch-free path
    for src in runtime_files:
        if src.exists():
            shutil.copy2(src, out_dir / src.name)

    # Copy schema + decode as packages
    for name in ("schema", "decode", "text.py"):
        src = PACKAGE / name
        dst = out_dir / name
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
        elif src.exists():
            shutil.copy2(src, dst)

    shutil.copy2(model_onnx, out_dir / "model.onnx")
    if wake_onnx and wake_onnx.exists():
        shutil.copy2(wake_onnx, out_dir / "wake.onnx")
    if reject_gate and reject_gate.exists():
        shutil.copy2(reject_gate, out_dir / "reject_gate.json")
    if summary and summary.exists():
        shutil.copy2(summary, out_dir / "summary.json")

    # Flat launcher
    (out_dir / "run_assistant.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "DIR=$(cd \"$(dirname \"$0\")\" && pwd)\n"
        "source \"$DIR/.venv/bin/activate\" 2>/dev/null || true\n"
        "python \"$DIR/assistant.py\" --model \"$DIR/model.onnx\" \\\n"
        "  ${WAKE:+--wake-model \"$DIR/wake.onnx\"} \\\n"
        "  ${GATE:+--reject-gate \"$DIR/reject_gate.json\"} \\\n"
        "  --mic \"$@\"\n"
    )
    (out_dir / "run_assistant.sh").chmod(0o755)

    meta = {
        "model_onnx": "model.onnx",
        "wake_onnx": "wake.onnx" if wake_onnx else None,
        "reject_gate": "reject_gate.json" if reject_gate else None,
        "bytes": {
            "model.onnx": (out_dir / "model.onnx").stat().st_size,
        },
    }
    (out_dir / "bundle.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Bundle ready: {out_dir}")
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--reject-gate", default=None)
    ap.add_argument("--wake", default=None)
    ap.add_argument("--summary", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    build_bundle(
        Path(args.model),
        Path(args.reject_gate) if args.reject_gate else None,
        Path(args.out) if args.out else None,
        Path(args.wake) if args.wake else None,
        Path(args.summary) if args.summary else None,
    )


if __name__ == "__main__":
    main()
