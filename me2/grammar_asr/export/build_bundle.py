"""Generate a self-contained Pi deploy bundle from canonical package sources.

The bundle is a flat directory meant to run as::

    cd ~/me2 && source .venv/bin/activate
    python assistant.py --model model.onnx --wake-model wake.onnx --mic
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
BUILD_DIR = PACKAGE / "build"


def _rewrite_flat(text: str, *, kind: str) -> str:
    """Rewrite package-relative imports for the flat Pi layout."""
    if kind == "assistant":
        text = text.replace(
            "from ..data.features import SAMPLE_RATE, normalize_feat, log_mel",
            "from mel_numpy import SAMPLE_RATE, normalize_feat, log_mel",
        )
        text = text.replace(
            "from ..decode.grammar_trie import build_grammar",
            "from decode.grammar_trie import build_grammar",
        )
        text = text.replace(
            "from ..decode.semantic import RejectGate, decode_model_output, SemanticPrediction",
            "from decode.semantic import RejectGate, decode_model_output, SemanticPrediction",
        )
        text = text.replace(
            "from ..schema import OUT_OF_SCOPE",
            "from schema import OUT_OF_SCOPE",
        )
        text = text.replace("from .actions import", "from actions import")
        text = text.replace("from .logging_jsonl import", "from logging_jsonl import")
        text = text.replace("from .vad import", "from vad import")
        text = text.replace("from .console import", "from console import")
    elif kind == "actions":
        text = text.replace("from .console import", "from console import")
    elif kind == "decode":
        text = text.replace("from ..schema import", "from schema import")
        text = text.replace("from ..text import", "from text import")
    elif kind == "text":
        text = text.replace("from .schema import", "from schema import")
    return text


def _copy_text(src: Path, dst: Path, *, kind: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(_rewrite_flat(src.read_text(), kind=kind))


def build_bundle(model_onnx: Path, reject_gate: Path | None,
                 out_dir: Path | None = None, wake_onnx: Path | None = None,
                 summary: Path | None = None) -> Path:
    out_dir = out_dir or (BUILD_DIR / "pi_bundle")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # Flat runtime modules (torch-free mel via mel_numpy)
    _copy_text(PACKAGE / "runtime" / "assistant.py", out_dir / "assistant.py",
               kind="assistant")
    _copy_text(PACKAGE / "runtime" / "actions.py", out_dir / "actions.py",
               kind="actions")
    for name in ("vad.py", "logging_jsonl.py", "console.py"):
        shutil.copy2(PACKAGE / "runtime" / name, out_dir / name)
    shutil.copy2(PACKAGE / "mel_numpy.py", out_dir / "mel_numpy.py")
    _copy_text(PACKAGE / "text.py", out_dir / "text.py", kind="text")
    shutil.copy2(PACKAGE / "requirements-rpi.txt", out_dir / "requirements-rpi.txt")

    # schema + decode packages with flat sibling imports
    shutil.copytree(
        PACKAGE / "schema", out_dir / "schema",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    decode_dst = out_dir / "decode"
    decode_dst.mkdir()
    for src in (PACKAGE / "decode").glob("*.py"):
        if src.name == "__init__.py":
            # Keep lightweight so `import decode.X` does not pull the world.
            (decode_dst / "__init__.py").write_text(
                '"""Pi decode package (flat sibling imports)."""\n'
            )
            continue
        _copy_text(src, decode_dst / src.name, kind="decode")

    shutil.copy2(model_onnx, out_dir / "model.onnx")
    if wake_onnx and wake_onnx.exists():
        shutil.copy2(wake_onnx, out_dir / "wake.onnx")
    if reject_gate and reject_gate.exists():
        shutil.copy2(reject_gate, out_dir / "reject_gate.json")
    if summary and summary.exists():
        shutil.copy2(summary, out_dir / "summary.json")

    (out_dir / "run_assistant.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "DIR=$(cd \"$(dirname \"$0\")\" && pwd)\n"
        "cd \"$DIR\"\n"
        "source \"$DIR/.venv/bin/activate\" 2>/dev/null || true\n"
        "export PYTHONPATH=\"$DIR${PYTHONPATH:+:$PYTHONPATH}\"\n"
        "exec python \"$DIR/assistant.py\" \\\n"
        "  --model \"$DIR/model.onnx\" \\\n"
        "  --wake-model \"$DIR/wake.onnx\" \\\n"
        "  --reject-gate \"$DIR/reject_gate.json\" \\\n"
        "  --prefer intent_head \\\n"
        "  --wake-threshold 0.70 \\\n"
        "  --mic --debug \"$@\"\n"
    )
    (out_dir / "run_assistant.sh").chmod(0o755)

    # Quick import smoke (optional; skip if deps missing)
    meta = {
        "model_onnx": "model.onnx",
        "wake_onnx": "wake.onnx" if wake_onnx and wake_onnx.exists() else None,
        "reject_gate": "reject_gate.json" if reject_gate and Path(reject_gate).exists() else None,
        "bytes": {
            "model.onnx": (out_dir / "model.onnx").stat().st_size,
        },
        "run": "python assistant.py --model model.onnx --wake-model wake.onnx "
               "--reject-gate reject_gate.json --prefer intent_head --mic --debug",
    }
    if (out_dir / "wake.onnx").exists():
        meta["bytes"]["wake.onnx"] = (out_dir / "wake.onnx").stat().st_size
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
