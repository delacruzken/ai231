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
        text = text.replace("from .spotify import", "from spotify import")
    elif kind == "actions":
        text = text.replace("from .console import", "from console import")
        text = text.replace("from .spotify import", "from spotify import")
    elif kind == "spotify":
        pass  # ME2_ROOT auto-detects flat vs package layout
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
    # Stage into a sibling temp dir so rebuilding into an existing bundle
    # (which often holds the only local copy of the ONNX files) cannot
    # delete the source models mid-copy.
    stage = out_dir.parent / f".{out_dir.name}.staging"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    # Preserve existing ONNX / gate when rebuilding from the same folder.
    preserved: dict[str, Path] = {}
    if out_dir.exists():
        for name in ("model.onnx", "wake.onnx", "reject_gate.json", "summary.json"):
            src = out_dir / name
            if src.is_file():
                dest = stage.parent / f".preserve_{name}"
                shutil.copy2(src, dest)
                preserved[name] = dest
    out_dir_final = out_dir
    out_dir = stage

    # Flat runtime modules (torch-free mel via mel_numpy)
    _copy_text(PACKAGE / "runtime" / "assistant.py", out_dir / "assistant.py",
               kind="assistant")
    _copy_text(PACKAGE / "runtime" / "actions.py", out_dir / "actions.py",
               kind="actions")
    for name in ("vad.py", "logging_jsonl.py", "console.py"):
        shutil.copy2(PACKAGE / "runtime" / name, out_dir / name)
    _copy_text(PACKAGE / "runtime" / "spotify.py", out_dir / "spotify.py",
               kind="spotify")
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

    def _copy_artifact(src: Path | None, name: str, *, required: bool = False) -> bool:
        if src is not None and Path(src).is_file():
            shutil.copy2(src, out_dir / name)
            return True
        if name in preserved:
            shutil.copy2(preserved[name], out_dir / name)
            return True
        if required:
            raise FileNotFoundError(
                f"Missing {name}: pass --model/--wake/--reject-gate or rebuild "
                f"from a bundle that already contains it"
            )
        return False

    _copy_artifact(model_onnx, "model.onnx", required=True)
    _copy_artifact(wake_onnx, "wake.onnx")
    _copy_artifact(reject_gate, "reject_gate.json")
    _copy_artifact(summary, "summary.json")

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

    # Atomic-ish swap: move stage into place, then drop preserved copies.
    if out_dir_final.exists():
        shutil.rmtree(out_dir_final)
    stage.rename(out_dir_final)
    for p in preserved.values():
        p.unlink(missing_ok=True)
    print(f"Bundle ready: {out_dir_final}")
    return out_dir_final


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
