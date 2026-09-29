"""
Export the trained CTC encoder to ONNX (fp32) and optionally int8.

The ONNX graph takes (1, T, 40) log-mel features and outputs (1, T', 28)
logits. For RPi deployment, the int8 variant (via onnxruntime quantization)
should land well under 1 MB, matching Anthony's footprint claim.

Usage:
  python -m grammar_asr.export_onnx --ckpt runs/base/best.pt --out model.onnx
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .asr_model import CTCEncoder
from .asr_model_res import ResCTCEncoder
from .asr_model_low import ResCTCEncoderLow
from .features import log_mel, normalize_feat, load_wav, SAMPLE_RATE


def _build(model_type: str):
    if model_type == "base":
        return CTCEncoder()
    if model_type == "res":
        return ResCTCEncoder()
    return ResCTCEncoderLow()


def export_onnx(ckpt: Path, out: Path, example_T: int = 195,
                model_type: str = "low"):
    model = _build(model_type)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    dummy = torch.randn(1, example_T, 40)
    with torch.no_grad():
        torch.onnx.export(
            model, dummy, out,
            input_names=["feat"], output_names=["logits"],
            dynamic_axes={
                "feat": {0: "batch", 1: "time"},
                "logits": {0: "batch", 1: "time"},
            },
            opset_version=13,
        )
    # Inline any external-data file so the .onnx is a single portable file.
    ext = Path(str(out) + ".data")
    if ext.exists():
        import onnx
        from onnx.external_data_helper import load_external_data_for_model
        m = onnx.load(str(out))
        load_external_data_for_model(m, str(out.parent))
        onnx.save(m, str(out))
        ext.unlink()
        print(f"Inlined external data; removed {ext.name}")
    size_mb = out.stat().st_size / 1e6
    print(f"Exported {out} ({size_mb:.3f} MB fp32, single file)")

    # verify with onnxruntime if available
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(out))
        x = np.random.randn(1, example_T, 40).astype(np.float32)
        y = sess.run(None, {"feat": x})[0]
        print(f"ONNX runtime check: in {x.shape} -> out {y.shape} OK")
    except ImportError:
        print("(onnxruntime not installed; skipped runtime check)")
    return out


def quantize_int8(onnx_path: Path, out: Path):
    """Dynamic int8 quantization (needs onnx + onnxruntime)."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QDQ
        from onnxruntime.quantization.registry import DEFAULT_QUANTIZATION
        quantize_dynamic(str(onnx_path), str(out),
                         weight_type=QDQ.QUInt8)
        size_mb = out.stat().st_size / 1e6
        print(f"Quantized {out} ({size_mb:.3f} MB int8)")
    except ImportError:
        print("(onnxruntime quantization unavailable; pip install onnxruntime)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out", type=str, default="model.onnx")
    ap.add_argument("--quantize", action="store_true")
    ap.add_argument("--model", type=str, default="low",
                    choices=["base", "res", "low"])
    args = ap.parse_args()
    out = export_onnx(Path(args.ckpt), Path(args.out), model_type=args.model)
    if args.quantize:
        q = Path(str(args.out).replace(".onnx", "_int8.onnx"))
        quantize_int8(out, q)
