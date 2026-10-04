"""Export trained checkpoints to ONNX (fp32 + optional dynamic int8)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..models import build_model


def _load_ckpt(path: Path, model_name: str, model_kwargs: dict):
    model = build_model(model_name, **model_kwargs)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    model.load_state_dict(state)
    model.eval()
    return model


class _ExportWrapper(torch.nn.Module):
    """Flatten multi-head outputs to a stable ONNX signature."""

    def __init__(self, model, kind: str):
        super().__init__()
        self.model = model
        self.kind = kind

    def forward(self, feat):
        out = self.model(feat)
        if self.kind == "ctc":
            return out["ctc_logits"]
        if self.kind == "kiwi":
            # intent + concatenated slot logits in fixed slot-intent order
            from ..schema import SLOT_INTENTS
            slots = [out["slot_logits"][k] for k in SLOT_INTENTS]
            return out["intent_logits"], torch.cat(slots, dim=-1)
        if self.kind == "hybrid":
            from ..schema import SLOT_INTENTS
            slots = [out["slot_logits"][k] for k in SLOT_INTENTS]
            return out["ctc_logits"], out["intent_logits"], torch.cat(slots, dim=-1)
        if self.kind == "wake":
            return out["wake_logits"]
        raise ValueError(self.kind)


def export_onnx(ckpt: Path, out: Path, model_name: str,
                model_kwargs: dict | None = None, example_T: int = 300) -> Path:
    model_kwargs = model_kwargs or {}
    model = _load_ckpt(ckpt, model_name, model_kwargs)
    kind = model.model_kind
    wrapper = _ExportWrapper(model, kind)
    dummy = torch.randn(1, example_T, 40)

    if kind == "ctc":
        input_names, output_names = ["feat"], ["ctc_logits"]
        dynamic = {"feat": {0: "batch", 1: "time"},
                   "ctc_logits": {0: "batch", 1: "time"}}
    elif kind == "kiwi":
        input_names, output_names = ["feat"], ["intent_logits", "slot_logits"]
        dynamic = {"feat": {0: "batch", 1: "time"},
                   "intent_logits": {0: "batch"},
                   "slot_logits": {0: "batch"}}
    elif kind == "hybrid":
        input_names = ["feat"]
        output_names = ["ctc_logits", "intent_logits", "slot_logits"]
        dynamic = {"feat": {0: "batch", 1: "time"},
                   "ctc_logits": {0: "batch", 1: "time"},
                   "intent_logits": {0: "batch"},
                   "slot_logits": {0: "batch"}}
    else:
        input_names, output_names = ["feat"], ["wake_logits"]
        dynamic = {"feat": {0: "batch", 1: "time"},
                   "wake_logits": {0: "batch"}}

    out.parent.mkdir(parents=True, exist_ok=True)
    # Prefer the legacy exporter; the dynamo/onnx exporter is unstable for BiGRU.
    try:
        torch.onnx.export(
            wrapper, dummy, str(out),
            input_names=input_names, output_names=output_names,
            dynamic_axes=dynamic, opset_version=13,
            dynamo=False,
        )
    except TypeError:
        torch.onnx.export(
            wrapper, dummy, str(out),
            input_names=input_names, output_names=output_names,
            dynamic_axes=dynamic, opset_version=13,
        )
    # Inline external data if present
    ext = Path(str(out) + ".data")
    if ext.exists():
        import onnx
        from onnx.external_data_helper import load_external_data_for_model
        m = onnx.load(str(out))
        load_external_data_for_model(m, str(out.parent))
        onnx.save(m, str(out))
        ext.unlink()
    print(f"Exported {out} ({out.stat().st_size/1e6:.3f} MB)")

    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        x = np.random.randn(1, example_T, 40).astype(np.float32)
        ys = sess.run(None, {"feat": x})
        print(f"ORT check OK: {[y.shape for y in ys]}")
    except ImportError:
        print("(onnxruntime missing; skipped ORT check)")
    return out


def quantize_int8(onnx_path: Path, out: Path) -> Path:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(str(onnx_path), str(out), weight_type=QuantType.QInt8)
    print(f"Quantized {out} ({out.stat().st_size/1e6:.3f} MB)")
    return out


def _kwargs_from_run(ckpt: Path) -> dict:
    """Pull model kwargs from sibling config.resolved.json when present."""
    cfg_path = ckpt.parent / "config.resolved.json"
    if not cfg_path.exists():
        return {}
    cfg = json.loads(cfg_path.read_text())
    model = dict((cfg.get("config") or {}).get("model") or {})
    model.pop("name", None)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=None,
                    choices=["ctc", "ctc_low", "low", "kiwi", "hybrid", "wake"])
    ap.add_argument("--quantize", action="store_true")
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--gru-hidden", type=int, default=None)
    ap.add_argument("--gru-layers", type=int, default=None)
    ap.add_argument("--num-blocks", type=int, default=None)
    args = ap.parse_args()
    kwargs = _kwargs_from_run(Path(args.ckpt))
    model_name = args.model or kwargs.pop("name", None) or "kiwi"
    if args.hidden is not None:
        kwargs["hidden"] = args.hidden
    if args.gru_hidden is not None:
        kwargs["gru_hidden"] = args.gru_hidden
    if args.gru_layers is not None:
        kwargs["gru_layers"] = args.gru_layers
    if args.num_blocks is not None:
        kwargs["num_blocks"] = args.num_blocks
    # config.resolved stores name inside config.model
    cfg_path = Path(args.ckpt).parent / "config.resolved.json"
    if cfg_path.exists() and args.model is None:
        full = json.loads(cfg_path.read_text())
        model_name = ((full.get("config") or {}).get("model") or {}).get("name", model_name)
        kwargs = dict((full.get("config") or {}).get("model") or {})
        kwargs.pop("name", None)
    out = export_onnx(Path(args.ckpt), Path(args.out), model_name, kwargs)
    if args.quantize:
        quantize_int8(out, Path(str(args.out).replace(".onnx", "_int8.onnx")))


if __name__ == "__main__":
    main()
