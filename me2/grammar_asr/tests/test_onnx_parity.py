"""ONNX export parity for the intent-CRNN smoke architecture."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from grammar_asr.export.export_onnx import export_onnx
from grammar_asr.models import build_model


def test_intent_onnx_parity(tmp_path: Path):
    model = build_model("intent", gru_hidden=32, gru_layers=1, dropout=0.0)
    model.eval()
    ckpt = tmp_path / "m.pt"
    onnx_path = tmp_path / "m.onnx"
    torch.save({"model": model.state_dict()}, ckpt)
    export_onnx(
            ckpt, onnx_path, "intent",
            {"gru_hidden": 32, "gru_layers": 1, "dropout": 0.0},
            example_T=100,
    )
    x = torch.randn(1, 100, 40)
    with torch.no_grad():
        ref = model(x)
        ref_intent = ref["intent_logits"].numpy()
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    outs = sess.run(None, {"feat": x.numpy().astype(np.float32)})
    assert np.allclose(outs[0], ref_intent, atol=1e-4)
