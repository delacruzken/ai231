"""ONNX export parity for the KIWI smoke architecture."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

ort = pytest.importorskip("onnxruntime")

from grammar_asr.models import build_model
from grammar_asr.export.export_onnx import export_onnx


def test_kiwi_onnx_parity():
    model = build_model("kiwi", gru_hidden=32, gru_layers=1, dropout=0.0)
    model.eval()
    with tempfile.TemporaryDirectory() as td:
        ckpt = Path(td) / "m.pt"
        onnx_path = Path(td) / "m.onnx"
        torch.save({"model": model.state_dict()}, ckpt)
        export_onnx(
            ckpt, onnx_path, "kiwi",
            {"gru_hidden": 32, "gru_layers": 1, "dropout": 0.0},
            example_T=120,
        )
        x = torch.randn(1, 120, 40)
        with torch.no_grad():
            ref = model(x)["intent_logits"].numpy()
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        got = sess.run(None, {"feat": x.numpy().astype(np.float32)})[0]
        np.testing.assert_allclose(ref, got, rtol=1e-3, atol=1e-3)
