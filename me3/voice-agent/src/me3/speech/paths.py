"""Model path helpers and MANIFEST loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from me3.config import AppConfig, load_config, models_dir


def models_root(cfg: AppConfig | None = None) -> Path:
    return models_dir(cfg)


def load_manifest(cfg: AppConfig | None = None) -> dict[str, Any]:
    root = models_root(cfg)
    path = root / "MANIFEST.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run scripts/fetch_speech_models.sh first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_model(rel: str, cfg: AppConfig | None = None) -> Path:
    path = models_root(cfg) / rel
    if not path.is_file() and not path.is_dir():
        raise FileNotFoundError(f"Model artifact missing: {path}")
    return path


def asr_paths(cfg: AppConfig | None = None) -> dict[str, Path]:
    m = load_manifest(cfg)["asr"]
    return {
        "encoder": resolve_model(m["encoder"], cfg),
        "decoder": resolve_model(m["decoder"], cfg),
        "joiner": resolve_model(m["joiner"], cfg),
        "tokens": resolve_model(m["tokens"], cfg),
    }


def tts_paths(cfg: AppConfig | None = None) -> dict[str, Any]:
    m = load_manifest(cfg)["tts"]
    return {
        "model": resolve_model(m["model"], cfg),
        "sample_rate_hz": int(m.get("sample_rate_hz", 22050)),
    }


def vad_path(cfg: AppConfig | None = None) -> Path:
    m = load_manifest(cfg)["vad"]
    return resolve_model(m["path"], cfg)
