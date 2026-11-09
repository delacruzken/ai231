"""Stage marks for fdbench (docs/STAGES.md level 2).

Loads fd-benchmark's stdlib `fdbench_marks` when available; otherwise a no-op
local sink so production paths stay safe.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("me3.benchmark.marks")

_mark: Callable[..., None] | None = None
_describe: Callable[[dict[str, Any]], None] | None = None


def _load() -> None:
    global _mark, _describe
    if _mark is not None:
        return
    try:
        from me3.config import ROOT, load_config, resolve_path

        cfg = load_config()
        adapters = resolve_path(cfg.paths.fd_benchmark_dir, root=ROOT) / "adapters"
        if adapters.is_dir() and str(adapters) not in sys.path:
            sys.path.insert(0, str(adapters))
        from fdbench_marks import mark as _m  # type: ignore

        _mark = _m
        try:
            from fdbench_marks import describe as _d  # type: ignore

            _describe = _d
        except Exception:
            _describe = None
        return
    except Exception:
        logger.debug("fdbench_marks unavailable; marks are no-ops", exc_info=True)

    def _noop(*_a: Any, **_k: Any) -> None:
        return None

    _mark = _noop
    _describe = None


def mark(stage: str, **fields: Any) -> None:
    _load()
    assert _mark is not None
    try:
        _mark(stage, **fields)
    except Exception:
        logger.debug("mark(%s) failed", stage, exc_info=True)


def describe_models(models: dict[str, Any]) -> None:
    _load()
    if _describe is None:
        # Fall back: send a mark-compatible describe via mark sink if present
        try:
            import json
            import os
            import socket
            from urllib.parse import urlparse

            url = os.environ.get("FDBENCH_MARKS", "").strip()
            if not url:
                return
            u = urlparse(url)
            if u.scheme != "udp" or not u.hostname:
                return
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            payload = json.dumps({"event": "describe", "models": models}).encode()
            sock.sendto(payload, (u.hostname, u.port or 8792))
        except Exception:
            return
        return
    try:
        _describe(models)
    except Exception:
        logger.debug("describe failed", exc_info=True)


def me3_model_card() -> dict[str, str]:
    from me3.config import load_config

    cfg = load_config()
    return {
        "ASR": cfg.models.asr,
        "LLM": cfg.models.llm,
        "TTS": cfg.models.tts,
        "VAD": cfg.models.vad,
        "harness": "me3.benchmark",
    }
