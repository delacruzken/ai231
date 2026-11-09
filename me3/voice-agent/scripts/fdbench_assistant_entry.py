"""fd-benchmark assistant entry for ME3 (Phase 8).

Point the launcher at this file:

  export FDBENCH_PROJECT=/path/to/me3/voice-agent
  export FDBENCH_ENTRY=$FDBENCH_PROJECT/scripts/fdbench_assistant_entry.py
  $FDBENCH_PROJECT/.venv/bin/python /path/to/me3/fd-benchmark/launch_assistant.py --check

Or copy/symlink to fd-benchmark/assistant_entry.py (that path is gitignored upstream).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_VOICE = Path(__file__).resolve().parents[1]
_SRC = _VOICE / "src"
os.environ.setdefault("FDBENCH_PROJECT", str(_VOICE))
for p in (_SRC, _VOICE):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Identifies the voice-agent project folder for launch_assistant.find_project.
PROJECT_MARKER = "pyproject.toml"


def audio_rates() -> tuple[int, int]:
    # Input must be 16 kHz for fdbench; Piper native out is typically 22050 — we resample to 16 kHz in the bridge.
    return 16000, 16000


def describe() -> dict:
    from me3.benchmark.marks import me3_model_card

    return me3_model_card()


async def build(transport, log_dir: Path):
    from me3.benchmark.bridge import build_me3_benchmark

    return await build_me3_benchmark(transport, log_dir)
