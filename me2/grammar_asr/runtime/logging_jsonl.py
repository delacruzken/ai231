"""Benchmark-compatible JSONL logger for vcm-benchmark."""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from typing import Optional


class BenchmarkLogger:
    def __init__(self, student_id: str = "me2",
                 log_dir: Optional[str] = None):
        log_dir = log_dir or os.path.expanduser("~/vcm_benchmark")
        self.dir = Path(log_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.path = self.dir / f"{student_id}_{stamp}.log"
        self._fh = open(self.path, "a", buffering=1, encoding="utf-8")

    def log(self, record: dict) -> None:
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def wake(self, score: float = 1.0) -> None:
        self.log({"event": "wake", "score": round(float(score), 4)})

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass
