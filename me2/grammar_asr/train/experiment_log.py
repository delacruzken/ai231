"""Experiment ledger helpers — append-only JSONL under runs/ledger.jsonl."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


def ledger_path(runs_dir: Path) -> Path:
    return runs_dir / "ledger.jsonl"


def append_ledger(runs_dir: Path, entry: Dict[str, Any]) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = ledger_path(runs_dir)
    row = dict(entry)
    row.setdefault("logged_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")
    # On finished command/wake runs, append HTML scoreboard version card.
    if row.get("status") == "done":
        try:
            from ..scripts.update_scoreboard import append_from_run_dir
            tag = row.get("tag")
            if tag:
                run_dir = Path(row.get("run_dir") or (runs_dir / tag))
                if run_dir.exists():
                    append_from_run_dir(
                        run_dir,
                        runs_dir=runs_dir,
                        version=row.get("version"),
                        experiment=row.get("experiment"),
                        notes=row.get("notes"),
                    )
        except Exception:
            pass
    return path


def write_notes(run_dir: Path, cfg: Dict[str, Any]) -> Path:
    notes = str(cfg.get("notes") or "").strip()
    text = (
        f"# {cfg.get('tag')}\n\n"
        f"- version: `{cfg.get('version')}`\n"
        f"- experiment: `{cfg.get('experiment')}`\n"
        f"- parent: `{cfg.get('parent')}`\n"
        f"- model: `{cfg.get('model', {}).get('name')}`\n"
        f"- seed: `{cfg.get('seed')}`\n\n"
        f"## Notes\n\n{notes or '_(none)_'}\n"
    )
    path = run_dir / "NOTES.md"
    path.write_text(text + "\n", encoding="utf-8")
    return path


def seed_baseline_ctc_entry(runs_dir: Path) -> Optional[Path]:
    """If ctc_hf_s0 exists and ledger is empty, register it as v0 baseline."""
    path = ledger_path(runs_dir)
    summary = runs_dir / "ctc_hf_s0" / "summary.json"
    if not summary.exists():
        return None
    if path.exists() and path.read_text().strip():
        return None
    s = json.loads(summary.read_text())
    append_ledger(runs_dir, {
        "tag": "ctc_hf_s0",
        "version": "v0",
        "experiment": "ctc_grammar_hf",
        "status": "done",
        "model": s.get("model"),
        "params": s.get("params"),
        "notes": "CTC + grammar decode baseline on HF gold (pre-intent-CRNN).",
        "parent": None,
        "best_val_command_acc": s.get("best_val_command_acc"),
        "val": s.get("val"),
        "test": s.get("test"),
        "reject_gate": s.get("reject_gate"),
        "run_dir": str(runs_dir / "ctc_hf_s0"),
        "finished_at": None,
    })
    return path
