#!/usr/bin/env python3
"""List / compare versioned ME2 training runs from the experiment ledger.

Examples:
  python -m grammar_asr.scripts.list_experiments
  python -m grammar_asr.scripts.list_experiments --markdown
  python -m grammar_asr.scripts.list_experiments --rank
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PACKAGE_ROOT / "runs"
LEDGER = RUNS_DIR / "ledger.jsonl"


def load_ledger() -> list[dict]:
    rows: list[dict] = []
    if LEDGER.exists():
        for line in LEDGER.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    # Also pick up any run dirs with summary.json not yet in ledger
    known = {r.get("tag") for r in rows}
    if RUNS_DIR.exists():
        for d in sorted(RUNS_DIR.iterdir()):
            if not d.is_dir():
                continue
            summary = d / "summary.json"
            if not summary.exists() or d.name in known:
                continue
            s = json.loads(summary.read_text())
            rows.append({
                "tag": d.name,
                "status": "done",
                "model": s.get("model"),
                "version": None,
                "experiment": None,
                "notes": "(imported from summary.json; not ledger-registered)",
                "params": s.get("params"),
                "best_val_command_acc": s.get("best_val_command_acc"),
                "test": s.get("test"),
                "val": s.get("val"),
                "run_dir": str(d),
                "imported": True,
            })
    return rows


def real_cmd(row: dict) -> float:
    test = row.get("test") or {}
    real = test.get("real") or {}
    return float(real.get("command_accuracy") or test.get("command_accuracy") or 0.0)


def fmt_pct(x) -> str:
    if x is None:
        return "-"
    return f"{100.0 * float(x):.1f}%"


def render_table(rows: list[dict], markdown: bool = False) -> str:
    headers = [
        "tag", "status", "model", "params",
        "val_cmd", "test_cmd", "test_real", "test_synth", "FAR", "notes",
    ]
    table = []
    for r in rows:
        test = r.get("test") or {}
        real = test.get("real") or {}
        synth = test.get("synthetic") or {}
        val = r.get("val") or {}
        table.append({
            "tag": r.get("tag"),
            "status": r.get("status"),
            "model": r.get("model"),
            "params": r.get("params"),
            "val_cmd": fmt_pct(val.get("command_accuracy") or r.get("best_val_command_acc")),
            "test_cmd": fmt_pct(test.get("command_accuracy")),
            "test_real": fmt_pct(real.get("command_accuracy")),
            "test_synth": fmt_pct(synth.get("command_accuracy")),
            "FAR": fmt_pct(test.get("false_accept_rate")),
            "notes": (r.get("notes") or "").strip().replace("\n", " ")[:60],
        })

    if markdown:
        lines = [
            "# ME2 experiment scoreboard",
            "",
            "Auto-refreshed from `runs/ledger.jsonl` (+ any `summary.json` not yet ledgered).",
            "",
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        for row in table:
            lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
        return "\n".join(lines) + "\n"

    widths = {h: max(len(h), max((len(str(r[h])) for r in table), default=0))
              for h in headers}
    lines = ["  ".join(h.ljust(widths[h]) for h in headers)]
    lines.append("  ".join("-" * widths[h] for h in headers))
    for row in table:
        lines.append("  ".join(str(row[h]).ljust(widths[h]) for h in headers))
    return "\n".join(lines) + "\n"


def write_scoreboard(runs_dir: Path | None = None) -> Path:
    runs_dir = runs_dir or RUNS_DIR
    rows = load_ledger()
    by_tag: dict[str, dict] = {}
    for r in rows:
        by_tag[r["tag"]] = r
    rows = sorted(by_tag.values(), key=real_cmd, reverse=True)
    out = runs_dir / "EXPERIMENTS.md"
    out.write_text(render_table(rows, markdown=True))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--rank", action="store_true",
                    help="Sort by test.real.command_accuracy desc")
    ap.add_argument("--out", type=str, default=None,
                    help="Also write table to this path")
    args = ap.parse_args(argv)

    rows = load_ledger()
    by_tag: dict[str, dict] = {}
    for r in rows:
        by_tag[r["tag"]] = r
    rows = list(by_tag.values())
    if args.rank:
        rows.sort(key=real_cmd, reverse=True)
    else:
        rows.sort(key=lambda r: r.get("started_at") or r.get("tag") or "")

    text = render_table(rows, markdown=bool(args.markdown))
    print(text, end="")
    out = Path(args.out) if args.out else None
    if out:
        if args.markdown and out.name == "EXPERIMENTS.md":
            out.write_text(text if text.startswith("#") else render_table(rows, True))
        else:
            out.write_text(text)
        print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
