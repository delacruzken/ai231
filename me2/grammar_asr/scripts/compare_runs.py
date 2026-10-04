#!/usr/bin/env python3
"""Compare experiment summaries and apply the Pareto selection rule.

Rule: maximize real-speaker joint command accuracy subject to
  false_accept_rate <= max_far and params <= max_params.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text())


def score(s: dict, max_far: float, max_params: int) -> tuple:
    test = s.get("test") or {}
    real = test.get("real") or {}
    far = float(test.get("false_accept_rate", 1.0))
    params = int(s.get("params") or 10**12)
    real_cmd = float(real.get("command_accuracy") or test.get("command_accuracy") or 0.0)
    feasible = far <= max_far and params <= max_params
    # Sort key: feasible first, then real command acc, then lower FAR, then fewer params
    return (1 if feasible else 0, real_cmd, -far, -params)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summaries", nargs="+", help="summary.json paths")
    ap.add_argument("--max-far", type=float, default=0.10)
    ap.add_argument("--max-params", type=int, default=500_000)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    rows = []
    for p in args.summaries:
        s = load_summary(Path(p))
        s["_path"] = p
        s["_score"] = score(s, args.max_far, args.max_params)
        rows.append(s)
    rows.sort(key=lambda r: r["_score"], reverse=True)
    report = {
        "rule": {
            "maximize": "test.real.command_accuracy",
            "max_far": args.max_far,
            "max_params": args.max_params,
        },
        "ranked": [
            {
                "path": r["_path"],
                "tag": r.get("tag"),
                "model": r.get("model"),
                "params": r.get("params"),
                "test": r.get("test"),
                "feasible": bool(r["_score"][0]),
            }
            for r in rows
        ],
        "selected": rows[0]["_path"] if rows else None,
    }
    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
