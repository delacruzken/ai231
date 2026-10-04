#!/usr/bin/env python3
"""Audit the official ME2 gold dataset (local or Hugging Face).

Records revision/fingerprint, split counts, speaker leakage, schema coverage,
and per-source summary. Never modifies the dataset.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from grammar_asr.data.dataset import (
        discover_dataset_root, load_split_samples, load_hf_split,
        dataset_fingerprint, _resolve_root,
    )
    from grammar_asr.schema import VARIATIONS, schema_fingerprint
except ImportError:  # pragma: no cover - script path execution
    ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(ROOT))
    from grammar_asr.data.dataset import (
        discover_dataset_root, load_split_samples, load_hf_split,
        dataset_fingerprint, _resolve_root,
    )
    from grammar_asr.schema import VARIATIONS, schema_fingerprint


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=str, default=None,
                    help="Local gold root (or set ME2_DATA_ROOT / AI231_DATA)")
    ap.add_argument("--hf", action="store_true",
                    help="Load from Hugging Face instead of local folders")
    ap.add_argument("--revision", type=str, default=None)
    ap.add_argument("--include-holdout", action="store_true",
                    help="Also fingerprint holdout (read-only; never train)")
    ap.add_argument("--out", type=str, default=None,
                    help="Write JSON report to this path")
    args = ap.parse_args()

    schema_fp = schema_fingerprint()
    report = {
        "schema": {
            "num_intents": schema_fp["num_intents"],
            "num_variations": schema_fp["num_variations"],
            "variations_csv_sha256": schema_fp["variations_csv_sha256"],
        },
        "source": {},
        "splits": {},
        "coverage": {},
        "ok": True,
        "warnings": [],
        "errors": [],
    }

    splits = {}
    root = None
    if args.hf:
        report["source"] = {
            "mode": "huggingface",
            "dataset": "airimonda/ai231-me2-voice-commands",
            "revision": args.revision,
        }
        for split in ("train", "test"):
            splits[split] = load_hf_split(split, revision=args.revision)
        if args.include_holdout:
            splits["holdout"] = load_hf_split("holdout", revision=args.revision)
    else:
        root = discover_dataset_root(args.data_root)
        if root is None:
            report["ok"] = False
            report["errors"].append(
                "No local gold dataset found. Pass --data-root, set "
                "ME2_DATA_ROOT, place data under me2/data/gold, use DGX "
                "/data/ai231, or pass --hf."
            )
            _emit(report, args.out)
            return 1
        root = _resolve_root(root)
        report["source"] = {"mode": "local", "root": str(root),
                            "revision": args.revision}
        for split in ("train", "test"):
            splits[split] = load_split_samples(root, split)
        if args.include_holdout:
            splits["holdout"] = load_split_samples(root, "holdout")

    fp = dataset_fingerprint(splits, root=root, revision=args.revision)
    report["fingerprint"] = fp
    report["splits"] = fp["splits"]
    report["speaker_leakage"] = fp["speaker_leakage"]
    if not fp["speaker_leakage"]["ok"]:
        report["ok"] = False
        report["errors"].append(
            f"Speaker leakage between splits: {fp['speaker_leakage']['leaks']}"
        )

    # Schema coverage on in-scope train/test
    expected_phrases = {v.phrase for v in VARIATIONS}
    for split, samples in splits.items():
        found = {s.variation for s in samples
                 if not s.out_of_scope and s.variation}
        missing = sorted(expected_phrases - found)
        unknown = sorted(found - expected_phrases)
        report["coverage"][split] = {
            "n_variations_present": len(found),
            "missing_variations": missing,
            "unknown_variations": unknown,
            "n_oos": sum(1 for s in samples if s.out_of_scope),
        }
        if split in ("train", "test") and missing:
            report["warnings"].append(
                f"{split}: missing {len(missing)} canonical variations"
            )
        if unknown:
            report["warnings"].append(
                f"{split}: {len(unknown)} non-canonical variation labels"
            )

    # Per-source counts
    sources = {}
    for split, samples in splits.items():
        for s in samples:
            sources.setdefault(s.source or "(blank)", {"n": 0, "splits": set()})
            sources[s.source or "(blank)"]["n"] += 1
            sources[s.source or "(blank)"]["splits"].add(split)
    report["sources"] = {
        k: {"n": v["n"], "splits": sorted(v["splits"])}
        for k, v in sorted(sources.items(), key=lambda kv: -kv[1]["n"])
    }

    # Do not hard-code expected counts; just record observed.
    report["notes"] = [
        "Dataset card metadata and prose may disagree on exact counts; "
        "this audit records the downloaded/local observed counts.",
        "Holdout must only be used for Pi / vcm-benchmark evaluation.",
        "supplemental_fil has unresolved consent/license provenance; "
        "use only as an explicit optional ablation.",
    ]

    _emit(report, args.out)
    return 0 if report["ok"] else 2


def _emit(report: dict, out: str | None):
    text = json.dumps(report, indent=2, default=list)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text + "\n", encoding="utf-8")
        print(f"Wrote {out}")
    print(text)


if __name__ == "__main__":
    raise SystemExit(main())
