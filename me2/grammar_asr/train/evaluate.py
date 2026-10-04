"""Semantic evaluation on official splits (never holdout for training loops)."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import torch

from ..decode.semantic import decode_model_output, RejectGate, SemanticPrediction
from ..decode.grammar_trie import build_grammar
from ..schema import OUT_OF_SCOPE


def _slice_key(batch, i):
    return {
        "is_synthetic": bool(batch["is_synthetic"][i]),
        "accent_group": batch["accent_group"][i],
        "source": batch["source"][i],
        "intent": batch["intent"][i],
    }


def evaluate_loader(model, loader, device, *,
                    reject_gate: Optional[RejectGate] = None,
                    prefer: str = "auto",
                    char_reward: float = 0.15,
                    max_batches: Optional[int] = None) -> dict:
    model.eval()
    trie = build_grammar()
    gate = reject_gate or RejectGate()

    intent_correct = command_correct = total = 0
    oos_total = oos_fa = in_total = in_fr = 0
    per_intent = defaultdict(lambda: {"correct": 0, "total": 0})
    real = {"intent_correct": 0, "command_correct": 0, "total": 0}
    synth = {"intent_correct": 0, "command_correct": 0, "total": 0}
    predictions: List[dict] = []

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if max_batches is not None and bi >= max_batches:
                break
            feat = batch["feat"].to(device)
            lengths = batch["T"].to(device)
            try:
                outputs = model(feat, lengths=lengths)
            except TypeError:
                outputs = model(feat)
            if not isinstance(outputs, dict):
                outputs = {"ctc_logits": outputs}

            B = feat.shape[0]
            for i in range(B):
                out_i = {}
                for k, v in outputs.items():
                    if torch.is_tensor(v):
                        out_i[k] = v[i:i + 1]
                    elif isinstance(v, dict):
                        out_i[k] = {kk: vv[i:i + 1] for kk, vv in v.items()}
                pred = decode_model_output(out_i, trie=trie, char_reward=char_reward,
                                           prefer=prefer)
                pred = gate.apply(pred)

                ref_intent = (OUT_OF_SCOPE if bool(batch["out_of_scope"][i])
                              else batch["intent"][i])
                ref_slot = "" if ref_intent == OUT_OF_SCOPE else batch["slot"][i]
                is_oos = ref_intent == OUT_OF_SCOPE

                total += 1
                intent_ok = pred.intent == ref_intent
                command_ok = intent_ok and (pred.slot == ref_slot)
                if intent_ok:
                    intent_correct += 1
                if command_ok:
                    command_correct += 1

                if is_oos:
                    oos_total += 1
                    if not pred.rejected and pred.intent != OUT_OF_SCOPE:
                        oos_fa += 1
                else:
                    in_total += 1
                    per_intent[ref_intent]["total"] += 1
                    if intent_ok:
                        per_intent[ref_intent]["correct"] += 1
                    if pred.rejected or pred.intent == OUT_OF_SCOPE:
                        in_fr += 1

                bucket = real if not bool(batch["is_synthetic"][i]) else synth
                bucket["total"] += 1
                if intent_ok:
                    bucket["intent_correct"] += 1
                if command_ok:
                    bucket["command_correct"] += 1

                predictions.append({
                    "file": batch["file"][i],
                    "ref_intent": ref_intent,
                    "ref_slot": ref_slot,
                    "pred_intent": pred.intent,
                    "pred_slot": pred.slot,
                    "rejected": pred.rejected,
                    "confidence": pred.confidence,
                    "source": pred.source,
                    **_slice_key(batch, i),
                })

    def rate(num, den):
        return float(num) / max(den, 1)

    result = {
        "n": total,
        "intent_accuracy": rate(intent_correct, total),
        "command_accuracy": rate(command_correct, total),
        "false_accept_rate": rate(oos_fa, oos_total),
        "false_reject_rate": rate(in_fr, in_total),
        "n_oos": oos_total,
        "n_in_scope": in_total,
        "per_intent": {
            k: {"correct": v["correct"], "total": v["total"],
                "acc": rate(v["correct"], v["total"])}
            for k, v in sorted(per_intent.items())
        },
        "real": {
            "n": real["total"],
            "intent_accuracy": rate(real["intent_correct"], real["total"]),
            "command_accuracy": rate(real["command_correct"], real["total"]),
        },
        "synthetic": {
            "n": synth["total"],
            "intent_accuracy": rate(synth["intent_correct"], synth["total"]),
            "command_accuracy": rate(synth["command_correct"], synth["total"]),
        },
        "reject_gate": gate.to_dict(),
        "predictions": predictions,
    }
    return result


def save_eval(result: dict, path: Path) -> None:
    slim = {k: v for k, v in result.items() if k != "predictions"}
    path.write_text(json.dumps(slim, indent=2) + "\n")
    pred_path = path.with_name(path.stem + "_predictions.jsonl")
    with open(pred_path, "w", encoding="utf-8") as f:
        for row in result.get("predictions") or []:
            f.write(json.dumps(row) + "\n")
