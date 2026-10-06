#!/usr/bin/env python3
"""Config-driven ME2 VCM trainer (CTC / intent-CRNN / hybrid), DGX-ready.

Examples:
  python -m grammar_asr.train.train --config grammar_asr/configs/smoke.yaml
  python -m grammar_asr.train.train --config grammar_asr/configs/intent_gold.yaml --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

from ..data.dataset import (
    discover_dataset_root, build_mixture, make_dataloaders, dataset_fingerprint,
    load_split_samples, _resolve_root,
)
from ..decode.semantic import RejectGate, calibrate_reject_gate, decode_model_output
from ..decode.grammar_trie import build_grammar, export_grammar_json
from ..models import build_model, count_params
from ..schema import schema_fingerprint
from .losses import multitask_loss
from .evaluate import evaluate_loader, save_eval

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ME2_ROOT = PACKAGE_ROOT.parent
RUNS_DIR = PACKAGE_ROOT / "runs"


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ME2_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def forward_model(model, feat, lengths):
    try:
        out = model(feat, lengths=lengths)
    except TypeError:
        out = model(feat)
    if not isinstance(out, dict):
        out = {"ctc_logits": out}
    return out


def train_one(cfg: Dict[str, Any], config_path: Path) -> Path:
    seed = int(cfg.get("seed", 0))
    set_seed(seed)
    device = resolve_device(str(cfg.get("device", "auto")))
    tag = cfg.get("tag") or f"{cfg['model']['name']}_{seed}"
    run_dir = RUNS_DIR / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = cfg.get("data", {})
    use_hf = bool(data_cfg.get("use_hf", False))
    root = None
    if not use_hf:
        root_arg = data_cfg.get("root")
        if root_arg and not Path(root_arg).is_absolute():
            # Resolve relative roots against me2/ (package parent), then CWD.
            cand = (ME2_ROOT / root_arg).resolve()
            root_arg = str(cand if cand.exists() else Path(root_arg).resolve())
        root = discover_dataset_root(root_arg)
        use_hf = root is None
    if data_cfg.get("require_local") and (use_hf or root is None):
        raise FileNotFoundError(
            "Local dataset required but not found. Set data.root or ME2_DATA_ROOT."
        )

    splits = build_mixture(
        None if use_hf else root,
        use_hf=use_hf,
        revision=data_cfg.get("revision"),
        include_negatives=bool(data_cfg.get("include_negatives", False)),
        include_fil=bool(data_cfg.get("include_fil", False)),
        include_synth_extra=bool(data_cfg.get("include_synth_extra", False)),
        include_numerals=bool(data_cfg.get("include_numerals", False)),
        val_fraction=float(data_cfg.get("val_fraction", 0.1)),
        seed=seed,
    )

    # Smoke mode: subsample
    if cfg.get("smoke"):
        n = int(cfg.get("smoke_n", 64))
        for k in list(splits):
            splits[k] = splits[k][:n]

    fp = dataset_fingerprint(
        {k: v for k, v in splits.items() if k != "holdout"},
        root=None if use_hf else root,
        revision=data_cfg.get("revision"),
    )
    if not fp["speaker_leakage"]["ok"]:
        raise RuntimeError(f"Speaker leakage detected: {fp['speaker_leakage']}")

    loaders = make_dataloaders(
        splits,
        batch_size=int(cfg.get("batch_size", 32)),
        num_workers=int(cfg.get("num_workers", 2)),
        max_seconds=float(cfg.get("max_seconds", 5.0)),
    )

    model_cfg = dict(cfg.get("model", {}))
    name = model_cfg.pop("name")
    model = build_model(name, **model_cfg).to(device)
    prefer = cfg.get("decode_prefer", "auto")
    if name in {"ctc", "ctc_low", "low"}:
        prefer = "grammar"
    elif name in {"intent", "intent_crnn", "crnn"}:
        prefer = "intent_head"

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("lr", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    epochs = int(cfg.get("epochs", 30))
    warmup = int(cfg.get("warmup_epochs", 0))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, epochs - warmup)
    )

    amp = bool(cfg.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    loss_cfg = cfg.get("loss", {})
    history = []
    best_metric = -1.0
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"

    meta = {
        "config_path": str(config_path),
        "config": cfg,
        "schema": schema_fingerprint(),
        "dataset": fp,
        "git_commit": git_commit(),
        "seed": seed,
        "device": str(device),
        "params": count_params(model),
        "model_name": name,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "env": {
            "python": sys.version.split()[0],
            "torch": str(torch.__version__),
            "cuda": str(torch.version.cuda),
            "cwd": os.getcwd(),
        },
    }
    # Keep checkpoint payloads picklable / weights_only-friendly.
    ckpt_meta = {
        "tag": tag,
        "model_name": name,
        "seed": seed,
        "git_commit": meta["git_commit"],
        "params": meta["params"],
    }
    (run_dir / "config.resolved.json").write_text(
        json.dumps(meta, indent=2, default=str) + "\n"
    )
    export_grammar_json(PACKAGE_ROOT / "grammar_commands.json")

    print(f"Run {tag} | model={name} params={count_params(model):,} | device={device}")
    print(f"Train {len(splits['train'])} | Val {len(splits['val'])} | "
          f"Test {len(splits['test'])} | hf={use_hf}")

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        tot = 0.0
        n_batch = 0
        for batch in loaders["train"]:
            feat = batch["feat"].to(device)
            lengths = batch["T"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                outputs = forward_model(model, feat, lengths)
                loss, parts = multitask_loss(
                    outputs, batch,
                    ctc_weight=float(loss_cfg.get("ctc_weight", 1.0)),
                    intent_weight=float(loss_cfg.get("intent_weight", 1.0)),
                    slot_weight=float(loss_cfg.get("slot_weight", 1.0)),
                    confusable_weight=float(loss_cfg.get("confusable_weight", 0.0)),
                    label_smoothing=float(loss_cfg.get("label_smoothing", 0.0)),
                )
            if not torch.isfinite(loss):
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            tot += float(loss.detach())
            n_batch += 1

        if epoch > warmup:
            sched.step()

        # Val selection metric: command accuracy (no reject gate yet)
        val = evaluate_loader(
            model, loaders["val"], device, prefer=prefer,
            char_reward=float(cfg.get("char_reward", 0.15)),
            max_batches=cfg.get("max_eval_batches"),
        )
        metric = val["command_accuracy"]
        row = {
            "epoch": epoch,
            "loss": tot / max(n_batch, 1),
            "val_intent_acc": val["intent_accuracy"],
            "val_command_acc": val["command_accuracy"],
            "val_far": val["false_accept_rate"],
            "val_frr": val["false_reject_rate"],
            "lr": opt.param_groups[0]["lr"],
            "seconds": time.time() - t0,
        }
        history.append(row)
        print(
            f"Epoch {epoch:3d}/{epochs} loss {row['loss']:.3f} "
            f"val_cmd {metric*100:5.2f}% intent {val['intent_accuracy']*100:5.2f}% "
            f"FAR {val['false_accept_rate']*100:4.1f}% "
            f"FRR {val['false_reject_rate']*100:4.1f}% ({row['seconds']:.0f}s)"
        )
        torch.save({"model": model.state_dict(), "epoch": epoch, "meta": ckpt_meta},
                   last_path)
        if metric >= best_metric:
            best_metric = metric
            torch.save({"model": model.state_dict(), "epoch": epoch, "meta": ckpt_meta},
                       best_path)
            print(f"  -> saved best (val command {metric*100:.2f}%)")

        (run_dir / "history.json").write_text(
            json.dumps({"history": history, "best_val_command_acc": best_metric},
                       indent=2) + "\n"
        )

    # Load best, calibrate reject gate on val, evaluate test
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    # Collect val predictions for calibration
    raw_val = evaluate_loader(model, loaders["val"], device, prefer=prefer,
                              reject_gate=RejectGate(min_confidence=0.0),
                              char_reward=float(cfg.get("char_reward", 0.15)))
    from ..decode.semantic import SemanticPrediction
    preds = [
        SemanticPrediction(
            intent=p["pred_intent"], slot=p["pred_slot"],
            variation="", variation_id=-1, rejected=False,
            score=p["confidence"], margin=0.0, confidence=p["confidence"],
            source=p["source"],
        )
        for p in raw_val["predictions"]
    ]
    labels_oos = [p["ref_intent"] == "OUT_OF_SCOPE" for p in raw_val["predictions"]]
    gate = calibrate_reject_gate(
        preds, labels_oos, target_far=float(cfg.get("target_far", 0.10))
    )
    gate.save(run_dir / "reject_gate.json")

    val_final = evaluate_loader(model, loaders["val"], device, prefer=prefer,
                                reject_gate=gate,
                                char_reward=float(cfg.get("char_reward", 0.15)))
    test_final = evaluate_loader(model, loaders["test"], device, prefer=prefer,
                                 reject_gate=gate,
                                 char_reward=float(cfg.get("char_reward", 0.15)))
    save_eval(val_final, run_dir / "val_metrics.json")
    save_eval(test_final, run_dir / "test_metrics.json")

    summary = {
        "tag": tag,
        "model": name,
        "params": count_params(model),
        "best_epoch": ckpt.get("epoch"),
        "best_val_command_acc": best_metric,
        "val": {k: val_final[k] for k in (
            "intent_accuracy", "command_accuracy", "false_accept_rate",
            "false_reject_rate", "real", "synthetic", "n")},
        "test": {k: test_final[k] for k in (
            "intent_accuracy", "command_accuracy", "false_accept_rate",
            "false_reject_rate", "real", "synthetic", "n")},
        "reject_gate": gate.to_dict(),
        "git_commit": meta["git_commit"],
        "dataset_fingerprint": fp["sha256_file_list"],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return run_dir


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(Path(args.config))
    if args.device:
        cfg["device"] = args.device
    if args.tag:
        cfg["tag"] = args.tag
    if args.smoke:
        cfg["smoke"] = True
    train_one(cfg, Path(args.config))


if __name__ == "__main__":
    main()
