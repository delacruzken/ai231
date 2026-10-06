"""Train a tiny from-scratch wake detector on positives + hard negatives.

Positives: short wake-phrase clips (directory of WAVs).
Negatives: random gold-train commands + noise (speaker-disjoint from holdout).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..data.features import SAMPLE_RATE, log_mel, normalize_feat, load_wav
from ..models import build_model


class WakeDataset(Dataset):
    def __init__(self, items, max_seconds=1.5):
        self.items = items
        self.max = int(max_seconds * SAMPLE_RATE)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        audio = load_wav(Path(path))
        if len(audio) > self.max:
            audio = audio[: self.max]
        else:
            audio = np.pad(audio, (0, self.max - len(audio)))
        feat = normalize_feat(log_mel(audio))
        return torch.from_numpy(feat), torch.tensor(label, dtype=torch.long)


def collate(batch):
    xs = torch.stack([b[0] for b in batch])
    ys = torch.stack([b[1] for b in batch])
    return xs, ys


def find_wavs(paths):
    files = []
    for root in paths:
        files.extend(Path(root).rglob("*.wav"))
    return sorted(set(files))


def stratified_split(paths, val_fraction, rng):
    paths = list(paths)
    rng.shuffle(paths)
    n_val = max(1, int(round(len(paths) * val_fraction)))
    return paths[n_val:], paths[:n_val]


def evaluate(model, loader, device, target_far):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))["wake_logits"]
            probs.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().tolist())
            labels.extend(y.tolist())

    probs_np = np.asarray(probs, dtype=np.float64)
    labels_np = np.asarray(labels, dtype=np.int64)
    positive = labels_np == 1
    negative = ~positive
    # Include values just above every score so zero-FAR thresholds are possible.
    candidates = {0.0, 1.0}
    candidates.update(float(v) for v in probs_np)
    candidates.update(float(np.nextafter(v, 1.0)) for v in probs_np)
    best = None
    for threshold in sorted(candidates):
        pred = probs_np >= threshold
        recall = float(pred[positive].mean()) if positive.any() else 0.0
        far = float(pred[negative].mean()) if negative.any() else 0.0
        if far <= target_far:
            key = (recall, -far, -threshold)
            if best is None or key > best[0]:
                best = (key, threshold, pred, recall, far)
    if best is None:
        threshold = 1.0
        pred = probs_np >= threshold
        recall = float(pred[positive].mean()) if positive.any() else 0.0
        far = float(pred[negative].mean()) if negative.any() else 0.0
    else:
        _, threshold, pred, recall, far = best
    precision = (
        float(labels_np[pred].mean()) if pred.any() else 0.0
    )
    specificity = (
        float((~pred[negative]).mean()) if negative.any() else 0.0
    )
    return {
        "threshold": round(float(threshold), 6),
        "accuracy": float((pred == labels_np).mean()),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "precision": precision,
        "recall": recall,
        "false_accept_rate": far,
        "false_reject_rate": 1.0 - recall,
        "n_positive": int(positive.sum()),
        "n_negative": int(negative.sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positives", required=True, help="Dir of wake WAVs")
    ap.add_argument("--negatives", required=True, action="append",
                    help="Dir of non-wake WAVs; repeat for multiple sources")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", type=str, default="grammar_asr/runs/wake/best.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--target-far", type=float, default=0.05)
    args = ap.parse_args()

    pos = find_wavs([args.positives])
    neg = find_wavs(args.negatives)
    if not pos or not neg:
        raise SystemExit("Need both positive and negative WAVs")
    rng = random.Random(args.seed)
    train_pos, val_pos = stratified_split(pos, args.val_fraction, rng)
    train_neg, val_neg = stratified_split(neg, args.val_fraction, rng)
    train_items = [(p, 1) for p in train_pos] + [(p, 0) for p in train_neg]
    val_items = [(p, 1) for p in val_pos] + [(p, 0) for p in val_neg]
    rng.shuffle(train_items)

    device = torch.device(args.device)
    model = build_model("wake").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    class_weights = torch.tensor(
        [
            len(train_items) / (2 * len(train_neg)),
            len(train_items) / (2 * len(train_pos)),
        ],
        dtype=torch.float32,
        device=device,
    )
    train_dl = DataLoader(WakeDataset(train_items), batch_size=args.batch_size,
                          shuffle=True, collate_fn=collate)
    val_dl = DataLoader(WakeDataset(val_items), batch_size=args.batch_size,
                        shuffle=False, collate_fn=collate)

    print(
        f"Train positive={len(train_pos)} negative={len(train_neg)} | "
        f"Val positive={len(val_pos)} negative={len(val_neg)}"
    )
    best = -1.0
    best_metrics = {}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)["wake_logits"]
            loss = F.cross_entropy(logits, y, weight=class_weights)
            opt.zero_grad()
            loss.backward()
            opt.step()
        metrics = evaluate(model, val_dl, device, args.target_far)
        score = metrics["balanced_accuracy"]
        print(
            f"epoch {epoch} balanced_acc={score*100:.1f}% "
            f"recall={metrics['recall']*100:.1f}% "
            f"FAR={metrics['false_accept_rate']*100:.1f}% "
            f"threshold={metrics['threshold']:.3f}"
        )
        if score >= best:
            best = score
            best_metrics = metrics
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "threshold": metrics["threshold"],
                "metrics": metrics,
            }, out)
    summary = {
        "best_val_balanced_accuracy": best,
        "validation": best_metrics,
        "out": str(out),
        "params": model.count_params(),
        "train_positive": len(train_pos),
        "train_negative": len(train_neg),
        "val_positive": len(val_pos),
        "val_negative": len(val_neg),
    }
    summary_path = out.with_suffix(".metrics.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
