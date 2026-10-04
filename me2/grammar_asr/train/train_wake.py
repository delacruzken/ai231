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
import soundfile as sf
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positives", required=True, help="Dir of wake WAVs")
    ap.add_argument("--negatives", required=True,
                    help="Dir of non-wake WAVs (commands/noise)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", type=str, default="grammar_asr/runs/wake/best.pt")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    pos = sorted(Path(args.positives).glob("*.wav"))
    neg = sorted(Path(args.negatives).glob("*.wav"))
    if not pos or not neg:
        raise SystemExit("Need both positive and negative WAVs")
    items = [(p, 1) for p in pos] + [(p, 0) for p in neg]
    random.Random(0).shuffle(items)
    n_val = max(1, len(items) // 10)
    train_items, val_items = items[n_val:], items[:n_val]

    device = torch.device(args.device)
    model = build_model("wake").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_dl = DataLoader(WakeDataset(train_items), batch_size=args.batch_size,
                          shuffle=True, collate_fn=collate)
    val_dl = DataLoader(WakeDataset(val_items), batch_size=args.batch_size,
                        shuffle=False, collate_fn=collate)

    best = -1.0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)["wake_logits"]
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        # val acc
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                pred = model(x)["wake_logits"].argmax(-1)
                correct += int((pred == y).sum())
                total += len(y)
        acc = correct / max(total, 1)
        print(f"epoch {epoch} val_acc {acc*100:.1f}%")
        if acc >= best:
            best = acc
            torch.save({"model": model.state_dict(), "epoch": epoch}, out)
    print(json.dumps({"best_val_acc": best, "out": str(out),
                      "params": model.count_params()}))


if __name__ == "__main__":
    main()
