"""
Train the CTC character-level ASR encoder on the OptionB dataset.

Targets are the normalized transcripts (digits spelled out, punctuation
stripped) — the same representation the grammar trie uses, so the acoustic
model and the decoder agree on the symbol space.

Metrics:
  - CER / WER on the greedy CTC decode (standard ASR metrics)
  - CTC loss

After training, run decode_grammar.py for end-to-end command accuracy via the
trie-constrained decoder.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .asr_model import CTCEncoder, ctc_loss, greedy_decode
from .features import (OptionBDataset, load_manifest, make_dataloaders,
                       SAMPLE_RATE)
from .grammar import normalize, char_to_idx, BLANK, NUM_CLASSES

OUT_DIR = Path(__file__).resolve().parent / "runs"


def encode_targets(transcripts):
    """List[str] -> (targets tensor padded, lengths tensor)."""
    enc = []
    lens = []
    for t in transcripts:
        norm = normalize(t)
        ids = [char_to_idx[c] for c in norm if c in char_to_idx]
        enc.append(ids)
        lens.append(len(ids))
    max_len = max(lens) if lens else 0
    pad = []
    for ids in enc:
        pad.append(ids + [0] * (max_len - len(ids)))
    return torch.tensor(pad, dtype=torch.long), torch.tensor(lens, dtype=torch.long)


def post_len(T: int, model: CTCEncoder) -> int:
    """Frames after the CNN temporal downsampling (factor 2^(L-1) = 16)."""
    return max(1, T // 16 + 1)


def collate(batch):
    feats = [b["feat"] for b in batch]
    T = max(f.shape[0] for f in feats)
    C = feats[0].shape[1]
    padded = torch.zeros(len(feats), T, C)
    for i, f in enumerate(feats):
        padded[i, :f.shape[0]] = f
    return {
        "feat": padded,
        "T": torch.tensor([f.shape[0] for f in feats]),
        "intents": [b["intent"] for b in batch],
        "transcripts": [b["transcript"] for b in batch],
        "conditions": [b["condition"] for b in batch],
    }


def cer(hyp_ids, ref_ids):
    """Char error rate via Levenshtein (no external dep)."""
    import numpy as np
    n, m = len(ref_ids), len(hyp_ids)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if ref_ids[i - 1] == hyp_ids[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[m] / max(n, 1)


def wer(hyp_text, ref_text):
    """Word error rate (space-split)."""
    ref = ref_text.split()
    hyp = hyp_text.split()
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]; dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[m] / max(n, 1)


def decode_to_text(idx_list):
    from .grammar import idx_to_char
    return "".join(idx_to_char[i] for i in idx_list if i != NUM_CLASSES - 1)


def evaluate(model, loader, device, max_batches=None):
    model.eval()
    tot_cer = 0.0
    tot_wer = 0.0
    n = 0
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if max_batches and bi >= max_batches:
                break
            feat = batch["feat"].to(device)
            logits = model(feat)
            decoded = greedy_decode(logits)
            for ids, ref in zip(decoded, batch["transcripts"]):
                hyp_text = decode_to_text(ids)
                ref_norm = normalize(ref)
                ref_ids = [char_to_idx[c] for c in ref_norm if c in char_to_idx]
                tot_cer += cer(ids, ref_ids)
                tot_wer += wer(hyp_text, ref_norm)
                n += 1
    return tot_cer / max(n, 1), tot_wer / max(n, 1), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--max-seconds", type=float, default=6.0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "mps"
                    if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--tag", type=str, default="base")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    run_dir = OUT_DIR / args.tag
    run_dir.mkdir(exist_ok=True)

    device = torch.device(args.device)
    print(f"Device: {device}")

    model = CTCEncoder(hidden=args.hidden, num_layers=args.layers).to(device)
    print(f"Params: {model.count_params():,}")

    train_dl, val_dl, test_dl, train_ds, val_ds, test_ds = make_dataloaders(
        batch_size=args.batch_size, num_workers=args.num_workers,
        max_seconds=args.max_seconds, collate_fn=collate)
    print(f"Train {len(train_ds)} | Val {len(val_ds)} | Test {len(test_ds)}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    blank = NUM_CLASSES - 1

    history = []
    best_val_cer = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        tot_loss, n_batch = 0.0, 0
        for batch in train_dl:
            feat = batch["feat"].to(device)
            T_raw = batch["T"].to(device)
            # post-CNN length
            T_post = torch.tensor(
                [post_len(int(t), model) for t in T_raw.tolist()],
                dtype=torch.long).to(device)
            targets, tgt_lens = encode_targets(batch["transcripts"])
            targets = targets.to(device)
            tgt_lens = tgt_lens.to(device)
            # CTC constraint: target length must be <= input length
            valid = tgt_lens <= T_post
            if not bool(valid.all()):
                # drop offending samples from this batch
                keep = valid.nonzero(as_tuple=True)[0]
                if len(keep) < 2:
                    continue
                feat = feat[keep]; T_post = T_post[keep]
                targets = targets[keep]; tgt_lens = tgt_lens[keep]

            logits = model(feat)
            loss = ctc_loss(logits, targets, T_post, tgt_lens, blank=blank)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot_loss += loss.item()
            n_batch += 1
        sched.step()
        avg_loss = tot_loss / max(n_batch, 1)

        val_cer, val_wer, vn = evaluate(model, val_dl, device)
        dt = time.time() - t0
        print(f"Epoch {epoch:2d}/{args.epochs} | loss {avg_loss:.3f} | "
              f"val CER {val_cer*100:5.2f}% WER {val_wer*100:5.2f}% "
              f"({vn}) | {dt:.0f}s")
        history.append({"epoch": epoch, "loss": avg_loss,
                        "val_cer": val_cer, "val_wer": val_wer})
        if val_cer < best_val_cer:
            best_val_cer = val_cer
            torch.save(model.state_dict(), run_dir / "best.pt")
            print(f"  -> saved best (val CER {val_cer*100:.2f}%)")

    # final test eval
    model.load_state_dict(torch.load(run_dir / "best.pt"))
    test_cer, test_wer, tn = evaluate(model, test_dl, device)
    print(f"\nTEST (best model): CER {test_cer*100:.2f}%  WER {test_wer*100:.2f}%  ({tn})")

    with open(run_dir / "history.json", "w") as f:
        json.dump({"args": vars(args), "history": history,
                   "test_cer": test_cer, "test_wer": test_wer}, f, indent=2)
    print(f"Saved to {run_dir}")


if __name__ == "__main__":
    main()
