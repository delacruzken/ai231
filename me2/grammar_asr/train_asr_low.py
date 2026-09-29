"""
Train the low-downsample residual CTC encoder (asr_model_low.ResCTCEncoderLow).

This is the production model: T' ~= T/2 so every grammar word (up to 46 chars)
is CTC-alignable. Same recipe as the other trainers.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .asr_model import ctc_loss
from .asr_model_low import ResCTCEncoderLow, post_len_low
from .features import make_dataloaders
from .grammar import NUM_CLASSES
from .train_asr import encode_targets, collate, evaluate, OUT_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--max-seconds", type=float, default=6.0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", type=str,
                    default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--tag", type=str, default="low")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    run_dir = OUT_DIR / args.tag
    run_dir.mkdir(exist_ok=True)
    device = torch.device(args.device)
    print(f"Device: {device}")

    model = ResCTCEncoderLow(hidden=args.hidden, num_blocks=args.blocks).to(device)
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
            T_post = torch.tensor(
                [post_len_low(int(t)) for t in T_raw.tolist()],
                dtype=torch.long).to(device)
            targets, tgt_lens = encode_targets(batch["transcripts"])
            targets = targets.to(device)
            tgt_lens = tgt_lens.to(device)
            valid = tgt_lens <= T_post
            if not bool(valid.all()):
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

    model.load_state_dict(torch.load(run_dir / "best.pt"))
    test_cer, test_wer, tn = evaluate(model, test_dl, device)
    print(f"\nTEST (best model): CER {test_cer*100:.2f}%  WER {test_wer*100:.2f}%  ({tn})")
    with open(run_dir / "history.json", "w") as f:
        json.dump({"args": vars(args), "history": history,
                   "test_cer": test_cer, "test_wer": test_wer}, f, indent=2)
    print(f"Saved to {run_dir}")


if __name__ == "__main__":
    main()
