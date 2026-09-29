"""
3-tier benchmark for the ME2 VCM (feeds tasks 3 & 4).

Tier 1 - Clean:      command accuracy on clean test WAVs
Tier 2 - Noisy:      command accuracy on noisy test WAVs (~30 dB SNR, built-in)
Tier 3 - Latency:    per-utterance inference time (p50/p95/max) on this device

Also reports CER/WER (ASR quality) and per-intent breakdown. Writes a JSON
summary + a Markdown table for the report.

Usage:
  python -m grammar_asr.benchmark --ckpt runs/base/best.pt --device cpu
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .asr_model import CTCEncoder, greedy_decode
from .asr_model_low import ResCTCEncoderLow
from .decode_grammar import evaluate_command_accuracy, make_model
from .features import (load_manifest, load_wav, log_mel, normalize_feat,
                       SAMPLE_RATE)
from .grammar import build_grammar, normalize, char_to_idx, idx_to_char, NUM_CLASSES
from .train_asr import evaluate as eval_cer_wer

OUT_DIR = Path(__file__).resolve().parent / "runs"


def measure_latency(model: CTCEncoder, samples, device: str, n: int = 50,
                    warmup: int = 5):
    """Per-utterance forward-pass latency (ms)."""
    model.eval()
    latencies = []
    for i, s in enumerate(samples):
        if i < warmup:
            continue
        if i >= warmup + n:
            break
        audio = load_wav(s.path)
        feat = torch.from_numpy(normalize_feat(log_mel(audio))).unsqueeze(0).to(device)
        # sync for fair timing
        if device != "cpu":
            torch.cuda.synchronize() if device == "cuda" else None
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(feat)
        if device != "cpu":
            torch.cuda.synchronize() if device == "cuda" else None
        latencies.append((time.perf_counter() - t0) * 1000)
    latencies.sort()
    return {
        "p50_ms": latencies[len(latencies) // 2],
        "p95_ms": latencies[int(len(latencies) * 0.95)],
        "max_ms": latencies[-1],
        "mean_ms": sum(latencies) / len(latencies),
        "n": len(latencies),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--model", type=str, default="low",
                    choices=["base", "res", "low"])
    ap.add_argument("--char-reward", type=float, default=0.15)
    ap.add_argument("--max-n", type=int, default=None,
                    help="cap samples per tier (for quick runs)")
    ap.add_argument("--tag", type=str, default="bench")
    args = ap.parse_args()

    device = torch.device(args.device)
    model = make_model(args.model).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    trie = build_grammar()

    test_clean = load_manifest(split="test", condition="clean")
    test_noisy = load_manifest(split="test", condition="noisy")
    if args.max_n:
        test_clean = test_clean[:args.max_n]
        test_noisy = test_noisy[:args.max_n]

    print(f"Device: {device} | model: {args.model} | char_reward: {args.char_reward}")
    print(f"Test clean: {len(test_clean)} | Test noisy: {len(test_noisy)}")

    # Tier 1: clean command accuracy (grammar decode)
    print("\n[Tier 1] Clean - grammar-constrained decode ...")
    t1 = evaluate_command_accuracy(model, test_clean, trie, device,
                                   "grammar", 0.1, args.char_reward, args.max_n)

    # Tier 2: noisy command accuracy
    print("[Tier 2] Noisy - grammar-constrained decode ...")
    t2 = evaluate_command_accuracy(model, test_noisy, trie, device,
                                   "grammar", 0.1, args.char_reward, args.max_n)

    # CER/WER on clean (ASR quality)
    print("[ASR] CER/WER on clean test ...")
    from .features import OptionBDataset
    from torch.utils.data import DataLoader
    from .train_asr import collate
    ds = OptionBDataset(test_clean, max_seconds=6.0)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0,
                    collate_fn=collate)
    cer, wer, n = eval_cer_wer(model, dl, device)

    # Tier 3: latency
    print("[Tier 3] Latency ...")
    lat = measure_latency(model, test_clean, device)

    result = {
        "device": str(device),
        "model": args.model,
        "char_reward": args.char_reward,
        "tier1_clean": {"accuracy": t1["accuracy"], "correct": t1["correct"],
                         "total": t1["total"], "per_intent": t1["per_intent"]},
        "tier2_noisy": {"accuracy": t2["accuracy"], "correct": t2["correct"],
                         "total": t2["total"], "per_intent": t2["per_intent"]},
        "asr_quality": {"cer": cer, "wer": wer, "n": n},
        "tier3_latency": lat,
    }

    OUT_DIR.mkdir(exist_ok=True)
    out_json = OUT_DIR / f"{args.tag}.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)

    # Markdown summary
    md = []
    md.append(f"# ME2 VCM Benchmark ({args.tag})\n")
    md.append(f"- Device: `{device}`  |  Model type: {args.model}  |  "
              f"char_reward: {args.char_reward}")
    md.append(f"- Checkpoint: `{args.ckpt}`  |  Params: {model.count_params():,}\n")
    md.append("## Summary\n")
    md.append("| Metric | Value |")
    md.append("|---|---|")
    md.append(f"| Clean command accuracy | **{t1['accuracy']*100:.2f}%** "
              f"({t1['correct']}/{t1['total']}) |")
    md.append(f"| Noisy command accuracy | **{t2['accuracy']*100:.2f}%** "
              f"({t2['correct']}/{t2['total']}) |")
    md.append(f"| CER (clean) | {cer*100:.2f}% |")
    md.append(f"| WER (clean) | {wer*100:.2f}% |")
    md.append(f"| Latency p50 | {lat['p50_ms']:.1f} ms |")
    md.append(f"| Latency p95 | {lat['p95_ms']:.1f} ms |")
    md.append(f"| Latency max | {lat['max_ms']:.1f} ms |\n")
    md.append("## Per-intent (clean)\n")
    md.append("| Intent | Correct/Total | Accuracy |")
    md.append("|---|---|---|")
    for intent, st in t1["per_intent"].items():
        md.append(f"| {intent} | {st['correct']}/{st['total']} | "
                  f"{st['acc']*100:.1f}% |")
    md_path = OUT_DIR / f"{args.tag}.md"
    md_path.write_text("\n".join(md))
    print(f"\nSaved {out_json} and {md_path}")
    print(f"\n=== SUMMARY ===")
    print(f"Clean acc : {t1['accuracy']*100:.2f}%")
    print(f"Noisy acc : {t2['accuracy']*100:.2f}%")
    print(f"CER/WER   : {cer*100:.2f}% / {wer*100:.2f}%")
    print(f"Latency   : p50={lat['p50_ms']:.1f}ms p95={lat['p95_ms']:.1f}ms")


if __name__ == "__main__":
    main()
