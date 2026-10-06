# Historical baseline (pre-rehaul)

The September 2026 Option-B-only grammar-CTC stack is retained here as a
**historical baseline**, not the current system of record.

| Item | Value |
| --- | --- |
| Model | `ResCTCEncoderLow` (1.93M params, 2× downsample CTC) |
| Train data | Option B synthetic-only set (~100 speakers) |
| Clean command (intent-only) accuracy | 80.76% (726/899) |
| Noisy command (intent-only) accuracy | 85.43% (768/899) |
| CER / WER (clean) | 1.90% / 5.26% |
| ONNX fp32 size | 7.76 MB |
| RPi5 WAV path (docs) | ~260–293 ms cold, 4/4 smoke WAVs |

### Why it was superseded

- Metrics scored **intent only** (slots ignored).
- No official gold splits / speaker-disjoint holdout contract.
- No OOS rejection, wake gating, or `vcm-benchmark` logging.
- Grammar phrases drifted (`Place a call`, `Pause for now`).
- Footprint far above compact intent classifiers (~0.2–0.4M).

See the current [README.md](../README.md) for the rehauled gold-dataset system.
