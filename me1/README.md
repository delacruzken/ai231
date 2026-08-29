# ME1 — EinOps / Einsum

A 3-layer CNN for MNIST classification in which **every** layer/operation is
implemented with `einops` / `einsum` — no `torch.nn.Conv2d` or `MaxPool2d`.

- **Convolution** — sliding windows built with `unfold` (a view), contracted with a
  single `torch.einsum('bcijpq,ocpq->boij')`. Verified bit-for-bit against
  `nn.Conv2d(padding=1)`.
- **Pooling** — non-overlapping 2×2 max pooling via `einops.rearrange` + `max`.
  Verified identical to `nn.MaxPool2d(2)`.
- **Architecture** — `conv(1→16,pad1) → pool → conv(16→32,pad1) → pool →
  conv(32→64,pad1) → flatten → Linear(3136→10)`. Spatial flow 28→28→14→14→7.
  54,666 parameters.
- **Training** — Adam (lr=1e-3), batch 64, 5 epochs on the full 60k train split.

### Result

**Test-split accuracy: 99.11 %** (9911 / 10000). A 4×4 grid of 16 sampled test
images (ground truth vs. prediction) is rendered in the notebook.

Open [`ME1_einops_einsum.ipynb`](ME1_einops_einsum.ipynb) and run
`Jupyter Notebook` → *Open* to view the executed outputs.

### Notes

Built and executed by the OnIt agent (agent-initialized and committed, no human).
The einsum equation deliberately uses distinct index labels (`i, j` for spatial,
`p, q` for kernel) to avoid an index-collision bug in the bundled torch 2.13 einsum
parser when the `w` label overlaps the width dimension.
