# Model selection (fill after DGX runs)

Pareto rule: maximize `test.real.command_accuracy` subject to
`false_accept_rate <= 0.10` and `params <= 500_000`.

```bash
python -m grammar_asr.scripts.compare_runs \
  grammar_asr/runs/ctc_gold_s0/summary.json \
  grammar_asr/runs/intent_gold_s1/summary.json \
  grammar_asr/runs/hybrid_gold_s0/summary.json \
  --max-far 0.10 --max-params 500000 \
  --out grammar_asr/runs/selection.json
```

Record the selected tag, params, real/synthetic command accuracy, FAR/FRR,
ONNX size, and Pi `vcm-benchmark` path here after training.
