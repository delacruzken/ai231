# Pi / vcm-benchmark results

Place finished official harness outputs here (one folder per run), e.g.:

```
artifacts/vcm_benchmark/<run_id>/
  report.md
  metrics.json
  trials.csv
  pi_metrics.csv
  pi_specs.json
  config.json
```

Generate with:

```bash
bash me2/grammar_asr/scripts/run_pi_benchmark.sh --prepare-pi   # needs PI_HOST=
# … run assistant on Pi + python benchmark.py in airimonda/vcm-benchmark …
bash me2/grammar_asr/scripts/run_pi_benchmark.sh --collect
```

Until a report is present, checklist item **#4 (Pi latency)** stays open.
