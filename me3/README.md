# ME3

Full-duplex voice agent for the AI231 machine exercise.

| Path | Role |
|---|---|
| [`ME3_Architecture_and_Development_Plan.md`](ME3_Architecture_and_Development_Plan.md) | Agreed architecture and phase plan |
| [`voice-agent/`](voice-agent/) | Application source (Pipecat shuttle, speech, tools, UI, fdbench bridge) |
| `fd-benchmark/` | **Not in this git tree** — clone locally (see below) |

## fd-benchmark (local only)

The course harness is a separate repo. Do not vendor its `.git`, datasets, runs, or venv into `ai231`.

```bash
cd me3
git clone https://github.com/trueshotjames/ai231-me3-full-duplex-benchmarking fd-benchmark
cd fd-benchmark
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# optional scoring deps on a laptop: pip install -r requirements-score.txt
```

Point the launcher at ME3 (either env or a local entry file):

```bash
export FDBENCH_PROJECT="$(cd ../voice-agent && pwd)"
export FDBENCH_ENTRY="$FDBENCH_PROJECT/scripts/fdbench_assistant_entry.py"
# or: cp "$FDBENCH_ENTRY" assistant_entry.py   # path is gitignored upstream
```

See [`voice-agent/README.md`](voice-agent/README.md) for Phase 8 checks and reset URL.
