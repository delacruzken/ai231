# ME3 voice-agent

English-first, interruptible full-duplex voice agent for Raspberry Pi 5.  
Architecture: [`../ME3_Architecture_and_Development_Plan.md`](../ME3_Architecture_and_Development_Plan.md).

**Phase 8 status:** fd-benchmark direct-feed integration (Pipecat transport shuttle → ME3 SessionController, stage marks, trial reset). Publication remains Phase 7.

## Keys

| Env | Needed? |
|---|---|
| `ME3_UI_TOKEN` | Yes — set in `.env`; UI auto-loads via `/api/ui-bootstrap` |
| `ANTHROPIC_API_KEY` | Live Haiku |
| `TAVILY_API_KEY` | Live search |
| `OPENCODE_URL` | Optional (default `http://127.0.0.1:4096`) |
| `OPENCODE_SERVER_PASSWORD` | Only if OpenCode server enables basic auth |
| `OPENCODE_FORCE_MOCK=true` | Force MockWorker |
| `GITHUB_PAT` | Live publish |
| `GITHUB_OWNER` / `GITHUB_REPO` | Live publish target |
| `GITHUB_SITE_ROOT` | Repo folder for Pages files (default `docs`) |

## Quick start

```bash
cd me3/voice-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,speech]"
cp .env.example .env
./scripts/fetch_speech_models.sh
cd ui && npm install && npm run build && cd ..
pytest
python -m me3.delegation.sample_bank
python -m me3.publication.sample_bank
# optional live worker:
# opencode serve --port 4096
python -m me3.app.main --serve
```

## Phase checks

| Phase | Command |
|---|---|
| 3 conversation | `python -m me3.conversation.sample_bank` |
| 4 tools | `python -m me3.tasks.sample_bank` |
| 5 retrieval | `python -m me3.retrieval.sample_bank` |
| 6 delegation | `python -m me3.delegation.sample_bank` |
| 7 publication | `python -m me3.publication.sample_bank` |
| 8 benchmark | `python -m me3.app.main --benchmark-check` then `fd-benchmark/launch_assistant.py --check` |
| UI | `python -m me3.app.main --serve` |

### fd-benchmark (Phase 8)

Clone the harness beside this tree (not committed here — see [`../README.md`](../README.md)):

```bash
# from voice-agent venv
python -m me3.app.main --benchmark-check
export FDBENCH_PROJECT="$(pwd)"
export FDBENCH_ENTRY="$FDBENCH_PROJECT/scripts/fdbench_assistant_entry.py"
# launcher check
../fd-benchmark/.venv/bin/python ../fd-benchmark/launch_assistant.py --check
# or with this venv (has pipecat):
python ../fd-benchmark/launch_assistant.py --check --project "$(pwd)"
```

Between trials, optional reset (while UI server is up):

```bash
curl -X POST http://127.0.0.1:8780/api/benchmark/reset -H "Authorization: Bearer $ME3_UI_TOKEN"
```

Pass that as fdbench `--reset-cmd`. Isolation: benchmark mode forces mock OpenCode + mock publication.

## Patch notes

See [`docs/PATCH_NOTES.md`](docs/PATCH_NOTES.md). Latest: Phase 8 benchmark.
