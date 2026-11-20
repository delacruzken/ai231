# Patch: 2026-11-20_asr-final-on-silence-commit

| Field | Value |
|---|---|
| ID | `2026-11-20_asr-final-on-silence-commit` |
| Date | 2026-11-20 |
| Phase | 8 follow-up / fdbench scoring |
| Verdict | `improved` (unit); live fdbench smoke pending |
| Related decisions | #21 |

## Challenge

Laptop `fdbench run --connect direct` smoke (`2026-11-20_08-51-10`) and aborted full bank (`2026-11-20_08-53-33`) completed turns with E2E/TTS marks, but **`course.wer` / `asr_final` stayed n=0**. Marks showed `asr_partial` + speculative `llm_request` only — never `asr_final`.

## Change

- On speculative silence-commit, emit `mark("asr_final", text=…)` **before** `llm_request` so fdbench can pair WER.
- `StreamingAsrEngine.force_finalize(emit=False)` and call it when LISTENING → THINKING so Zipformer does not sit waiting for endpoint after ASR is echo-gated.

## Hypothesis

ME3 turn-taking commits at `silence_normal_ms` (~700 ms) / incomplete (~1500 ms), while sherpa endpoint rules need ~1.8–2.8 s of trailing silence **in the stream**. After commit, THINKING/SPEAKING gates ASR, so `is_endpoint` never fires. Publishing the committed partial as `asr_final` matches how the agent actually answers and unblocks course WER.

## How measured

- Smoke/full marks + `stage_latency.json` (before): `asr_final` count 0
- `pytest tests/test_conversation_unit.py` — speculative commit now asserts `asr_final` before speculative `llm_request`
- Follow-up: re-run fdbench smoke (`--phrases 3 --runs 1 --rounds 0`) and confirm `course.wer` n&gt;0

## Before / after

- **Before:** partials + speculative LLM; WER/ASR final empty; report “assistant sent no stage marks”
- **After (unit):** silence commit marks `asr_final` then speculative `llm_request`
- **Not claimed:** WER &lt; 15% on a full bank

## Follow-up

Re-run smoke then full stage-latency bank. Keep high sherpa endpoint rules for mid-phrase incomplete patience; do not lower them just to chase marks.
