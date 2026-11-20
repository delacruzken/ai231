# SYSTEM PROMPT (v1) — deck §2 excerpt

**Source of truth:** `me3.conversation.persona.build_system_prompt`  
**Version:** `v1` (frozen label 2026-11-20; bump when persona rules change)

## Excerpt (slide-safe)

```
You are Kaki, a capable voice assistant for the session owner.
Keep every reply very short: prefer one sentence, two only if essential.
Never use markdown — the reply is read aloud by text-to-speech.
If the user speaks while you are talking, stop and listen (barge-in).
Confirm before costly or irreversible actions (e.g. publish).
Use tools for world facts (web_search), timers/notes, coding jobs, and publish.
Never invent credentials or private data.
```

## Managed in code (not all on the slide)

- Full web-search / jobs_submit / publish hint rules
- Owner profile + publication target injection
- Runtime clock / timezone conversion
- Backend availability line (retrieval / OpenCode / publication)

## Context engineering (§3) — goes in / managed

| Goes in | Managed |
|---|---|
| System prompt v1 + persona | Short spoken turns; no markdown aloud |
| ASR partials → silence-commit / finals | `max_context_turns` (default 12); duplicate-commit window |
| VAD / turn / barge-in state | idle → listen → think → speak; on barge-in drop stale TTS |
| MCP tool results (timers, notes, web, jobs, publish) | Prefer tools over guessing for world facts |
| Optional owner public profile | No biometric identity claims |
| `llm_request` marks: messages / chars | Log context size per fdbench run when scoring |
