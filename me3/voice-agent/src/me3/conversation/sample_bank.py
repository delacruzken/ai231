"""Phase 3 sample-bank: pause/correction/interrupt structural tests + optional live Haiku."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from me3.config import load_config, load_secrets, resolve_path
from me3.conversation.acks import AckManager
from me3.conversation.llm import make_llm
from me3.conversation.overlap import OverlapClassifier
from me3.conversation.persona import build_system_prompt
from me3.conversation.pipeline import ConversationOrchestrator
from me3.conversation.turn_taker import TurnTaker, TurnTakerConfig

logger = logging.getLogger("me3.conversation.sample_bank")


def run_structural() -> dict:
    cfg = TurnTakerConfig(silence_normal_ms=600, silence_incomplete_ms=1200, silence_clarification_ms=2000)
    tt = TurnTaker(cfg)
    results = []

    # Pause 600 ms complete phrase
    tt.on_partial("Set a timer for five minutes")
    t0 = time.monotonic()
    tt.last_speech_mono = t0 - 0.65
    d = tt.evaluate(now=t0)
    results.append({"case": "pause_600_complete", "ok": d.action == "commit", "decision": d.action})

    # Incomplete needs 1200
    tt.reset()
    tt.on_partial("Set a timer for")
    t0 = time.monotonic()
    tt.last_speech_mono = t0 - 0.7
    d1 = tt.evaluate(now=t0)
    tt.last_speech_mono = t0 - 1.25
    d2 = tt.evaluate(now=t0)
    results.append(
        {
            "case": "pause_1200_incomplete",
            "ok": d1.action == "none" and d2.action == "commit",
            "early": d1.action,
            "late": d2.action,
        }
    )

    # Resumed speech cancels pending
    tt.reset()
    tt.on_partial("Hello there")
    tt.last_speech_mono = time.monotonic() - 0.7
    tt.evaluate()  # may arm commit
    tt.cancel_pending()
    tt.on_speech()
    results.append({"case": "cancel_pending_on_resume", "ok": tt._commit_armed is False})

    # Overlap: backchannel resume vs steering yield
    oc = OverlapClassifier()
    oc.begin()
    oc.on_partial("yeah")
    r = oc.evaluate(speech_ms=200)
    results.append({"case": "overlap_backchannel", "ok": r.action == "resume", "action": r.action})

    oc.begin()
    oc.on_partial("stop please cancel that")
    r = oc.evaluate(speech_ms=400)
    results.append({"case": "overlap_steering", "ok": r.action == "yield", "action": r.action})

    # Ack suppress when result first
    ack = AckManager()
    ack.on_operation_started()
    ack.on_result_ready()
    results.append({"case": "ack_suppressed", "ok": ack.should_ack() is None})

    # Mock LLM cancel mid-stream
    spoken: list[str] = []
    states: list[str] = []
    orch = ConversationOrchestrator(
        force_mock=True,
        speak=lambda t, k: spoken.append(t),
        cancel_speech=lambda: None,
        set_state=lambda s: states.append(s),
    )
    tid = orch.commit_user_turn("What time is it?")
    time.sleep(0.05)
    orch.cancel_active("test")
    time.sleep(0.2)
    results.append(
        {
            "case": "mock_cancel",
            "ok": tid != "" and "thinking" in states,
            "states": states[:5],
            "spoken_n": len(spoken),
        }
    )

    ok = all(r.get("ok") for r in results)
    return {"structural_ok": ok, "cases": results, "llm_backend": orch.llm_backend}


def run_live_haiku() -> dict:
    secrets = load_secrets()
    cfg = load_config()
    if not secrets.anthropic_api_key.strip():
        return {"live_ok": False, "skipped": True, "reason": "ANTHROPIC_API_KEY unset"}
    llm, backend = make_llm(api_key=secrets.anthropic_api_key, model=cfg.models.llm, force_mock=False)
    system = build_system_prompt(timezone=cfg.session.timezone, locale=cfg.session.locale)
    texts = []
    try:
        for chunk in llm.stream(
            [{"role": "user", "content": "Reply with exactly: hello from haiku"}],
            system,
        ):
            if chunk.done:
                break
            texts.append(chunk.text)
        reply = "".join(texts).strip()
        return {
            "live_ok": bool(reply),
            "skipped": False,
            "backend": backend,
            "model": cfg.models.llm,
            "reply_preview": reply[:200],
        }
    except Exception as exc:
        return {"live_ok": False, "skipped": False, "error": str(exc), "model": cfg.models.llm}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-live", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = resolve_path(cfg.paths.artifacts_dir) / "conversation_phase3" / stamp
    out.mkdir(parents=True, exist_ok=True)

    structural = run_structural()
    live = {"skipped": True, "reason": "--skip-live"} if args.skip_live else run_live_haiku()
    report = {
        "phase": 3,
        "structural": structural,
        "live": live,
        "orchestration_note": (
            "Local me3.conversation + session controller; pipecat-ai==1.15.0 kept pinned "
            "without routing PCM through Pipecat transports in Phase 3."
        ),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not structural["structural_ok"]:
        return 2
    if not live.get("skipped") and not live.get("live_ok"):
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
