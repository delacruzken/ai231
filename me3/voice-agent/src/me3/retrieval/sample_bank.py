"""Phase 5 sample bank: mock evidence + optional live Tavily + profile/ambiguity."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from me3.config import ROOT, load_config, load_secrets
from me3.conversation.llm import MockLLM
from me3.conversation.persona import build_system_prompt
from me3.mcp.common import RequestMeta
from me3.profile.owner import OwnerProfileStore
from me3.retrieval.service import RetrievalService, make_retrieval
from me3.retrieval.tavily import TavilyClient
from me3.tasks.service import TaskService
from me3.tasks.tools import dispatch_tool


def run_structural(out: Path) -> dict:
    db = out / "sample.db"
    svc = TaskService(db_path=db, force_mock_retrieval=True)
    cases: list[dict] = []

    search = svc.handle("web.search", RequestMeta(idempotency_key="p5-s1"), query="Philippines Senate election")
    evidence = search.data.get("evidence") or []
    cases.append(
        {
            "case": "mock_search_citations",
            "ok": search.ok and len(evidence) >= 2 and all(e.get("url") for e in evidence),
            "count": len(evidence),
            "provider": search.data.get("provider"),
        }
    )

    fetch = svc.handle("web.fetch", RequestMeta(), url=evidence[0]["url"] if evidence else "https://example.test")
    cases.append({"case": "mock_fetch", "ok": fetch.ok and fetch.data.get("count", 0) >= 1})

    crawl = svc.handle("web.crawl", RequestMeta(), url="https://example.test/docs", limit=2, max_depth=1)
    cases.append({"case": "mock_crawl_bounded", "ok": crawl.ok and 1 <= crawl.data.get("count", 0) <= 3})

    store = OwnerProfileStore(db)
    before = store.get()
    store.set_public({"display_name": "ME3 Owner", "headline": "Demo"}, approved=True)
    after = store.get()
    prompt = build_system_prompt(owner_profile=after, retrieval_provider="mock")
    cases.append(
        {
            "case": "owner_profile_inject",
            "ok": before["approved"] is False and after["approved"] is True and "ME3 Owner" in prompt,
        }
    )

    llm = MockLLM()

    def executor(name, args):
        return dispatch_tool(svc, name, args, turn_id="p5-amb")

    chunks = list(
        llm.stream(
            [{"role": "user", "content": "Who are the Senate winners in the Philippines?"}],
            system=prompt,
            turn_id="p5-amb",
            tools_enabled=True,
            tool_executor=executor,
        )
    )
    reply = "".join(c.text for c in chunks if c.text)
    cases.append(
        {
            "case": "ambiguity_clarify_cue",
            "ok": ("nationwide" in reply.lower() or "local" in reply.lower()) and bool(chunks[0].tool_results),
            "reply": reply[:200],
        }
    )

    store.close()
    svc.close()
    return {"structural_ok": all(c["ok"] for c in cases), "cases": cases}


def run_live() -> dict:
    secrets = load_secrets()
    key = secrets.tavily_api_key.strip()
    if not key:
        return {"live_ok": False, "skipped": True, "reason": "TAVILY_API_KEY unset"}
    try:
        client = TavilyClient(key)
        results = client.search("Tavily API documentation", max_results=2)
        ok = len(results) >= 1 and all(r.url for r in results)
        return {
            "live_ok": ok,
            "skipped": False,
            "provider": "tavily",
            "count": len(results),
            "titles": [r.title for r in results][:3],
            "urls": [r.url for r in results][:3],
        }
    except Exception as exc:
        return {"live_ok": False, "skipped": False, "error": str(exc)[:300]}


def run(out_dir: Path | None = None) -> dict:
    cfg = load_config()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = out_dir or (ROOT / "artifacts" / "retrieval_phase5" / stamp)
    out.mkdir(parents=True, exist_ok=True)
    structural = run_structural(out)
    live = run_live()
    backend, provider = make_retrieval()
    report = {
        "phase": 5,
        "structural": structural,
        "live": live,
        "default_provider": provider,
        "timezone": cfg.session.timezone,
        "verdict_hint": (
            "improved"
            if structural["structural_ok"] and live.get("live_ok")
            else ("inconclusive" if structural["structural_ok"] and live.get("skipped") else "blocked")
        ),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    run()


if __name__ == "__main__":
    main()
