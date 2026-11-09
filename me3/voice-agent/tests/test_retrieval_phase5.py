"""Phase 5: evidence envelopes, mock retrieval, owner profile, ambiguity cue."""

from __future__ import annotations

from pathlib import Path

from me3.conversation.llm import MockLLM
from me3.conversation.persona import build_system_prompt
from me3.mcp.common import RequestMeta
from me3.profile.owner import OwnerProfileStore, format_owner_for_prompt
from me3.retrieval.mock import MockRetrieval
from me3.retrieval.service import RetrievalService, make_retrieval
from me3.tasks.service import TaskService
from me3.tasks.tools import dispatch_tool


def test_make_retrieval_defaults_to_mock_without_key(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "")
    backend, provider = make_retrieval(api_key="", force_mock=False)
    assert provider == "mock"
    assert isinstance(backend, MockRetrieval)


def test_mock_search_evidence_envelope():
    svc = RetrievalService(MockRetrieval(), provider="mock")
    out = svc.search("Philippines Senate results")
    assert out["count"] >= 2
    for e in out["evidence"]:
        assert e["url"] and e["retrieved_at"] and e["provider"] == "mock"
    assert "nationwide" in out["guidance"].lower() or "local" in out["guidance"].lower()


def test_task_service_web_tools(tmp_path: Path):
    svc = TaskService(db_path=tmp_path / "r.db", force_mock_retrieval=True)
    search = svc.handle("web.search", RequestMeta(), query="hello world")
    assert search.ok and search.data["count"] >= 1
    url = search.data["evidence"][0]["url"]
    fetched = svc.handle("web.fetch", RequestMeta(), url=url)
    assert fetched.ok and fetched.data["count"] >= 1
    crawled = svc.handle("web.crawl", RequestMeta(), url="https://example.test", limit=2)
    assert crawled.ok and crawled.data["count"] >= 1
    svc.close()


def test_owner_profile_approved_only(tmp_path: Path):
    store = OwnerProfileStore(tmp_path / "p.db")
    empty = store.get()
    assert empty["approved"] is False
    assert "not approved" in format_owner_for_prompt(empty).lower()
    store.set_public(
        {"display_name": "Ada", "headline": "Builder", "links": ["https://example.test"]},
        approved=True,
    )
    got = store.get()
    assert got["approved"] is True
    assert got["public"]["display_name"] == "Ada"
    prompt = build_system_prompt(owner_profile=got, retrieval_provider="mock")
    assert "Ada" in prompt
    assert "biometric" not in prompt.lower() or "Do not claim biometric" in prompt
    store.close()


def test_mock_llm_senate_ambiguity(tmp_path: Path):
    svc = TaskService(db_path=tmp_path / "llm.db", force_mock_retrieval=True)
    llm = MockLLM()

    def executor(name, args):
        return dispatch_tool(svc, name, args, turn_id="p5")

    chunks = list(
        llm.stream(
            [{"role": "user", "content": "Who won the Philippine Senate race?"}],
            system="test",
            turn_id="p5",
            tools_enabled=True,
            tool_executor=executor,
        )
    )
    text = "".join(c.text for c in chunks if c.text).lower()
    assert "nationwide" in text or "local" in text
    assert any(c.tool_results for c in chunks)
    svc.close()
