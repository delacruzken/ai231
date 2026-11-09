"""Deterministic retrieval for structural tests without Tavily spend."""

from __future__ import annotations

from me3.retrieval.evidence import EvidenceEnvelope, truncate


class MockRetrieval:
    provider = "mock"

    def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        search_depth: str = "basic",
        include_answer: bool = False,
        topic: str | None = None,
    ) -> list[EvidenceEnvelope]:
        q = (query or "").lower()
        if "senate" in q or "philippines" in q or "election" in q:
            return [
                EvidenceEnvelope(
                    url="https://example.test/comelec/national-senate",
                    title="Mock COMELEC — national Senate canvass (illustrative)",
                    content=truncate(
                        "MOCK EVIDENCE ONLY. National Senate rankings differ from local "
                        "vote leaders. Treat as fixture text for ambiguity tests; not real results."
                    ),
                    snippet="National vs local Senate results are different races.",
                    published_at="2025-05-20",
                    provider="mock",
                    meta={"fixture": "senate_national", "topic": topic},
                ),
                EvidenceEnvelope(
                    url="https://example.test/news/local-senate-leaders",
                    title="Mock local coverage — provincial Senate vote leaders",
                    content=truncate(
                        "MOCK EVIDENCE ONLY. Local media may report provincial vote leaders "
                        "that are not identical to nationwide winners."
                    ),
                    snippet="Clarify nationwide winners vs local vote leaders.",
                    published_at="2025-05-19",
                    provider="mock",
                    meta={"fixture": "senate_local"},
                ),
            ][:max_results]
        return [
            EvidenceEnvelope(
                url="https://example.test/search",
                title=f"Mock result for: {query[:80]}",
                content=truncate(
                    f"MOCK EVIDENCE for query={query!r}. search_depth={search_depth}. "
                    "Cite this URL if you must demonstrate grounding; it is not real web content."
                ),
                snippet=f"Mock snippet for {query[:60]}",
                provider="mock",
                meta={"include_answer": include_answer},
            )
        ][:max_results]

    def extract(self, urls: list[str]) -> list[EvidenceEnvelope]:
        out: list[EvidenceEnvelope] = []
        for u in urls[:5]:
            out.append(
                EvidenceEnvelope(
                    url=u,
                    title=f"Mock extract: {u}",
                    content=truncate(f"MOCK extracted body from {u}."),
                    snippet=f"Mock extract of {u}",
                    provider="mock",
                )
            )
        return out

    def crawl(
        self,
        url: str,
        *,
        max_depth: int = 1,
        limit: int = 10,
        instructions: str | None = None,
    ) -> list[EvidenceEnvelope]:
        pages = min(limit, 3)
        return [
            EvidenceEnvelope(
                url=f"{url.rstrip('/')}/page-{i}",
                title=f"Mock crawl page {i} from {url}",
                content=truncate(
                    f"MOCK crawl depth<={max_depth} instructions={instructions!r} page={i}."
                ),
                snippet=f"Crawl page {i}",
                provider="mock",
                meta={"crawl": True},
            )
            for i in range(1, pages + 1)
        ]
