"""Retrieval facade: Tavily when keyed, else MockRetrieval."""

from __future__ import annotations

import logging
from typing import Any, Protocol

from me3.config import load_secrets
from me3.retrieval.evidence import EvidenceEnvelope, evidence_to_dict
from me3.retrieval.mock import MockRetrieval
from me3.retrieval.tavily import TavilyClient

logger = logging.getLogger("me3.retrieval.service")


class RetrievalBackend(Protocol):
    def search(self, query: str, **kwargs: Any) -> list[EvidenceEnvelope]: ...

    def extract(self, urls: list[str]) -> list[EvidenceEnvelope]: ...

    def crawl(self, url: str, **kwargs: Any) -> list[EvidenceEnvelope]: ...


def make_retrieval(*, api_key: str | None = None, force_mock: bool = False) -> tuple[RetrievalBackend, str]:
    key = (api_key if api_key is not None else load_secrets().tavily_api_key) or ""
    if force_mock or not key.strip():
        return MockRetrieval(), "mock"
    return TavilyClient(key), "tavily"


class RetrievalService:
    def __init__(self, backend: RetrievalBackend | None = None, *, provider: str | None = None):
        if backend is None:
            backend, provider = make_retrieval()
        self.backend = backend
        self.provider = provider or getattr(backend, "provider", "unknown")

    def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
        q = (query or "").strip()
        if not q:
            raise ValueError("query is required")
        envelopes = self.backend.search(q, **kwargs)
        return {
            "query": q,
            "provider": self.provider,
            "evidence": [evidence_to_dict(e) for e in envelopes],
            "count": len(envelopes),
            "guidance": (
                "Treat results as evidence with citations. Prefer official sources. "
                "If nationwide vs local election questions are ambiguous, ask one clarifying question."
            ),
        }

    def fetch(self, url: str | None = None, urls: list[str] | None = None, **_kwargs: Any) -> dict[str, Any]:
        targets: list[str] = []
        if urls:
            targets.extend(str(u) for u in urls if u)
        if url:
            targets.append(str(url))
        targets = [u.strip() for u in targets if u and u.strip()]
        if not targets:
            raise ValueError("url or urls required")
        envelopes = self.backend.extract(targets)
        return {
            "provider": self.provider,
            "evidence": [evidence_to_dict(e) for e in envelopes],
            "count": len(envelopes),
        }

    def crawl(self, url: str, **kwargs: Any) -> dict[str, Any]:
        u = (url or "").strip()
        if not u:
            raise ValueError("url is required")
        envelopes = self.backend.crawl(u, **kwargs)
        return {
            "provider": self.provider,
            "root_url": u,
            "evidence": [evidence_to_dict(e) for e in envelopes],
            "count": len(envelopes),
        }
