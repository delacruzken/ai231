"""Tavily HTTP client (search / extract / crawl)."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any

from me3.retrieval.evidence import EvidenceEnvelope, truncate

logger = logging.getLogger("me3.retrieval.tavily")

API_BASE = "https://api.tavily.com"


class TavilyClient:
    def __init__(self, api_key: str, *, timeout_s: float = 30.0):
        self.api_key = api_key.strip()
        self.timeout_s = timeout_s

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        payload = dict(body)
        # Search/extract accept api_key in JSON; crawl (and newer endpoints)
        # require Authorization: Bearer. Send both for compatibility.
        payload.setdefault("api_key", self.api_key)
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{API_BASE}{path}",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            logger.warning("tavily HTTP %s %s: %s", path, exc.code, detail)
            raise RuntimeError(f"Tavily {path} HTTP {exc.code}: {detail}") from exc
        except Exception as exc:
            logger.exception("tavily request failed path=%s", path)
            raise RuntimeError(f"Tavily {path} failed: {exc}") from exc

    def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        search_depth: str = "basic",
        include_answer: bool = False,
        topic: str | None = None,
    ) -> list[EvidenceEnvelope]:
        body: dict[str, Any] = {
            "query": query,
            "max_results": max(1, min(int(max_results), 10)),
            "search_depth": search_depth,
            "include_answer": include_answer,
        }
        if topic:
            body["topic"] = topic
        raw = self._post("/search", body)
        out: list[EvidenceEnvelope] = []
        for item in raw.get("results") or []:
            out.append(
                EvidenceEnvelope(
                    url=str(item.get("url") or ""),
                    title=str(item.get("title") or ""),
                    content=truncate(str(item.get("content") or "")),
                    snippet=truncate(str(item.get("content") or ""), 400),
                    published_at=item.get("published_date") or item.get("published_at"),
                    provider="tavily",
                    meta={"score": item.get("score"), "answer": raw.get("answer")},
                )
            )
        return out

    def extract(self, urls: list[str]) -> list[EvidenceEnvelope]:
        if not urls:
            return []
        raw = self._post("/extract", {"urls": urls[:5]})
        out: list[EvidenceEnvelope] = []
        for item in raw.get("results") or []:
            out.append(
                EvidenceEnvelope(
                    url=str(item.get("url") or ""),
                    title=str(item.get("title") or ""),
                    content=truncate(str(item.get("raw_content") or item.get("content") or ""), 4000),
                    snippet=truncate(str(item.get("raw_content") or item.get("content") or ""), 400),
                    provider="tavily",
                    meta={"images": item.get("images")},
                )
            )
        for fail in raw.get("failed_results") or []:
            out.append(
                EvidenceEnvelope(
                    url=str(fail.get("url") or ""),
                    error=str(fail.get("error") or "extract failed"),
                    provider="tavily",
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
        body: dict[str, Any] = {
            "url": url,
            "max_depth": max(1, min(int(max_depth), 2)),
            "limit": max(1, min(int(limit), 20)),
        }
        if instructions:
            body["instructions"] = instructions
        raw = self._post("/crawl", body)
        out: list[EvidenceEnvelope] = []
        for item in raw.get("results") or []:
            out.append(
                EvidenceEnvelope(
                    url=str(item.get("url") or ""),
                    title=str(item.get("title") or ""),
                    content=truncate(str(item.get("raw_content") or item.get("content") or ""), 4000),
                    snippet=truncate(str(item.get("raw_content") or item.get("content") or ""), 400),
                    provider="tavily",
                    meta={"crawl": True},
                )
            )
        return out
