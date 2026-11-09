"""Evidence envelopes — web text is evidence, never authority."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class EvidenceEnvelope:
    url: str
    title: str = ""
    content: str = ""
    snippet: str = ""
    retrieved_at: str = ""
    published_at: str | None = None
    error: str | None = None
    provider: str = "unknown"
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.retrieved_at:
            self.retrieved_at = datetime.now(timezone.utc).isoformat()


def evidence_to_dict(e: EvidenceEnvelope) -> dict[str, Any]:
    return asdict(e)


def truncate(text: str, limit: int = 2400) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"
