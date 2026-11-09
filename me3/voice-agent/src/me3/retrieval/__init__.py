"""Web retrieval via Tavily (Phase 5) with MockRetrieval fallback."""

from me3.retrieval.evidence import EvidenceEnvelope, evidence_to_dict
from me3.retrieval.service import RetrievalService, make_retrieval

__all__ = [
    "EvidenceEnvelope",
    "evidence_to_dict",
    "RetrievalService",
    "make_retrieval",
]
