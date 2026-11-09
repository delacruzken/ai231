"""Shared MCP helpers: request IDs, error shapes, idempotency stubs."""

from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class ErrorKind(str, Enum):
    RETRYABLE = "retryable"
    NEEDS_INPUT = "needs_input"
    NEEDS_APPROVAL = "needs_approval"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    UNKNOWN_OUTCOME = "unknown_outcome"


class RequestMeta(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str | None = None
    turn_id: str | None = None
    utterance_id: str | None = None
    job_id: str | None = None
    idempotency_key: str | None = None


class ToolError(BaseModel):
    kind: ErrorKind
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    ok: bool
    request_id: str
    data: dict[str, Any] = Field(default_factory=dict)
    error: ToolError | None = None


def unsupported(meta: RequestMeta, tool: str) -> ToolResult:
    return ToolResult(
        ok=False,
        request_id=meta.request_id,
        error=ToolError(
            kind=ErrorKind.UNSUPPORTED,
            message=f"{tool} is a Phase 0 stub; implementation lands in a later phase",
        ),
    )
