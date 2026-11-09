"""Delegation MCP: jobs.submit/status/update/cancel/result (Phase 6)."""

from __future__ import annotations

from me3.config import load_secrets
from me3.delegation.service import DelegationService
from me3.mcp.common import RequestMeta, ToolResult, unsupported

TOOLS = (
    "jobs.submit",
    "jobs.status",
    "jobs.update",
    "jobs.cancel",
    "jobs.result",
)

_service: DelegationService | None = None


def get_service() -> DelegationService:
    global _service
    if _service is None:
        secrets = load_secrets()
        force_mock = bool(secrets.opencode_force_mock)
        _service = DelegationService(force_mock=force_mock or None)
    return _service


def handle(tool: str, meta: RequestMeta | None = None, **kwargs) -> ToolResult:
    meta = meta or RequestMeta()
    if tool not in TOOLS:
        return unsupported(meta, tool)
    return get_service().handle(tool, meta=meta, **kwargs)


def main() -> None:
    print("me3 delegation mcp — OpenCode/mock one-job queue ready", flush=True)


if __name__ == "__main__":
    main()
