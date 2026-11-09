"""Tasks MCP: timers, notes, and web retrieval (Phase 4–5)."""

from __future__ import annotations

from me3.config import load_config, load_secrets
from me3.mcp.common import RequestMeta, ToolResult, unsupported
from me3.tasks.service import TaskService

TIMER_NOTE_TOOLS = (
    "timer.create",
    "timer.list",
    "timer.update",
    "timer.cancel",
    "notes.add",
    "notes.search",
    "notes.edit",
    "notes.delete",
)

WEB_TOOLS = (
    "web.search",
    "web.fetch",
    "web.crawl",
)

TOOLS = TIMER_NOTE_TOOLS + WEB_TOOLS

_service: TaskService | None = None


def get_service() -> TaskService:
    global _service
    if _service is None:
        cfg = load_config()
        secrets = load_secrets()
        force_mock = not bool(secrets.tavily_api_key.strip())
        _service = TaskService(
            timezone_name=cfg.session.timezone,
            force_mock_retrieval=force_mock,
        )
    return _service


def handle(tool: str, meta: RequestMeta | None = None, **kwargs) -> ToolResult:
    meta = meta or RequestMeta()
    if tool not in TOOLS:
        return unsupported(meta, tool)
    return get_service().handle(tool, meta=meta, **kwargs)


def main() -> None:
    print("me3 tasks mcp — timers/notes/web retrieval ready", flush=True)


if __name__ == "__main__":
    main()
