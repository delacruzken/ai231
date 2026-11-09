"""Anthropic tool schemas and dispatcher for Phase 4 timer/note tools."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from typing import TYPE_CHECKING

from me3.mcp.common import RequestMeta
from me3.tasks.service import TaskService

if TYPE_CHECKING:
    from me3.delegation.service import DelegationService
    from me3.publication.service import PublicationService

# Anthropic tool definitions (snake names map to MCP tool ids)
TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "timer_create",
        "description": "Create a countdown timer. Prefer duration_seconds for relative timers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "label": {"type": "string", "description": "Short label for the timer"},
                "duration_seconds": {
                    "type": "integer",
                    "description": "Seconds until the timer fires",
                },
                "duration": {
                    "type": "string",
                    "description": "Human duration like '5 minutes' if seconds unknown",
                },
            },
        },
    },
    {
        "name": "timer_list",
        "description": "List active timers.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "timer_cancel",
        "description": "Cancel an active timer by id.",
        "input_schema": {
            "type": "object",
            "properties": {"timer_id": {"type": "string"}},
            "required": ["timer_id"],
        },
    },
    {
        "name": "notes_add",
        "description": "Save a persistent note.",
        "input_schema": {
            "type": "object",
            "properties": {
                "body": {"type": "string"},
                "title": {"type": "string"},
            },
            "required": ["body"],
        },
    },
    {
        "name": "notes_search",
        "description": "Search saved notes by text.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
        },
    },
    {
        "name": "notes_edit",
        "description": "Edit an existing note by id.",
        "input_schema": {
            "type": "object",
            "properties": {
                "note_id": {"type": "string"},
                "body": {"type": "string"},
                "title": {"type": "string"},
            },
            "required": ["note_id"],
        },
    },
    {
        "name": "notes_delete",
        "description": "Soft-delete a note by id.",
        "input_schema": {
            "type": "object",
            "properties": {"note_id": {"type": "string"}},
            "required": ["note_id"],
        },
    },
    {
        "name": "web_search",
        "description": (
            "Search the web via Tavily. Returns evidence envelopes with URL/title/content. "
            "Cite sources; do not treat snippets as authority. "
            "If results are ambiguous or conflict, ask one short clarifying question."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer"},
                "topic": {"type": "string", "description": "Optional Tavily topic e.g. news"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "web_fetch",
        "description": "Extract readable content from one or more URLs (evidence envelopes).",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "urls": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
    {
        "name": "web_crawl",
        "description": "Bounded crawl from a root URL when linked exploration is needed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "max_depth": {"type": "integer"},
                "limit": {"type": "integer"},
                "instructions": {"type": "string"},
            },
            "required": ["url"],
        },
    },
    {
        "name": "jobs_submit",
        "description": (
            "Submit a coding/build job to the OpenCode worker. Call this whenever the owner "
            "wants a site, website, landing page, app, or coding job built/created/started/"
            "submitted, or says proceed/go ahead after agreeing on such work. Only one active "
            "job is allowed — if busy, report that instead of pretending to submit. Returns a "
            "job id promptly; do not claim the work is finished until jobs_status says completed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "specification": {
                    "type": "string",
                    "description": (
                        "Clear build brief for the worker (what to build, key pages/features, "
                        "and any owner constraints). Reconstruct from messy ASR if needed."
                    ),
                },
                "instructions": {"type": "string"},
            },
            "required": ["specification"],
        },
    },
    {
        "name": "jobs_status",
        "description": (
            "Get the active job (if any) and recent jobs including completed/failed ones. "
            "Jobs persist across chat restarts — use this when the owner asks what finished "
            "or what was previously requested. Do not assume 'no jobs' without calling this."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"job_id": {"type": "string"}},
        },
    },
    {
        "name": "jobs_update",
        "description": "Respond to needs_input/needs_approval on a job.",
        "input_schema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string"},
                "response": {"type": "string"},
                "approve": {"type": "boolean"},
            },
            "required": ["job_id"],
        },
    },
    {
        "name": "jobs_cancel",
        "description": "Cancel an active job.",
        "input_schema": {
            "type": "object",
            "properties": {"job_id": {"type": "string"}},
            "required": ["job_id"],
        },
    },
    {
        "name": "jobs_result",
        "description": "Fetch job outcome, artifacts, and revisions.",
        "input_schema": {
            "type": "object",
            "properties": {"job_id": {"type": "string"}},
            "required": ["job_id"],
        },
    },
    {
        "name": "publish_validate",
        "description": (
            "Validate a site folder before GitHub Pages publish. "
            "source_dir is optional — if omitted, uses the latest completed job workspace "
            "that contains index.html (or job_id / hint to pick one). "
            "Rejects .env, keys, tokens, and other private files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "source_dir": {
                    "type": "string",
                    "description": "Optional local folder; auto-resolved from completed jobs when omitted",
                },
                "job_id": {"type": "string", "description": "Optional completed job id to publish"},
                "hint": {
                    "type": "string",
                    "description": "Optional keyword to match a job (e.g. games, portfolio)",
                },
            },
        },
    },
    {
        "name": "publish_site",
        "description": (
            "Publish a site to the owner's GitHub Pages docs root. "
            "source_dir is optional — omit it to auto-use the matching completed job workspace. "
            "Requires confirm=true after owner approval. Never publish secrets. "
            "Do not ask the owner for a folder path when a completed job workspace exists."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "source_dir": {
                    "type": "string",
                    "description": "Optional; auto-resolved from completed jobs when omitted",
                },
                "job_id": {"type": "string"},
                "hint": {"type": "string", "description": "Keyword to pick among jobs, e.g. games"},
                "message": {"type": "string", "description": "Commit message"},
                "confirm": {
                    "type": "boolean",
                    "description": "Must be true to perform the live commit",
                },
                "expect_substring": {
                    "type": "string",
                    "description": "Optional text that should appear on the live Pages URL",
                },
            },
        },
    },
    {
        "name": "publish_status",
        "description": "Check publication provider health, Pages URL, and recent publishes.",
        "input_schema": {
            "type": "object",
            "properties": {"publication_id": {"type": "string"}},
        },
    },
]

_NAME_TO_MCP = {
    "timer_create": "timer.create",
    "timer_list": "timer.list",
    "timer_cancel": "timer.cancel",
    "notes_add": "notes.add",
    "notes_search": "notes.search",
    "notes_edit": "notes.edit",
    "notes_delete": "notes.delete",
    "web_search": "web.search",
    "web_fetch": "web.fetch",
    "web_crawl": "web.crawl",
    "jobs_submit": "jobs.submit",
    "jobs_status": "jobs.status",
    "jobs_update": "jobs.update",
    "jobs_cancel": "jobs.cancel",
    "jobs_result": "jobs.result",
    "publish_validate": "publish.validate",
    "publish_site": "publish.site",
    "publish_status": "publish.status",
}


def dispatch_tool(
    service: TaskService,
    name: str,
    arguments: dict[str, Any],
    *,
    turn_id: str | None = None,
    idempotency_key: str | None = None,
    delegation: "DelegationService | None" = None,
    publication: "PublicationService | None" = None,
) -> dict[str, Any]:
    mcp_name = _NAME_TO_MCP.get(name)
    if not mcp_name:
        return {"ok": False, "error": f"unknown tool: {name}"}
    key = idempotency_key or f"{turn_id or 'x'}:{name}:{uuid4()}"
    # Stable idempotency when caller supplies one; otherwise unique per call
    if idempotency_key is None and turn_id:
        # Deterministic within a turn for identical tool+args retries
        import hashlib
        import json

        digest = hashlib.sha256(
            f"{turn_id}|{name}|{json.dumps(arguments, sort_keys=True)}".encode()
        ).hexdigest()[:24]
        key = f"turn:{digest}"

    meta = RequestMeta(idempotency_key=key, turn_id=turn_id)
    if mcp_name.startswith("jobs."):
        if delegation is None:
            return {"ok": False, "error": "delegation service unavailable"}
        result = delegation.handle(mcp_name, meta=meta, **(arguments or {}))
    elif mcp_name.startswith("publish."):
        if publication is None:
            return {"ok": False, "error": "publication service unavailable"}
        result = publication.handle(mcp_name, meta=meta, **(arguments or {}))
    else:
        result = service.handle(mcp_name, meta=meta, **(arguments or {}))
    if result.ok:
        return {"ok": True, "tool": mcp_name, **result.data}
    err = result.error.message if result.error else "failed"
    return {"ok": False, "tool": mcp_name, "error": err}
