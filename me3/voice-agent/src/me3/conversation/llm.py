"""Anthropic Haiku streaming + MockLLM with Phase 4 tool use."""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator
from uuid import uuid4

from me3.tasks.tools import TOOL_DEFINITIONS, dispatch_tool

logger = logging.getLogger("me3.conversation.llm")

ToolExecutor = Callable[[str, dict[str, Any]], dict[str, Any]]


@dataclass
class LlmChunk:
    turn_id: str
    text: str
    done: bool = False
    tool_results: list[dict[str, Any]] = field(default_factory=list)


class BaseLLM:
    def stream(
        self,
        messages: list[dict],
        system: str,
        turn_id: str | None = None,
        *,
        tools_enabled: bool = False,
        tool_executor: ToolExecutor | None = None,
    ) -> Iterator[LlmChunk]:
        raise NotImplementedError

    def cancel(self, turn_id: str) -> None:
        raise NotImplementedError


class MockLLM(BaseLLM):
    """Deterministic replies + regex tool routing for structural tests."""

    def __init__(self):
        self._cancel: set[str] = set()
        self._lock = threading.Lock()

    def cancel(self, turn_id: str) -> None:
        with self._lock:
            self._cancel.add(turn_id)

    def stream(
        self,
        messages: list[dict],
        system: str,
        turn_id: str | None = None,
        *,
        tools_enabled: bool = False,
        tool_executor: ToolExecutor | None = None,
    ) -> Iterator[LlmChunk]:
        tid = turn_id or str(uuid4())
        user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user = m.get("content") or ""
                break

        tool_results: list[dict[str, Any]] = []
        if tools_enabled and tool_executor:
            tool_results = self._maybe_tools(user, tool_executor)
            with self._lock:
                if tid in self._cancel:
                    return
            if tool_results:
                reply = self._reply_from_tools(user, tool_results)
                yield LlmChunk(tid, reply, done=False, tool_results=tool_results)
                yield LlmChunk(tid, "", done=True, tool_results=tool_results)
                return

        reply = f"Got it. You said: {user[:120]}."
        words = reply.split(" ")
        buf = ""
        for i, w in enumerate(words):
            with self._lock:
                if tid in self._cancel:
                    return
            buf = (buf + " " + w).strip()
            if (i + 1) % 4 == 0 or i == len(words) - 1:
                yield LlmChunk(tid, buf if i == len(words) - 1 else buf + " ", done=False)
                if i != len(words) - 1:
                    buf = ""
        yield LlmChunk(tid, "", done=True)

    def _maybe_tools(self, user: str, executor: ToolExecutor) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        lower = user.lower()
        # timer create
        m = re.search(
            r"(?:set|start|create)?\s*(?:a\s+)?timer(?:\s+for)?\s+(\d+)\s*(seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h)\b",
            lower,
        )
        if m:
            n = int(m.group(1))
            unit = m.group(2)
            if unit.startswith("h"):
                secs = n * 3600
            elif unit.startswith("m"):
                secs = n * 60
            else:
                secs = n
            results.append(
                {
                    "name": "timer_create",
                    "result": executor("timer_create", {"label": "timer", "duration_seconds": secs}),
                }
            )
            return results
        if re.search(r"\b(list|show)\b.*\btimers?\b|\btimers?\b.*\b(list|show)\b", lower):
            results.append({"name": "timer_list", "result": executor("timer_list", {})})
            return results
        m = re.search(r"(?:cancel|stop|delete)\s+(?:the\s+)?timer\s+(\S+)", lower)
        if m:
            results.append(
                {"name": "timer_cancel", "result": executor("timer_cancel", {"timer_id": m.group(1)})}
            )
            return results
        m = re.search(
            r"(?:note|remember|save)\s*(?:that|:)?\s+(.+)$",
            user,
            re.I,
        )
        if m and not lower.startswith("set a timer"):
            body = m.group(1).strip()
            if body:
                results.append({"name": "notes_add", "result": executor("notes_add", {"body": body})})
                return results
        m = re.search(r"(?:find|search|show)\s+notes?(?:\s+(?:about|for|with))?\s*(.*)$", lower)
        if m:
            results.append(
                {"name": "notes_search", "result": executor("notes_search", {"query": m.group(1).strip()})}
            )
            return results
        # publish site (before generic "site/website" job routing)
        if re.search(r"\b(publish|deploy)\b.+\b(site|page|pages|github)\b", lower) or re.search(
            r"\bpublish_status\b|\bpages status\b|\bconfirm(?:ed)?\b", lower
        ):
            if "status" in lower and "confirm" not in lower:
                results.append({"name": "publish_status", "result": executor("publish_status", {})})
            else:
                m = re.search(r"(?:from|folder|dir(?:ectory)?)\s+(\S+)", user, re.I)
                source = m.group(1) if m else ""
                confirm = bool(
                    re.search(r"\b(confirm(?:ed)?|approved|go ahead|publish it|push it)\b", lower)
                )
                hint = ""
                if "game" in lower:
                    hint = "game"
                elif "portfolio" in lower or "ken" in lower:
                    hint = "portfolio"
                args: dict[str, Any] = {"confirm": confirm}
                if source:
                    args["source_dir"] = source
                if hint:
                    args["hint"] = hint
                results.append(
                    {
                        "name": "publish_site",
                        "result": executor("publish_site", args),
                    }
                )
            return results
        # delegated coding job (broad: ASR often garbling "submit/build/landing page")
        m = re.search(
            r"(?:build|create|make|delegate|start|submit|spin\s*up)\s+(?:a\s+|an\s+|the\s+)?"
            r"(?:job|site|website|page|landing\s*page|app)\b(.+)$",
            user,
            re.I,
        )
        if m or re.search(
            r"\b(opencode|coding job|delegate|submit(?:\s+a)?\s+job|landing\s*page|"
            r"please\s+proceed|go\s+ahead\s+and\s+(?:build|create|submit))\b",
            lower,
        ):
            spec = (m.group(1).strip() if m else user).strip(" .:") or user
            results.append(
                {
                    "name": "jobs_submit",
                    "result": executor("jobs_submit", {"specification": spec}),
                }
            )
            return results
        # web research
        m = re.search(
            r"(?:search(?:\s+the)?\s+web(?:\s+for)?|look\s+up|research|who\s+won|what\s+are)\s+(.+)$",
            user,
            re.I,
        )
        if m or any(k in lower for k in ("senate", "search for", "google")):
            query = m.group(1).strip() if m else user
            results.append({"name": "web_search", "result": executor("web_search", {"query": query})})
            return results
        return results

    def _reply_from_tools(self, user: str, tool_results: list[dict[str, Any]]) -> str:
        parts = []
        for tr in tool_results:
            r = tr.get("result") or {}
            if not r.get("ok"):
                parts.append(f"That failed: {r.get('error', 'unknown error')}.")
                continue
            name = tr.get("name")
            if name == "timer_create":
                t = r.get("timer") or {}
                parts.append(f"Okay, timer set for {t.get('label', 'you')}.")
            elif name == "timer_list":
                timers = r.get("timers") or []
                parts.append(f"You have {len(timers)} active timer{'s' if len(timers) != 1 else ''}.")
            elif name == "timer_cancel":
                parts.append("Timer cancelled.")
            elif name == "notes_add":
                parts.append("Got it — I saved that note.")
            elif name == "notes_search":
                notes = r.get("notes") or []
                parts.append(f"Found {len(notes)} note{'s' if len(notes) != 1 else ''}.")
            elif name in ("web_search", "web_fetch", "web_crawl"):
                evidence = r.get("evidence") or []
                lower = user.lower()
                if "senate" in lower and len(evidence) >= 2:
                    parts.append(
                        "I found sources, but nationwide Senate winners and local vote leaders "
                        "can differ — which one do you mean?"
                    )
                elif evidence:
                    top = evidence[0]
                    title = top.get("title") or "a source"
                    parts.append(f"Based on {title}, briefly: {(top.get('snippet') or '')[:160]}")
                    if top.get("url"):
                        parts.append(f"Source: {top['url']}")
                else:
                    parts.append("Search returned no usable evidence.")
            elif name == "jobs_submit":
                job = r.get("job") or {}
                parts.append(
                    f"Started job {job.get('id', '')[:8]}. Status {job.get('status', 'queued')}."
                )
            elif name == "jobs_status":
                summary = r.get("summary")
                if summary:
                    parts.append(str(summary))
                else:
                    job = r.get("job") or {}
                    recent = r.get("jobs") or []
                    if job:
                        parts.append(f"Job {job.get('id', '')[:8]} is {job.get('status')}.")
                    elif recent:
                        top = recent[0]
                        parts.append(
                            f"No active job. Most recent: {top.get('status')} — "
                            f"{(top.get('specification') or '')[:80]}"
                        )
                    else:
                        parts.append("No jobs on record.")
            elif name == "jobs_cancel":
                parts.append("Job cancelled.")
            elif name == "publish_validate":
                parts.append(
                    f"Validated {r.get('file_count', 0)} publishable file(s)."
                    if r.get("ok")
                    else "Validation failed — private or disallowed files found."
                )
            elif name == "publish_site":
                pub = r.get("publication") or {}
                pages = r.get("pages") or {}
                url = pages.get("url") or pub.get("pages_url") or ""
                parts.append(
                    f"Published commit {(pub.get('commit_sha') or '')[:8]}. "
                    + (f"Pages: {url}" if url else "Pages check pending.")
                )
            elif name == "publish_status":
                pages = r.get("pages") or {}
                parts.append(
                    f"Publication backend {r.get('provider', 'unknown')}. "
                    f"Pages URL: {pages.get('html_url') or 'unknown'}."
                )
            elif name == "jobs_update":
                job = r.get("job") or {}
                parts.append(f"Updated job — now {job.get('status', 'running')}.")
            elif name == "jobs_result":
                job = r.get("job") or {}
                parts.append(f"Job {job.get('status')}: {(job.get('outcome') or '')[:120]}")
            else:
                parts.append("Done.")
        return " ".join(parts) or "Done."


class AnthropicLLM(BaseLLM):
    def __init__(self, api_key: str, model: str):
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self._cancel: set[str] = set()
        self._lock = threading.Lock()
        self.resolved_model = model

    def cancel(self, turn_id: str) -> None:
        with self._lock:
            self._cancel.add(turn_id)

    def stream(
        self,
        messages: list[dict],
        system: str,
        turn_id: str | None = None,
        *,
        tools_enabled: bool = False,
        tool_executor: ToolExecutor | None = None,
    ) -> Iterator[LlmChunk]:
        tid = turn_id or str(uuid4())
        api_messages: list[dict[str, Any]] = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
            if m["role"] in ("user", "assistant")
        ]
        tool_results_acc: list[dict[str, Any]] = []
        try:
            if tools_enabled and tool_executor:
                # Tool round(s) non-streaming for reliability, then stream final text.
                # Tool rounds need a high max_tokens budget: jobs_submit specifications
                # easily exceed 120 tokens; that limit caused Haiku to skip the tool and
                # only say "I'll submit…" without creating a job.
                for _ in range(4):
                    with self._lock:
                        if tid in self._cancel:
                            return
                    kwargs: dict[str, Any] = {
                        "model": self.model,
                        "max_tokens": 1024,
                        "system": system,
                        "messages": api_messages,
                        "tools": TOOL_DEFINITIONS,
                    }
                    resp = self._client.messages.create(**kwargs)
                    if resp.stop_reason == "tool_use":
                        tool_uses = [b for b in resp.content if b.type == "tool_use"]
                        api_messages.append({"role": "assistant", "content": resp.content})
                        tool_result_blocks = []
                        for tu in tool_uses:
                            with self._lock:
                                if tid in self._cancel:
                                    return
                            out = tool_executor(tu.name, dict(tu.input or {}))
                            tool_results_acc.append({"name": tu.name, "result": out})
                            tool_result_blocks.append(
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tu.id,
                                    "content": json.dumps(out),
                                }
                            )
                        api_messages.append({"role": "user", "content": tool_result_blocks})
                        continue
                    # Final text response — stream chunks from assembled text
                    text = "".join(getattr(b, "text", "") for b in resp.content if hasattr(b, "text"))
                    if not text.strip() and tool_results_acc:
                        text = _spoken_fallback_from_tools(tool_results_acc)
                    if text:
                        yield LlmChunk(tid, text, done=False, tool_results=tool_results_acc)
                    yield LlmChunk(tid, "", done=True, tool_results=tool_results_acc)
                    return

            with self._client.messages.stream(
                model=self.model,
                max_tokens=256,
                system=system,
                messages=api_messages,
            ) as stream:
                for text in stream.text_stream:
                    with self._lock:
                        if tid in self._cancel:
                            logger.info("llm cancelled turn=%s", tid)
                            return
                    yield LlmChunk(tid, text, done=False)
            yield LlmChunk(tid, "", done=True)
        except Exception:
            logger.exception("anthropic stream failed model=%s", self.model)
            raise


def _spoken_fallback_from_tools(tool_results: list[dict[str, Any]]) -> str:
    """Short TTS line when the model returns tools but no spoken text."""
    for tr in tool_results:
        r = tr.get("result") or {}
        name = tr.get("name")
        if not r.get("ok"):
            return f"That failed: {r.get('error', 'unknown error')}."
        if name == "jobs_submit":
            job = r.get("job") or {}
            return f"Started job {str(job.get('id') or '')[:8]}. Status {job.get('status', 'queued')}."
        if name == "jobs_status":
            summary = r.get("summary")
            if summary:
                return str(summary)
            job = r.get("job") or {}
            if job:
                return f"Job {str(job.get('id') or '')[:8]} is {job.get('status')}."
            return "No active job."
        if name == "jobs_cancel":
            return "Job cancelled."
    return "Done."


def make_llm(
    *,
    api_key: str,
    model: str,
    force_mock: bool = False,
) -> tuple[BaseLLM, str]:
    """Return (llm, backend_name)."""
    if force_mock or not api_key.strip():
        return MockLLM(), "mock"
    return AnthropicLLM(api_key=api_key, model=model), "anthropic"
