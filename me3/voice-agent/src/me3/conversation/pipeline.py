"""Conversation orchestrator: turn commit → LLM → TTS with cancel/overlap hooks."""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Any, Callable
from uuid import uuid4

from me3.config import AppConfig, load_config, load_secrets
from me3.conversation.acks import AckConfig, AckManager
from me3.conversation.llm import make_llm
from me3.conversation.memory import ConversationMemory
from me3.conversation.overlap import OverlapClassifier, OverlapConfig
from me3.conversation.persona import build_system_prompt
from me3.conversation.turn_taker import TurnTaker, TurnTakerConfig
from me3.delegation.service import DelegationService
from me3.profile.owner import OwnerProfileStore
from me3.publication.service import PublicationService
from me3.tasks.service import TaskService
from me3.tasks.tools import dispatch_tool

logger = logging.getLogger("me3.conversation.pipeline")


class ConversationOrchestrator:
    def __init__(
        self,
        cfg: AppConfig | None = None,
        *,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        speak: Callable[[str, str], None] | None = None,
        cancel_speech: Callable[[], None] | None = None,
        set_state: Callable[[str], None] | None = None,
        force_mock: bool | None = None,
        task_service: TaskService | None = None,
        delegation: DelegationService | None = None,
        publication: PublicationService | None = None,
        owner_store: OwnerProfileStore | None = None,
    ):
        self.cfg = cfg or load_config()
        secrets = load_secrets()
        self.on_event = on_event or (lambda e: None)
        self.speak = speak or (lambda text, kind: None)
        self.cancel_speech = cancel_speech or (lambda: None)
        self.set_state = set_state or (lambda s: None)
        self.tasks = task_service or TaskService(timezone_name=self.cfg.session.timezone)
        self.delegation = delegation
        self.publication = publication
        self.owner_store = owner_store or OwnerProfileStore()

        force = self.cfg.session.force_mock_llm if force_mock is None else force_mock
        self.llm, self.llm_backend = make_llm(
            api_key=secrets.anthropic_api_key,
            model=self.cfg.models.llm,
            force_mock=force,
        )
        self.refresh_system_prompt()
        self.memory = ConversationMemory()
        self.turn_taker = TurnTaker(
            TurnTakerConfig(
                silence_normal_ms=self.cfg.session.silence_normal_ms,
                silence_incomplete_ms=self.cfg.session.silence_incomplete_ms,
                silence_clarification_ms=self.cfg.session.silence_clarification_ms,
            )
        )
        self.overlap = OverlapClassifier(OverlapConfig(yield_ms=self.cfg.session.overlap_yield_ms))
        self.acks = AckManager(
            AckConfig(
                delay_ms=self.cfg.session.ack_delay_ms,
                backchannel_enabled=self.cfg.session.backchannel_enabled,
                backchannel_cooldown_ms=self.cfg.session.backchannel_cooldown_ms,
            )
        )
        self.active_turn_id: str | None = None
        self.active_user_text: str | None = None
        self.active_llm_speculative: bool = False
        self._lock = threading.RLock()
        self._worker: threading.Thread | None = None
        self._assistant_buf = ""
        self._last_commit_norm = ""
        self._last_commit_mono = 0.0
        # Cover full think+speak for short answers; screenshots showed ~7–8s dupes.
        self._duplicate_commit_window_s = 12.0

    def refresh_system_prompt(self) -> None:
        from me3.config import load_secrets

        secrets = load_secrets()
        owner = (secrets.github_owner or "").strip()
        repo = (secrets.github_repo or "").strip()
        pages = (secrets.github_pages_url or "").strip()
        if not pages and owner and repo:
            pages = f"https://{owner}.github.io/{repo}/"
        publication_target = {
            "configured": bool(owner and repo and (secrets.github_pat or "").strip()),
            "owner": owner,
            "repo": repo,
            "branch": (secrets.github_branch or "main").strip(),
            "site_root": (secrets.github_site_root or "docs").strip(),
            "pages_url": pages,
        }
        self.system_prompt = build_system_prompt(
            timezone=self.cfg.session.timezone,
            locale=self.cfg.session.locale,
            owner_profile=self.owner_store.get(),
            retrieval_provider=getattr(self.tasks, "retrieval_provider", None),
            delegation_provider=getattr(self.delegation, "provider", None) if self.delegation else None,
            publication_provider=getattr(self.publication, "provider", None) if self.publication else None,
            publication_target=publication_target,
        )

    def reset_listening(self) -> None:
        self.turn_taker.reset()
        self.overlap.reset()
        self.acks.reset()

    def cancel_active(self, reason: str = "cancel") -> None:
        with self._lock:
            tid = self.active_turn_id
            self.active_turn_id = None
            self.active_user_text = None
            self.active_llm_speculative = False
        if tid:
            self.llm.cancel(tid)
        self.cancel_speech()
        self.acks.reset()
        self.overlap.reset()
        self.on_event({"type": "turn_cancelled", "reason": reason, "turn_id": tid})

    def on_user_speech(self) -> None:
        self.turn_taker.on_speech()
        self.turn_taker.cancel_pending()

    def on_asr_partial(self, text: str) -> None:
        self.turn_taker.on_partial(text)
        self.overlap.on_partial(text)

    def on_asr_final(self, text: str) -> None:
        """Confirm or correct a speculative silence-commit; never re-arm turn-taker.

        Late Zipformer finals often arrive after the LLM turn already finished.
        Feeding those into ``turn_taker.on_partial`` re-seeded the same utterance
        and caused a second commit ~silence_normal_ms later (duplicate Q&A).
        """
        text = (text or "").strip()
        if not text:
            return
        with self._lock:
            tid = self.active_turn_id
            committed = self.active_user_text
            speculative = self.active_llm_speculative
            last_norm = self._last_commit_norm
            last_mono = self._last_commit_mono
        now = time.monotonic()
        norm = re.sub(r"[^a-z0-9\s]", "", text.lower()).strip()

        # No in-flight speculative turn: ignore finals that match the last commit
        # (stale endpoint after speak) and do not seed turn_taker.
        if not tid or not speculative:
            if (
                norm
                and last_norm
                and (
                    norm == last_norm
                    or _asr_text_compatible(last_norm, norm)
                    or _asr_text_compatible(norm, last_norm)
                )
                and (now - last_mono) < self._duplicate_commit_window_s
            ):
                logger.info("ignore stale asr_final matching last commit: %s", text[:80])
                self.turn_taker.reset()
                return
            # Genuinely new utterance finalized while idle — let silence clock own commit
            # via partials; a lone final should not immediately re-commit.
            self.turn_taker.on_partial(text)
            return

        if _asr_text_compatible(committed or "", text):
            # Keep the early call; clear turn-taker so the final cannot re-commit.
            self.active_llm_speculative = False
            self.turn_taker.reset()
            return
        self.cancel_active(reason="asr_final_correction")
        self.set_state("listening")
        self.commit_user_turn(text, speculative=False)

    def tick(self, *, enabled: bool, state: str) -> None:
        """Called from session pump while ON."""
        if not enabled:
            return
        if state == "listening":
            decision = self.turn_taker.evaluate()
            if decision.action == "clarify":
                self.speak("Could you finish that thought?", "clarify")
                self.on_event({"type": "clarify", "silence_ms": decision.silence_ms})
            elif decision.action == "commit" and self.turn_taker.partial_text:
                # Early commit on silence+partial: speculative until asr_final confirms.
                self.commit_user_turn(self.turn_taker.partial_text, speculative=True)
        elif state == "thinking":
            phrase = self.acks.should_ack()
            if phrase:
                self.speak(phrase, "ack")
                self.on_event({"type": "ack", "text": phrase})
        elif state == "overlap_paused":
            # speech_ms approximate from overlap start
            speech_ms = 0.0
            if self.overlap.started_mono:
                speech_ms = (time.monotonic() - self.overlap.started_mono) * 1000.0
            d = self.overlap.evaluate(speech_ms=speech_ms)
            if d.action == "resume":
                self.on_event({"type": "overlap", "action": "resume", "reason": d.reason})
                self.set_state("speaking")
                self.overlap.reset()
            elif d.action == "yield":
                self.on_event({"type": "overlap", "action": "yield", "reason": d.reason})
                self.cancel_active(reason=f"overlap_{d.reason}")
                self.set_state("listening")
                self.turn_taker.reset()
                if self.turn_taker.partial_text:
                    # keep partial as new turn seed
                    pass

    def begin_overlap(self) -> None:
        self.overlap.begin()
        self.set_state("overlap_paused")
        # Pause: flush current playback (simplified Phase 3 — no resume buffer yet beyond generation bump)
        self.cancel_speech()
        self.on_event({"type": "overlap", "action": "pause"})

    def commit_user_turn(self, text: str, *, speculative: bool = False) -> str:
        text = text.strip()
        if not text:
            return ""
        norm = re.sub(r"[^a-z0-9\s]", "", text.lower()).strip()
        now = time.monotonic()
        with self._lock:
            # Drop near-duplicate commits (silence+ASR final, or post-TTS re-endpoint).
            if norm and self._last_commit_norm and (now - self._last_commit_mono) < self._duplicate_commit_window_s:
                if (
                    norm == self._last_commit_norm
                    or _asr_text_compatible(self._last_commit_norm, norm)
                    or _asr_text_compatible(norm, self._last_commit_norm)
                ):
                    logger.info(
                        "skip duplicate commit within %.0fs: %s",
                        self._duplicate_commit_window_s,
                        text[:80],
                    )
                    if self.active_turn_id and self.active_llm_speculative and not speculative:
                        self.active_llm_speculative = False
                    self.turn_taker.reset()
                    return self.active_turn_id or ""
            if self.active_turn_id and self.active_user_text:
                active_norm = re.sub(r"[^a-z0-9\s]", "", self.active_user_text.lower()).strip()
                if norm and active_norm and (
                    norm == active_norm
                    or _asr_text_compatible(active_norm, norm)
                    or _asr_text_compatible(norm, active_norm)
                ):
                    logger.info("skip commit; already answering same utterance")
                    if speculative is False:
                        self.active_llm_speculative = False
                    self.turn_taker.reset()
                    return self.active_turn_id
            turn_id = str(uuid4())
            self.active_turn_id = turn_id
            self.active_user_text = text
            self.active_llm_speculative = speculative
            self._last_commit_norm = norm
            self._last_commit_mono = now
        self.turn_taker.reset()
        self.memory.add_turn(
            "user",
            text,
            meta={"source": "asr_partial" if speculative else "asr_final"},
        )
        self.on_event({"type": "transcript", "role": "user", "text": text, "turn_id": turn_id})
        self.set_state("thinking")
        self.acks.on_operation_started()
        self.on_event(
            {
                "type": "turn_started",
                "turn_id": turn_id,
                "user": text,
                "speculative": speculative,
            }
        )

        def tool_executor(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            out = dispatch_tool(
                self.tasks,
                name,
                arguments,
                turn_id=turn_id,
                delegation=self.delegation,
                publication=self.publication,
            )
            self.on_event({"type": "tool_result", "name": name, "result": out, "turn_id": turn_id})
            return out

        def worker():
            from me3.benchmark.marks import mark

            try:
                messages = self.memory.recent_messages(self.cfg.session.max_context_turns)
                # ensure last user is present
                if not messages or messages[-1].get("content") != text:
                    messages = list(messages) + [{"role": "user", "content": text}]
                full = ""
                sentence_buf = ""
                first = True
                first_token = True
                mark_kwargs: dict[str, Any] = {
                    "messages": len(messages),
                    "chars": sum(len(m.get("content") or "") for m in messages),
                }
                if speculative:
                    # fdbench ignores speculative llm_request when pairing asr_final for WER.
                    mark_kwargs["speculative"] = True
                mark("llm_request", **mark_kwargs)
                for chunk in self.llm.stream(
                    messages,
                    self.system_prompt,
                    turn_id=turn_id,
                    tools_enabled=True,
                    tool_executor=tool_executor,
                ):
                    with self._lock:
                        if self.active_turn_id != turn_id:
                            mark("llm_cancel")
                            return
                    if chunk.done:
                        break
                    if not chunk.text:
                        continue
                    if first_token:
                        mark("llm_first_token")
                        first_token = False
                    full += chunk.text
                    sentence_buf += chunk.text
                    self.on_event(
                        {
                            "type": "transcript",
                            "role": "assistant",
                            "text": full,
                            "partial": True,
                            "turn_id": turn_id,
                        }
                    )
                    # Speak completed sentences as they arrive
                    while True:
                        m = re.search(r"[.!?]\s+", sentence_buf)
                        if not m:
                            break
                        sent = sentence_buf[: m.end()].strip()
                        sentence_buf = sentence_buf[m.end() :]
                        if not sent:
                            continue
                        if first:
                            self.acks.on_result_ready()
                            self.set_state("speaking")
                            first = False
                        self.speak(sent, "assistant")
                rem = sentence_buf.strip()
                if rem:
                    with self._lock:
                        if self.active_turn_id != turn_id:
                            return
                    if first:
                        self.acks.on_result_ready()
                        self.set_state("speaking")
                        first = False
                    self.speak(rem, "assistant")
                if full.strip():
                    self.memory.add_turn(
                        "assistant",
                        full.strip(),
                        meta={"turn_id": turn_id, "llm": self.llm_backend},
                    )
                    self.on_event(
                        {
                            "type": "transcript",
                            "role": "assistant",
                            "text": full.strip(),
                            "partial": False,
                            "turn_id": turn_id,
                        }
                    )
                self.on_event({"type": "turn_done", "turn_id": turn_id})
                mark("bot_stopped")
            except Exception as exc:
                logger.exception("turn failed")
                self.on_event({"type": "error", "message": str(exc), "turn_id": turn_id})
            finally:
                with self._lock:
                    if self.active_turn_id == turn_id:
                        self.active_turn_id = None
                        self.active_user_text = None
                        self.active_llm_speculative = False
                self.set_state("listening")
                self.turn_taker.reset()

        self._worker = threading.Thread(target=worker, name=f"llm-{turn_id[:8]}", daemon=True)
        self._worker.start()
        return turn_id


def _asr_text_compatible(committed: str, final: str) -> bool:
    """True when final transcript is the same utterance as the early commit (allowing completion)."""
    a = re.sub(r"[^a-z0-9\s]", "", (committed or "").lower()).split()
    b = re.sub(r"[^a-z0-9\s]", "", (final or "").lower()).split()
    if not a or not b:
        return not a and not b
    if a == b:
        return True
    # Partial was a prefix of the final (common when silence fires early).
    if len(a) <= len(b) and b[: len(a)] == a:
        return True
    return False
