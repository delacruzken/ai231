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
        self._lock = threading.RLock()
        self._worker: threading.Thread | None = None
        self._assistant_buf = ""

    def refresh_system_prompt(self) -> None:
        self.system_prompt = build_system_prompt(
            timezone=self.cfg.session.timezone,
            locale=self.cfg.session.locale,
            owner_profile=self.owner_store.get(),
            retrieval_provider=getattr(self.tasks, "retrieval_provider", None),
            delegation_provider=getattr(self.delegation, "provider", None) if self.delegation else None,
            publication_provider=getattr(self.publication, "provider", None) if self.publication else None,
        )

    def reset_listening(self) -> None:
        self.turn_taker.reset()
        self.overlap.reset()
        self.acks.reset()

    def cancel_active(self, reason: str = "cancel") -> None:
        with self._lock:
            tid = self.active_turn_id
            self.active_turn_id = None
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
        self.turn_taker.on_partial(text)

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
                self.commit_user_turn(self.turn_taker.partial_text)
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

    def commit_user_turn(self, text: str) -> str:
        text = text.strip()
        if not text:
            return ""
        turn_id = str(uuid4())
        with self._lock:
            self.active_turn_id = turn_id
        self.turn_taker.reset()
        self.memory.add_turn("user", text, meta={"source": "asr"})
        self.on_event({"type": "transcript", "role": "user", "text": text, "turn_id": turn_id})
        self.set_state("thinking")
        self.acks.on_operation_started()
        self.on_event({"type": "turn_started", "turn_id": turn_id, "user": text})

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
                mark("llm_request", messages=len(messages), chars=sum(len(m.get("content") or "") for m in messages))
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
                self.set_state("listening")
                self.turn_taker.reset()

        self._worker = threading.Thread(target=worker, name=f"llm-{turn_id[:8]}", daemon=True)
        self._worker.start()
        return turn_id
