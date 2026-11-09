"""Event-driven acknowledgments and optional listener backchannels."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


ACK_PHRASES = ("Okay.", "One moment.", "Got it.")
BACKCHANNEL_PHRASES = ("Mm-hmm.", "Yeah.", "Right.")


@dataclass
class AckState:
    pending_since: float | None = None
    ack_sent: bool = False
    last_backchannel_mono: float = 0.0
    suppressed: bool = False


@dataclass
class AckConfig:
    delay_ms: int = 700
    backchannel_enabled: bool = False
    backchannel_cooldown_ms: int = 10000


class AckManager:
    def __init__(self, cfg: AckConfig | None = None):
        self.cfg = cfg or AckConfig()
        self.state = AckState()
        self._ack_index = 0

    def reset(self) -> None:
        self.state = AckState()

    def on_operation_started(self) -> None:
        self.state.pending_since = time.monotonic()
        self.state.ack_sent = False
        self.state.suppressed = False

    def on_result_ready(self) -> None:
        """Suppress ack if substantive result arrived first."""
        self.state.suppressed = True
        self.state.pending_since = None

    def should_ack(self, now: float | None = None) -> str | None:
        now = now if now is not None else time.monotonic()
        if self.state.suppressed or self.state.ack_sent or self.state.pending_since is None:
            return None
        if (now - self.state.pending_since) * 1000.0 < self.cfg.delay_ms:
            return None
        self.state.ack_sent = True
        phrase = ACK_PHRASES[self._ack_index % len(ACK_PHRASES)]
        self._ack_index += 1
        return phrase

    def should_backchannel(self, dialogue_ok: bool, now: float | None = None) -> str | None:
        if not self.cfg.backchannel_enabled or not dialogue_ok:
            return None
        now = now if now is not None else time.monotonic()
        if (now - self.state.last_backchannel_mono) * 1000.0 < self.cfg.backchannel_cooldown_ms:
            return None
        self.state.last_backchannel_mono = now
        return BACKCHANNEL_PHRASES[int(now) % len(BACKCHANNEL_PHRASES)]
