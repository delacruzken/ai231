"""Background timer poller — wall-clock deadlines, restart reconciliation."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

from me3.tasks.service import TaskService

logger = logging.getLogger("me3.tasks.scheduler")

NotifyCallback = Callable[[dict[str, Any]], None]


class TimerScheduler:
    def __init__(
        self,
        service: TaskService,
        *,
        interval_s: float = 0.5,
        on_notification: NotifyCallback | None = None,
    ):
        self.service = service
        self.interval_s = interval_s
        self.on_notification = on_notification
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        # Immediate reconcile for missed deadlines after restart
        self._tick()
        self._thread = threading.Thread(target=self._loop, name="timer-scheduler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                self._tick()
            except Exception:
                logger.exception("scheduler tick failed")

    def _tick(self) -> None:
        fired = self.service.fire_due_timers()
        for item in fired:
            notif = item.get("notification")
            if notif and self.on_notification:
                try:
                    self.on_notification(notif)
                except Exception:
                    logger.exception("on_notification failed")
