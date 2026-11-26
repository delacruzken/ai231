"""Color-coded dialogue logs (speaker vs model vs tools)."""

from __future__ import annotations

import logging
import os
import sys


_RESET = "\033[0m"
_CYAN = "\033[96m"  # you / ASR
_MAGENTA = "\033[95m"  # kaki / model
_YELLOW = "\033[93m"  # tools
_RED = "\033[91m"
_DIM = "\033[2m"


def _use_color() -> bool:
    # Default on so nohup/tail logs still show YOU vs KAKI tint; set NO_COLOR=1 to disable.
    if os.environ.get("NO_COLOR", "").strip():
        return False
    if os.environ.get("ME3_FORCE_COLOR", "").strip() in ("0", "false", "no"):
        return False
    return True


class DialogueFormatter(logging.Formatter):
    """Tint whole lines: cyan = you, magenta = model, yellow = tools."""

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if not _use_color():
            return msg
        plain = record.getMessage()
        if (
            plain.startswith("YOU ")
            or plain.startswith("asr_partial")
            or plain.startswith("asr_event")
            or plain.startswith("user_turn")
        ):
            return f"{_CYAN}{msg}{_RESET}"
        if (
            plain.startswith("KAKI ")
            or plain.startswith("assistant_")
            or plain.startswith("llm_first_token")
            or "forcing web_search" in plain
        ):
            return f"{_MAGENTA}{msg}{_RESET}"
        if (
            plain.startswith("TOOL ")
            or plain.startswith("tool_call")
            or plain.startswith("tool_result")
        ):
            return f"{_YELLOW}{msg}{_RESET}"
        if record.levelno >= logging.ERROR:
            return f"{_RED}{msg}{_RESET}"
        if record.levelno >= logging.WARNING:
            return f"{_YELLOW}{msg}{_RESET}"
        if plain.startswith("barge"):
            return f"{_DIM}{msg}{_RESET}"
        return msg
