"""ANSI console helpers for live assistant logging."""
from __future__ import annotations

import os
import sys


def color_enabled() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("ME2_COLOR", "").lower() in {"0", "false", "no"}:
        return False
    if os.environ.get("ME2_COLOR", "").lower() in {"1", "true", "yes"}:
        return True
    return sys.stdout.isatty()


class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_CYAN = "\033[96m"


def paint(text: str, *codes: str) -> str:
    if not color_enabled() or not codes:
        return text
    return "".join(codes) + text + C.RESET


def tag(name: str, kind: str = "dim") -> str:
    styles = {
        "vad": (C.DIM,),
        "wake_hit": (C.BOLD, C.BRIGHT_GREEN),
        "wake_miss": (C.YELLOW,),
        "wake_skip": (C.DIM,),
        "listen": (C.BOLD, C.BRIGHT_CYAN),
        "cmd_ok": (C.BOLD, C.BRIGHT_GREEN),
        "cmd_rej": (C.BOLD, C.BRIGHT_RED),
        "sleep": (C.DIM,),
        "ready": (C.CYAN,),
        "action": (C.BOLD, C.MAGENTA),
    }
    return paint(f"[{name}]", *styles.get(kind, (C.DIM,)))


def score_vs_thr(score: float, thr: float, digits: int = 3) -> str:
    """Color a score relative to a threshold (green ≥ thr, yellow close, red below)."""
    fmt = f"{{:.{digits}f}}"
    s = fmt.format(score)
    t = fmt.format(thr)
    delta = score - thr
    if score >= thr:
        body = paint(f"{s} ≥ {t}", C.BOLD, C.BRIGHT_GREEN)
        mark = paint("PASS", C.BOLD, C.BRIGHT_GREEN)
    elif delta >= -0.05:
        body = paint(f"{s} < {t}", C.BOLD, C.BRIGHT_YELLOW)
        mark = paint("CLOSE", C.BOLD, C.BRIGHT_YELLOW)
    else:
        body = paint(f"{s} < {t}", C.BOLD, C.BRIGHT_RED)
        mark = paint("FAIL", C.BOLD, C.BRIGHT_RED)
    return f"{body} {mark}"


def conf_vs_thr(conf: float, thr: float) -> str:
    return score_vs_thr(conf, thr, digits=3)
