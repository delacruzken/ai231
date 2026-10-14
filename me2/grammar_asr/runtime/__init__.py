"""Runtime helpers for on-device assistant inference."""

from __future__ import annotations

__all__ = [
    "VoiceAssistant",
    "EnergyVAD",
    "MockDispatcher",
    "HttpDemoDispatcher",
    "BenchmarkLogger",
]


def __getattr__(name: str):
    if name == "VoiceAssistant":
        from .assistant import VoiceAssistant
        return VoiceAssistant
    if name == "EnergyVAD":
        from .vad import EnergyVAD
        return EnergyVAD
    if name == "MockDispatcher":
        from .actions import MockDispatcher
        return MockDispatcher
    if name == "HttpDemoDispatcher":
        from .actions import HttpDemoDispatcher
        return HttpDemoDispatcher
    if name == "BenchmarkLogger":
        from .logging_jsonl import BenchmarkLogger
        return BenchmarkLogger
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
