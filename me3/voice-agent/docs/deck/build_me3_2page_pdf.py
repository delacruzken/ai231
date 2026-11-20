#!/usr/bin/env python3
"""Build ME3 filled 2-page deck PDF (960x540 template geometry)."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.colors import Color, HexColor, white, black
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

W, H = 960, 540
OUT = Path(__file__).resolve().parent / "ME3_2page_FILLED.pdf"

NAVY = HexColor("#0F2744")
TEAL = HexColor("#1F7F86")
ORANGE = HexColor("#E8742A")
GRAY = HexColor("#5A6573")
LIGHT = HexColor("#F4F6F8")
LINE = HexColor("#D0D5DD")
GREEN = HexColor("#1B7F4E")
RED = HexColor("#B42318")
BOX = HexColor("#E8EEF5")


def _wrap(c: canvas.Canvas, text: str, font: str, size: float, max_w: float) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if c.stringWidth(trial, font, size) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [""]


def header(c: canvas.Canvas, page: int) -> None:
    c.setFillColor(NAVY)
    c.rect(0, H - 36, W, 36, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(18, H - 23, "MACHINE EXERCISE — 2-PAGE  ·  Full-Duplex Voice Agent on a Raspberry Pi")
    c.setFont("Helvetica", 8)
    c.drawRightString(W - 18, H - 22, f"ME3 Kaki  ·  2015-02211  ·  {page}/2")
    c.setFillColor(TEAL)
    c.rect(0, H - 40, W, 4, fill=1, stroke=0)


def footer(c: canvas.Canvas) -> None:
    c.setFillColor(GRAY)
    c.setFont("Helvetica", 7)
    c.drawString(
        18,
        10,
        "Draft for comparison  ·  Latency = laptop fdbench direct 2026-11-20_09-04-05  ·  Pi Results pending  ·  docs/deck/ME3_2page_FILLED.md",
    )


def section_title(c: canvas.Canvas, x: float, y: float, num: str, title: str) -> None:
    c.setFillColor(TEAL)
    c.roundRect(x, y - 2, 22, 16, 3, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 9)
    c.drawCentredString(x + 11, y + 2, num)
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(x + 28, y + 1, title)


def page1(c: canvas.Canvas) -> None:
    header(c, 1)
    footer(c)

    # --- §1 System diagram ---
    section_title(c, 16, H - 58, "1", "SYSTEM DIAGRAM")
    c.setFont("Helvetica", 7)
    c.setFillColor(GRAY)
    c.drawString(200, H - 56, "audio path · harness · MCP/tools · cloud · barge-in")

    # On-device box
    c.setStrokeColor(NAVY)
    c.setFillColor(BOX)
    c.setLineWidth(1.2)
    c.roundRect(16, 168, 600, 290, 6, fill=1, stroke=1)
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 8)
    c.drawString(26, 442, "ON DEVICE (Pi 5 target · laptop today)")

    # Mic
    c.setFillColor(TEAL)
    c.roundRect(30, 400, 70, 28, 4, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 7)
    c.drawCentredString(65, 411, "USB mic")

    # arrows / pipeline blocks
    def block(x, y, w, h, title, sub, fill=TEAL):
        c.setFillColor(fill)
        c.roundRect(x, y, w, h, 4, fill=1, stroke=0)
        c.setFillColor(white)
        c.setFont("Helvetica-Bold", 7)
        c.drawCentredString(x + w / 2, y + h / 2 + 4, title)
        c.setFont("Helvetica", 6)
        c.drawCentredString(x + w / 2, y + h / 2 - 7, sub)

    block(120, 400, 90, 28, "Silero VAD", "in-process")
    block(230, 400, 110, 28, "Local ASR", "Kroko Zipformer")

    # Harness box
    c.setStrokeColor(TEAL)
    c.setFillColor(HexColor("#DCEEF0"))
    c.setLineWidth(1)
    c.roundRect(30, 250, 560, 130, 5, fill=1, stroke=1)
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 8)
    c.drawString(40, 362, "ME3 CUSTOM HARNESS  ·  SessionController + ConversationOrchestrator + UI :8780")
    c.setFont("Helvetica", 6.5)
    c.setFillColor(GRAY)
    c.drawString(40, 348, "turn-taking · echo gate · two-phase barge-in · tool dispatch")
    c.setFillColor(NAVY)
    c.setFont("Helvetica", 7)
    for i, line in enumerate(
        [
            "Tasks MCP-shaped → timers, notes, web_* (Tavily)",
            "Delegation → jobs_* (OpenCode / mock)",
            "Publication → publish_* (GitHub Pages)",
        ]
    ):
        c.drawString(48, 328 - i * 14, "•  " + line)

    block(40, 200, 120, 32, "Local TTS", "Piper lessac-medium", fill=HexColor("#2A5F8F"))
    block(180, 200, 80, 32, "Speaker", "USB / M1A", fill=HexColor("#2A5F8F"))

    # Barge-in callout
    c.setStrokeColor(ORANGE)
    c.setFillColor(HexColor("#FFF4EC"))
    c.setLineWidth(1)
    c.roundRect(290, 195, 285, 42, 4, fill=1, stroke=1)
    c.setFillColor(ORANGE)
    c.setFont("Helvetica-Bold", 7)
    c.drawString(300, 220, "BARGE-IN")
    c.setFillColor(NAVY)
    c.setFont("Helvetica", 6.5)
    c.drawString(300, 206, "speech → soft-pause TTS → confirm → cancel LLM + listen")

    # Cloud box
    c.setStrokeColor(ORANGE)
    c.setFillColor(HexColor("#FFF8F3"))
    c.roundRect(630, 300, 310, 158, 6, fill=1, stroke=1)
    c.setFillColor(ORANGE)
    c.setFont("Helvetica-Bold", 8)
    c.drawString(642, 440, "CLOUD")
    c.setFillColor(NAVY)
    c.setFont("Helvetica", 7)
    for i, line in enumerate(
        [
            "Anthropic Claude Haiku 4.5 — LLM + tools",
            "Tavily — web search / extract",
            "GitHub — Pages publish (PAT on device)",
            "",
            "Mic PCM / ASR / TTS stay on device",
            "(not cloud speech APIs)",
        ]
    ):
        c.drawString(642, 420 - i * 14, line)

    # Flow strip
    c.setFillColor(LIGHT)
    c.roundRect(630, 195, 310, 90, 5, fill=1, stroke=0)
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 7)
    c.drawString(642, 265, "HARNESS (honest)")
    c.setFont("Helvetica", 6.5)
    for i, line in enumerate(
        [
            "Live duplex = custom ME3 runtime",
            "Pipecat 1.15 = fdbench transport only",
            "ASR/TTS: in-process engines; same code",
            "also packaged as me3.mcp.asr / tts WS",
        ]
    ):
        c.drawString(642, 248 - i * 12, line)

    # Bottom flow
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 8)
    c.drawString(16, 145, "voice in → VAD → ASR → LLM (+tools) → TTS → voice out")
    c.setFillColor(ORANGE)
    c.setFont("Helvetica", 7.5)
    c.drawString(16, 130, "barge-in: speech detected → stop TTS + LLM → listen")

    # --- §2 Prompt ---
    section_title(c, 16, 108, "2", "PROMPT  ·  SYSTEM PROMPT (v1)")
    c.setFillColor(LIGHT)
    c.roundRect(16, 42, 450, 58, 4, fill=1, stroke=0)
    c.setFillColor(NAVY)
    c.setFont("Helvetica", 6.5)
    prompt_lines = [
        "You are Kaki, a capable voice assistant for the session owner.",
        "Keep every reply very short. Never use markdown — TTS reads aloud.",
        "If the user speaks, stop and listen. Confirm before publish / irreversible actions.",
        "Use tools for world facts, timers/notes, coding jobs, publish. No invented secrets.",
    ]
    for i, line in enumerate(prompt_lines):
        c.drawString(24, 86 - i * 11, line)

    # --- §3 Context ---
    section_title(c, 480, 108, "3", "CONTEXT ENGINEERING")
    c.setFont("Helvetica-Bold", 6.5)
    c.setFillColor(TEAL)
    c.drawString(490, 90, "GOES IN")
    c.drawString(720, 90, "MANAGED")
    c.setFillColor(NAVY)
    c.setFont("Helvetica", 6)
    left = [
        "system prompt v1 + persona",
        "ASR partials / silence-commit finals",
        "VAD + turn + barge-in state",
        "tool results (tasks/jobs/publish)",
        "owner public profile (approved)",
    ]
    right = [
        "max_context_turns=12; dedupe window",
        "idle→listen→think→speak",
        "flush TTS on barge-in",
        "world facts must web_search",
        "log messages/chars on llm_request",
    ]
    for i, (a, b) in enumerate(zip(left, right)):
        c.drawString(490, 76 - i * 10, "• " + a)
        c.drawString(720, 76 - i * 10, "• " + b)


def _table(
    c: canvas.Canvas,
    x: float,
    y: float,
    col_w: list[float],
    headers: list[str],
    rows: list[list[str]],
    row_h: float = 18,
    header_h: float = 16,
    font_size: float = 5.5,
) -> None:
    c.setFillColor(NAVY)
    c.rect(x, y - header_h, sum(col_w), header_h, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", font_size)
    cx = x
    for w, h in zip(col_w, headers):
        c.drawString(cx + 3, y - header_h + 5, h)
        cx += w
    yy = y - header_h
    for ri, row in enumerate(rows):
        yy -= row_h
        c.setFillColor(LIGHT if ri % 2 == 0 else white)
        c.rect(x, yy, sum(col_w), row_h, fill=1, stroke=0)
        c.setStrokeColor(LINE)
        c.setLineWidth(0.3)
        c.rect(x, yy, sum(col_w), row_h, fill=0, stroke=1)
        c.setFillColor(NAVY)
        c.setFont("Helvetica", font_size)
        cx = x
        for w, cell in zip(col_w, row):
            # wrap lightly by truncating with ellipsis if needed
            text = cell
            while c.stringWidth(text, "Helvetica", font_size) > w - 6 and len(text) > 3:
                text = text[:-2] + "…"
            c.drawString(cx + 3, yy + 6, text)
            cx += w


def page2(c: canvas.Canvas) -> None:
    header(c, 2)
    footer(c)

    section_title(c, 16, H - 58, "4", "MCP SERVICES + MODEL SERVING")
    _table(
        c,
        16,
        H - 68,
        [70, 175, 175, 130, 340],
        ["Component", "MCP / transport", "Tools", "Model", "Served where / notes"],
        [
            ["ASR", "me3.mcp.asr WS + in-process live", "PCM → partial/final", "Kroko Zipformer", "On device; final at silence-commit"],
            ["TTS", "me3.mcp.tts WS + in-process live", "speak / cancel", "Piper lessac-medium", "On device; cancel on barge-in"],
            ["LLM", "harness HTTP client", "tools / function-calling", "claude-haiku-4-5", "Cloud Anthropic; streaming"],
            ["VAD", "in-process", "detect_speech(frames)", "Silero ONNX", "On device; 16 kHz mono"],
            ["Tasks", "me3.mcp.tasks", "timer.* notes.* web.*", "→ Tavily", "On device; search API cloud"],
            ["Delegation", "me3.mcp.delegation", "jobs.submit/status/…", "OpenCode / mock", "On device worker"],
            ["Publication", "publish service/tools", "publish.validate/site", "GitHub Pages", "On device → GitHub PAT"],
        ],
        row_h=16,
        font_size=5.5,
    )

    section_title(c, 16, 268, "5", "EVALUATION & BENCHMARK")
    c.setFillColor(GRAY)
    c.setFont("Helvetica", 6)
    c.drawString(
        200,
        270,
        "One clock · VAD end→ASR final→LLM TTFT→TTS TTFB→speaker · 10 phrases × 3 runs · laptop direct (Pi pending)",
    )

    # Metrics table
    _table(
        c,
        16,
        258,
        [95, 200, 70, 70, 70, 40],
        ["Metric", "Definition", "Target", "Mean", "p95", "n"],
        [
            ["E2E latency", "end of speech → audio out", "< 4.00 s", "2.80 s", "3.81 s", "20"],
            ["ASR final", "end of speech → transcript", "< 0.80 s", "1.96 s", "2.95 s", "20"],
            ["LLM TTFT", "transcript → first token", "< 1.50 s", "1.40 s", "2.39 s", "6"],
            ["TTS TTFB", "text → first audio chunk", "< 0.60 s", "0.09 s", "0.24 s", "19"],
            ["Barge-in", "speech → playback stopped", "< 0.80 s", "—", "—", "0"],
            ["WER", "ASR accuracy on test set", "< 15 %", "5.7 %", "—", "20"],
        ],
        row_h=15,
        font_size=6,
    )

    # Status chips
    c.setFont("Helvetica", 6.5)
    c.setFillColor(GREEN)
    c.drawString(560, 250, "Met: E2E · TTS · WER")
    c.setFillColor(ORANGE)
    c.drawString(560, 236, "Over: ASR final · LLM p95")
    c.setFillColor(RED)
    c.drawString(560, 222, "Empty: Barge-in (no cut-in trials)")

    # Demo / Results / Challenges
    y0 = 118
    boxes = [
        (
            16,
            "DEMO",
            [
                "1. Session ON → short greeting",
                "2. Current-fact Q → web_search",
                "3. Timer or note (task tool)",
                "4. Barge-in mid-answer",
                "5. Optional job/publish (mock OK)",
            ],
        ),
        (
            250,
            "RESULTS",
            [
                "Run: 2026-11-20_09-04-05",
                "Rig: laptop Darwin · 16 GB · direct",
                "Models: Kroko · Haiku · Piper · Silero",
                "Paste latency_chart.svg from run",
                "⚠ Replace with Pi card before final",
            ],
        ),
        (
            500,
            "CHALLENGES",
            [
                "WER n=0 → silence-commit asr_final (5.7%)",
                "Early cutoffs → incomplete silence patience",
                "Noise barge → energy floor + soft-pause",
                "Laptop echo → ASR gate while speaking",
                "Harness honesty → custom + Pipecat bench",
            ],
        ),
    ]
    for x, title, lines in boxes:
        c.setFillColor(NAVY)
        c.roundRect(x, y0, 220, 88, 4, fill=1, stroke=0)
        c.setFillColor(white)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(x + 8, y0 + 72, title)
        c.setFont("Helvetica", 6)
        for i, line in enumerate(lines):
            c.drawString(x + 8, y0 + 58 - i * 11, line)

    c.setFillColor(GRAY)
    c.setFont("Helvetica", 6)
    c.drawString(
        16,
        48,
        "Refs: MCP · sherpa-onnx / Piper · class fd-benchmark · repo me3/voice-agent · SYSTEM_TOPOLOGY.md · benchmarks/2026-11-20_09-04-05_RESULTS.md",
    )
    c.setFillColor(TEAL)
    c.setFont("Helvetica-Bold", 7)
    c.drawString(
        16,
        34,
        "vs others: stronger local speech + WER; weaker Pipecat-as-live-harness cartoon + Pi Results + barge-in row",
    )


def main() -> None:
    c = canvas.Canvas(str(OUT), pagesize=(W, H))
    page1(c)
    c.showPage()
    page2(c)
    c.save()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
