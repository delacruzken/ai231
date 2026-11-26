"""Forced web-evidence routing for world-fact questions."""

from __future__ import annotations

from me3.conversation.live_facts import needs_live_web_evidence
from me3.conversation.persona import build_system_prompt


def test_world_fact_questions_require_search():
    for text in (
        "Who is the President of the United States?",
        "What is the weather in Manila today?",
        "What is the latest news about OpenAI?",
        "Who won the World Cup?",
        "What is Apple's stock price?",
        "Is it raining in Tokyo right now?",
    ):
        assert needs_live_web_evidence(text), text


def test_session_local_skips_forced_search():
    for text in (
        "Hello",
        "What time is it?",
        "Set a timer for five minutes",
        "Save a note about groceries",
        "List my jobs",
        "Publish the site to GitHub Pages",
        "Thanks",
    ):
        assert not needs_live_web_evidence(text), text


def test_persona_requires_world_fact_search():
    prompt = build_system_prompt()
    assert "MUST use web_search" in prompt
    assert "training memory" in prompt


def test_persona_handles_asr_misspellings_and_dated_facts():
    prompt = build_system_prompt()
    assert "speech recognition" in prompt
    assert "search the most likely corrected name" in prompt
    assert "Do not use the words 'today'" in prompt


def test_web_search_tool_does_not_default_to_finance_topic():
    from me3.tasks.tools import TOOL_DEFINITIONS

    web = next(t for t in TOOL_DEFINITIONS if t["name"] == "web_search")
    topic = web["input_schema"]["properties"]["topic"]
    assert "Leave topic unset" in web["description"]
    assert "Omit for most questions" in topic["description"]
    assert "finance" in topic["enum"]  # still allowed, just not the default
