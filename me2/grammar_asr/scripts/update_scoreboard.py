#!/usr/bin/env python3
"""Build / append the ME2 HTML experiment scoreboard (slide-style tables).

Every finished training run should call ``append_scoreboard_version`` so a new
versioned card is added. Rebuild anytime:

  python -m grammar_asr.scripts.update_scoreboard
  python -m grammar_asr.scripts.update_scoreboard --seed   # import existing runs
"""
from __future__ import annotations

import argparse
import html
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PACKAGE_ROOT / "runs"
HISTORY = RUNS_DIR / "scoreboard_history.jsonl"
HTML_OUT = RUNS_DIR / "SCOREBOARD.html"


# Canonical version tags for runs that predate explicit version fields.
KNOWN_VERSIONS = {
    "ctc_hf_s0": "v0",
    "wake_boss": "v0",
    "wake_kaki": "v1",
    "wake_kaki_v2": "v2",
    "wake_kaki_v3": "v3",
    "wake_kaki_v4": "v4",
    "wake_kaki_v5": "v5",
    "intent_hf_v1": "v1",
    "intent_hf_v2_conformer": "v2",
    "intent_hf_v2_matchbox": "v2",
    "intent_hf_v2_crnn": "v2",
    "intent_hf_v2b_crnn": "v2b",
    "intent_hf_v2b_matchbox": "v2b",
    "intent_hf_v3_crnn": "v3",
}


def normalize_version(tag: str, version: Any = None) -> str:
    """Return a clean version label (v0 / v1 / v2 / v2b / v3 …), never 'v?'."""
    if tag in KNOWN_VERSIONS:
        return KNOWN_VERSIONS[tag]
    v = str(version or "").strip()
    if v and v not in {"v?", "?", "None", "null"}:
        return v if v.startswith("v") else f"v{v}"
    # Parse trailing _vN / _vNb from the tag itself.
    if "_v" in tag:
        tail = tag.rsplit("_v", 1)[-1]
        if tail and tail[0].isdigit():
            return f"v{tail}"
    return "v0"


def _pct(x: Any, digits: int = 1) -> str:
    if x is None:
        return "—"
    try:
        return f"{100.0 * float(x):.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def _mb(params: Any, bytes_per: float = 4.0) -> str:
    if params is None:
        return "—"
    return f"{int(params) * bytes_per / 1e6:.2f} MB fp32"


def _num(x: Any) -> str:
    if x is None:
        return "—"
    try:
        return f"{int(x):,}"
    except (TypeError, ValueError):
        return str(x)


def load_history() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not HISTORY.exists():
        return rows
    for line in HISTORY.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def append_history(card: Dict[str, Any]) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    card = dict(card)
    card.setdefault("recorded_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
    with HISTORY.open("a", encoding="utf-8") as f:
        f.write(json.dumps(card, default=str) + "\n")


def card_from_command_summary(summary_path: Path, **extra) -> Dict[str, Any]:
    s = json.loads(summary_path.read_text(encoding="utf-8"))
    run_dir = summary_path.parent
    notes_path = run_dir / "NOTES.md"
    notes = ""
    if notes_path.exists():
        notes = notes_path.read_text(encoding="utf-8")[:500]
    cfg_path = run_dir / "config.resolved.json"
    cfg = {}
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    model_name = s.get("model") or (cfg.get("config") or {}).get("model", {}).get("name")
    tag = s.get("tag") or run_dir.name
    version = normalize_version(tag, s.get("version") or extra.get("version"))
    experiment = s.get("experiment") or extra.get("experiment") or model_name
    test = s.get("test") or {}
    val = s.get("val") or {}
    real = test.get("real") or {}
    synth = test.get("synthetic") or {}
    gate = s.get("reject_gate") or {}
    return {
        "kind": "command",
        "tag": tag,
        "version": version,
        "experiment": experiment,
        "title": extra.get("title") or f"ME2 command · {experiment} · {version}",
        "subtitle": extra.get("subtitle")
        or "Always-on wake + on-device command recognition · no cloud",
        "notes": (s.get("notes") or "").strip() or notes,
        "model": {
            "name": model_name,
            "params": s.get("params"),
            "architecture": extra.get("architecture")
            or _arch_blurb(model_name),
            "onnx_mb": _mb(s.get("params")),
            "best_epoch": s.get("best_epoch"),
            "decode": extra.get("decode")
            or (
                "intent head"
                if model_name in {
                    "intent", "intent_crnn", "crnn",
                    "conformer", "matchbox", "intent_conformer", "intent_matchbox",
                }
                else "CTC + grammar"
            ),
        },
        "train": {
            "cluster": extra.get("cluster") or "UP DGX ai-n002 · A100-40GB",
            "seed": (cfg.get("config") or {}).get("seed") or cfg.get("seed"),
            "epochs": (cfg.get("config") or {}).get("epochs"),
            "batch_size": (cfg.get("config") or {}).get("batch_size"),
            "lr": (cfg.get("config") or {}).get("lr"),
            "amp": (cfg.get("config") or {}).get("amp"),
            "finished_at": s.get("finished_at"),
            "best_val_command_acc": s.get("best_val_command_acc"),
        },
        "validation": {
            "val_cmd": val.get("command_accuracy"),
            "val_intent": val.get("intent_accuracy"),
            "val_far": val.get("false_accept_rate"),
            "val_frr": val.get("false_reject_rate"),
            "val_real": (val.get("real") or {}).get("command_accuracy"),
            "test_cmd": test.get("command_accuracy"),
            "test_intent": test.get("intent_accuracy"),
            "test_real": real.get("command_accuracy"),
            "test_synth": synth.get("command_accuracy"),
            "test_far": test.get("false_accept_rate"),
            "test_frr": test.get("false_reject_rate"),
            "test_n": test.get("n"),
            "test_real_n": real.get("n"),
            "reject_min_confidence": gate.get("min_confidence"),
        },
        "dataset": {
            "source": "Course gold HF · ai231-me2-voice-commands",
            "train_hours": "6.10 h / 10,733",
            "test_hours": "2.45 h / 4,443",
            "labels": "19 intents + OOS · 93 variations · 6 slots × 3",
            "fingerprint": (s.get("dataset_fingerprint") or "")[:16] + "…",
        },
        **{k: v for k, v in extra.items() if k not in {
            "title", "subtitle", "architecture", "decode", "cluster", "version",
            "experiment",
        }},
    }


def card_from_wake_metrics(metrics_path: Path, **extra) -> Dict[str, Any]:
    m = json.loads(metrics_path.read_text(encoding="utf-8"))
    run_dir = metrics_path.parent
    v = m.get("validation") or {}
    tag = run_dir.name
    version = normalize_version(tag, extra.get("version"))
    wake_name = "boss" if "boss" in tag else "kaki"
    return {
        "kind": "wake",
        "tag": tag,
        "version": version,
        "experiment": extra.get("experiment") or f"wake_{wake_name}",
        "title": extra.get("title") or f"ME2 wake · {wake_name} · {version}",
        "subtitle": "Binary MicroWakeNet · always-on keyword gate",
        "notes": extra.get("notes") or "",
        "model": {
            "name": "wake",
            "params": m.get("params"),
            "architecture": "MicroWakeNet · DS-conv · binary wake/not-wake",
            "onnx_mb": _mb(m.get("params")),
            "threshold": v.get("threshold"),
        },
        "train": {
            "cluster": extra.get("cluster") or "UP DGX / local",
            "train_positive": m.get("train_positive"),
            "train_negative": m.get("train_negative"),
            "val_positive": m.get("val_positive"),
            "val_negative": m.get("val_negative"),
        },
        "validation": {
            "balanced_acc": v.get("balanced_accuracy"),
            "accuracy": v.get("accuracy"),
            "precision": v.get("precision"),
            "recall": v.get("recall"),
            "far": v.get("false_accept_rate"),
            "frr": v.get("false_reject_rate"),
            "threshold": v.get("threshold"),
        },
        "dataset": {
            "source": "me2/data/wake/kaki (+ gold train as general negatives)",
            "positives": m.get("train_positive", 0) + m.get("val_positive", 0),
            "negatives_split": f"train {m.get('train_negative')} / val {m.get('val_negative')}",
        },
    }


def _arch_blurb(name: Optional[str]) -> str:
    if name in {"intent", "intent_crnn", "crnn"}:
        return (
            "Mic · 16 kHz → log-mel 40 × T (≤5 s) → "
            "IntentCRNN · 4 DS-conv · BiGRU 96×2 · attn pool → "
            "intent head (20) + slot heads (6×3)"
        )
    if name in {"conformer", "intent_conformer", "tiny_conformer"}:
        return (
            "Mic · 16 kHz → log-mel 40 × T (≤5 s) → "
            "Tiny Conformer M · d=64 · 4 blocks · attn pool → "
            "intent head (20) + slot heads (6×3)"
        )
    if name in {"matchbox", "intent_matchbox"}:
        return (
            "Mic · 16 kHz → log-mel 40 × T (≤5 s) → "
            "MatchboxNet M · width 104 · 4 blocks · attn pool → "
            "intent head (20) + slot heads (6×3)"
        )
    if name in {"ctc", "ctc_low", "low"}:
        return (
            "Mic · 16 kHz → log-mel 40 × T → "
            "ResCTCEncoderLow · hidden 192 · 5 blocks · 2× downsample CTC → "
            "grammar trie decode"
        )
    return str(name or "—")


def _rows(pairs: List[tuple[str, str]]) -> str:
    parts = []
    for k, v in pairs:
        parts.append(
            f"<tr><th>{html.escape(k)}</th><td>{v}</td></tr>"
        )
    return "\n".join(parts)


def render_arch_diagram(kind: str, model_name: Optional[str] = None) -> str:
    """Deck-style Model flowchart (matches class results slide)."""
    name = (model_name or kind or "").lower()
    if kind == "wake" or name == "wake":
        return """
<div class="arch" aria-label="Model architecture">
  <div class="arch-title">Model</div>
  <div class="box box-in">Mic · 16 kHz → log-mel 40 × T (~1.5 s)</div>
  <div class="box box-enc">MicroWakeNet · DS-conv stack · binary head</div>
  <div class="box box-enc2">Always-on · energy-gated · threshold calibrated</div>
  <div class="arch-row">
    <div class="box box-intent">Wake / not-wake</div>
    <div class="box box-slot">Threshold gate</div>
  </div>
  <div class="arch-row">
    <div class="box box-act">Listen unlock</div>
    <div class="box box-pi">Mac / Raspberry Pi</div>
  </div>
</div>
"""
    if name in {"ctc", "ctc_low", "low"} or "ctc" in name:
        return """
<div class="arch" aria-label="Model architecture">
  <div class="arch-title">Model</div>
  <div class="box box-in">Mic · 16 kHz → log-mel 40 × T (≤5.0 s)</div>
  <div class="box box-enc">ResCTCEncoderLow · hidden 192 · 5 blocks · 2× downsample</div>
  <div class="box box-enc2">Character CTC · grammar trie constrained decode</div>
  <div class="arch-row">
    <div class="box box-intent">Phrase → intent</div>
    <div class="box box-slot">Phrase → slot</div>
  </div>
  <div class="arch-row">
    <div class="box box-act">Actuator</div>
    <div class="box box-pi">Mac / Raspberry Pi</div>
  </div>
</div>
"""
    if name in {"conformer", "intent_conformer", "tiny_conformer"}:
        return """
<div class="arch" aria-label="Model architecture">
  <div class="arch-title">Model</div>
  <div class="box box-in">Mic · 16 kHz → log-mel 40 × T (≤5.0 s)</div>
  <div class="box box-enc">Tiny Conformer M · d=64 · 4 blocks · 4 heads · FF×2</div>
  <div class="box box-enc2">Conv subsample · attn pool · v2 recipe (real-weighted + SpecAug + misfire)</div>
  <div class="arch-row">
    <div class="box box-intent">Intent head (20)</div>
    <div class="box box-slot">Slot heads (6 × 3)</div>
  </div>
  <div class="arch-row">
    <div class="box box-act">Actuator</div>
    <div class="box box-pi">Mac / Raspberry Pi</div>
  </div>
</div>
"""
    if name in {"matchbox", "intent_matchbox"}:
        return """
<div class="arch" aria-label="Model architecture">
  <div class="arch-title">Model</div>
  <div class="box box-in">Mic · 16 kHz → log-mel 40 × T (≤5.0 s)</div>
  <div class="box box-enc">MatchboxNet M · width 104 · 4 separable blocks</div>
  <div class="box box-enc2">Time-channel sep conv · attn pool · v2 recipe</div>
  <div class="arch-row">
    <div class="box box-intent">Intent head (20)</div>
    <div class="box box-slot">Slot heads (6 × 3)</div>
  </div>
  <div class="arch-row">
    <div class="box box-act">Actuator</div>
    <div class="box box-pi">Mac / Raspberry Pi</div>
  </div>
</div>
"""
    # Default: IntentCRNN (class deck layout)
    return """
<div class="arch" aria-label="Model architecture">
  <div class="arch-title">Model</div>
  <div class="box box-in">Mic · 16 kHz → log-mel 40 × T (≤5.0 s)</div>
  <div class="box box-enc">CRNN encoder · 4 DS-conv blocks · 2-layer BiGRU 96</div>
  <div class="box box-enc2">4-head attention pool · non-causal, gated by wake word</div>
  <div class="arch-row">
    <div class="box box-intent">Intent head (20)</div>
    <div class="box box-slot">Slot heads (6 × 3)</div>
  </div>
  <div class="arch-row">
    <div class="box box-act">Actuator</div>
    <div class="box box-pi">Mac / Raspberry Pi</div>
  </div>
</div>
"""


def render_command_card(card: Dict[str, Any]) -> str:
    m, t, v, d = card["model"], card["train"], card["validation"], card["dataset"]
    model_tbl = _rows([
        ("Parameters / weights",
         f"{_num(m.get('params'))} · {html.escape(str(m.get('onnx_mb') or '—'))}"),
        ("Decode", html.escape(str(m.get("decode") or "—"))),
        ("Best epoch", html.escape(str(m.get("best_epoch") or "—"))),
    ])
    train_tbl = _rows([
        ("Cluster", html.escape(str(t.get("cluster") or "—"))),
        ("Optimizer",
         f"Adam · lr {html.escape(str(t.get('lr') or '—'))} · "
         f"batch {html.escape(str(t.get('batch_size') or '—'))} · "
         f"AMP={html.escape(str(t.get('amp')))}"),
        ("Epochs / seed",
         f"{html.escape(str(t.get('epochs') or '—'))} · seed "
         f"{html.escape(str(t.get('seed') or '—'))}"),
        ("Best val command", _pct(t.get("best_val_command_acc"))),
        ("Finished", html.escape(str(t.get("finished_at") or "—"))),
    ])
    val_tbl = _rows([
        ("Keyword / intent (val)",
         f"{_pct(v.get('val_cmd'))} / {_pct(v.get('val_intent'))}"),
        ("Test command / intent",
         f"{_pct(v.get('test_cmd'))} / {_pct(v.get('test_intent'))}"),
        ("Real speech / synthetic",
         f"<strong>{_pct(v.get('test_real'))}</strong> / {_pct(v.get('test_synth'))}"),
        ("False accept / reject (test)",
         f"{_pct(v.get('test_far'))} / {_pct(v.get('test_frr'))}"),
        ("Reject gate min_confidence",
         html.escape(str(v.get("reject_min_confidence") or "—"))),
        ("Test clips (real)",
         f"{_num(v.get('test_n'))} ({_num(v.get('test_real_n'))} real)"),
    ])
    data_tbl = _rows([
        ("Source", html.escape(str(d.get("source") or "—"))),
        ("Hours / utts",
         f"train {html.escape(str(d.get('train_hours')))} · "
         f"test {html.escape(str(d.get('test_hours')))}"),
        ("Labels", html.escape(str(d.get("labels") or "—"))),
        ("Fingerprint", html.escape(str(d.get("fingerprint") or "—"))),
    ])
    note = html.escape((card.get("notes") or "").strip())
    arch = render_arch_diagram(card.get("kind") or "command", m.get("name"))
    return f"""
<section class="slide" id="{html.escape(card['tag'])}">
  <div class="slide-top">
    <div class="ver">{html.escape(str(card.get('version')))}</div>
    <div class="titles">
      <h2>{html.escape(card.get('title') or card['tag'])}</h2>
      <p class="sub">{html.escape(card.get('subtitle') or '')}</p>
      <p class="meta">{html.escape(str(card.get('recorded_at') or ''))}
        · <code>{html.escape(card['tag'])}</code>
        · <code>{html.escape(str(card.get('experiment') or ''))}</code></p>
    </div>
  </div>
  <div class="quad">
    <div class="panel model-panel">{arch}<table>{model_tbl}</table></div>
    <div class="panel"><h3>Training on A100</h3><table>{train_tbl}</table></div>
    <div class="panel"><h3>Validation</h3><table>{val_tbl}</table></div>
    <div class="panel"><h3>Dataset</h3><table>{data_tbl}</table></div>
  </div>
  {"<p class='notes'>" + note + "</p>" if note else ""}
</section>
"""


def render_wake_card(card: Dict[str, Any]) -> str:
    m, t, v, d = card["model"], card["train"], card["validation"], card["dataset"]
    model_tbl = _rows([
        ("Parameters / weights",
         f"{_num(m.get('params'))} · {html.escape(str(m.get('onnx_mb') or '—'))}"),
        ("Calibrated threshold", f"{float(m.get('threshold') or 0):.3f}"),
    ])
    train_tbl = _rows([
        ("Cluster", html.escape(str(t.get("cluster") or "—"))),
        ("Train + / −",
         f"{_num(t.get('train_positive'))} / {_num(t.get('train_negative'))}"),
        ("Val + / −",
         f"{_num(t.get('val_positive'))} / {_num(t.get('val_negative'))}"),
    ])
    val_tbl = _rows([
        ("Balanced acc / accuracy",
         f"{_pct(v.get('balanced_acc'))} / {_pct(v.get('accuracy'))}"),
        ("Precision / recall",
         f"{_pct(v.get('precision'))} / {_pct(v.get('recall'))}"),
        ("False accept / reject",
         f"{_pct(v.get('far'))} / {_pct(v.get('frr'))}"),
        ("Threshold", f"{float(v.get('threshold') or 0):.3f}"),
    ])
    data_tbl = _rows([
        ("Source", html.escape(str(d.get("source") or "—"))),
        ("Positives", _num(d.get("positives"))),
        ("Negatives", html.escape(str(d.get("negatives_split") or "—"))),
    ])
    note = html.escape((card.get("notes") or "").strip())
    arch = render_arch_diagram("wake", "wake")
    return f"""
<section class="slide wake" id="{html.escape(card['tag'])}">
  <div class="slide-top">
    <div class="ver wake">{html.escape(str(card.get('version')))}</div>
    <div class="titles">
      <h2>{html.escape(card.get('title') or card['tag'])}</h2>
      <p class="sub">{html.escape(card.get('subtitle') or '')}</p>
      <p class="meta">{html.escape(str(card.get('recorded_at') or ''))}
        · <code>{html.escape(card['tag'])}</code></p>
    </div>
  </div>
  <div class="quad">
    <div class="panel model-panel">{arch}<table>{model_tbl}</table></div>
    <div class="panel"><h3>Training</h3><table>{train_tbl}</table></div>
    <div class="panel"><h3>Validation</h3><table>{val_tbl}</table></div>
    <div class="panel"><h3>Dataset</h3><table>{data_tbl}</table></div>
  </div>
  {"<p class='notes'>" + note + "</p>" if note else ""}
</section>
"""


def render_html(cards: List[Dict[str, Any]]) -> str:
    # newest first
    cards = list(reversed(cards))
    body = []
    for c in cards:
        if c.get("kind") == "wake":
            body.append(render_wake_card(c))
        else:
            body.append(render_command_card(c))
    toc_parts = []
    for c in cards:
        tag = c["tag"]
        ver = normalize_version(tag, c.get("version"))
        kind = "wake" if c.get("kind") == "wake" else "command"
        ver_cls = "toc-ver wake" if kind == "wake" else "toc-ver"
        toc_parts.append(
            f'<li><a href="#{html.escape(tag)}">'
            f'<span class="{ver_cls}">{html.escape(ver)}</span>'
            f'<span class="toc-kind">{kind}</span>'
            f'<code>{html.escape(tag)}</code></a></li>'
        )
    toc = "".join(toc_parts)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>ME2 · VCM scoreboard</title>
<style>
  :root {{
    --bg: #e8e4dc;
    --slide: #f7f4ee;
    --ink: #1c1a17;
    --muted: #5c564c;
    --line: #cfc7b8;
    --maroon: #7a1f2b;
    --wake: #b7791f;
    --ok: #1f7a4d;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 1.5rem;
    font: 13.5px/1.4 "Source Sans 3", "Helvetica Neue", Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--ink);
  }}
  .page-title {{ margin: 0 0 .25rem; color: var(--maroon); font-size: 1.45rem; }}
  .lead {{ color: var(--muted); margin: 0 0 1rem; max-width: 70rem; }}
  nav {{
    background: #fff; border: 1px solid var(--line); border-radius: 4px;
    padding: .85rem 1rem; margin-bottom: 1.25rem;
  }}
  nav ol {{
    margin: .4rem 0 0; padding-left: 1.2rem;
    columns: 1; column-count: 1; max-width: 36rem;
  }}
  nav li {{ break-inside: avoid; margin: .2rem 0; }}
  nav a {{ color: var(--maroon); text-decoration: none; }}
  nav a:hover {{ text-decoration: underline; }}
  .toc-ver {{
    display: inline-block; min-width: 2rem; text-align: center; font-weight: 700;
    color: #fff; background: var(--maroon); padding: 0 .35rem; border-radius: 3px;
    font-size: .75rem; margin-right: .35rem;
  }}
  .toc-ver.wake {{ background: var(--wake); }}
  .toc-kind {{
    display: inline-block; min-width: 4.2rem; color: var(--muted);
    font-size: .72rem; text-transform: uppercase; letter-spacing: .04em;
    margin-right: .35rem;
  }}
  .slide {{
    background: var(--slide); border: 1px solid var(--line);
    box-shadow: 0 8px 24px rgba(40,30,10,.08);
    padding: 1.1rem 1.25rem 1.2rem; margin: 0 auto 1.5rem;
    max-width: 1100px; aspect-ratio: 988 / 790;
    display: flex; flex-direction: column;
  }}
  @media (max-width: 900px) {{
    .slide {{ aspect-ratio: auto; }}
  }}
  .slide.wake {{ border-top: 4px solid var(--wake); }}
  .slide-top {{ display: flex; gap: .85rem; align-items: flex-start; }}
  .ver {{
    background: var(--maroon); color: #fff; font-weight: 800;
    padding: .4rem .55rem; border-radius: 3px; font-size: .9rem;
  }}
  .ver.wake {{ background: var(--wake); }}
  h2 {{ margin: 0; font-size: 1.2rem; color: var(--maroon); }}
  .sub {{ margin: .15rem 0 0; color: var(--muted); font-size: .92rem; }}
  .meta {{ margin: .25rem 0 0; color: var(--muted); font-size: .8rem; }}
  code {{
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    background: #efe9df; padding: .05rem .3rem; border-radius: 3px; font-size: .8rem;
  }}
  .pipeline {{
    margin: .75rem 0 .85rem; padding: .55rem .7rem;
    background: #fff; border: 1px solid var(--line); border-radius: 3px;
    font-size: .82rem; color: #333; line-height: 1.35;
  }}
  .quad {{
    flex: 1; display: grid; gap: .75rem;
    grid-template-columns: 1fr 1fr; grid-template-rows: 1fr 1fr;
  }}
  .panel {{
    background: #fff; border: 1px solid var(--line); border-radius: 3px;
    padding: .45rem .55rem .55rem; overflow: auto;
  }}
  .model-panel {{ display: flex; flex-direction: column; gap: .45rem; }}
  .arch {{
    font-size: .72rem; line-height: 1.25;
  }}
  .arch-title {{
    font-size: .72rem; letter-spacing: .08em; text-transform: uppercase;
    color: var(--maroon); font-weight: 700; margin-bottom: .3rem;
    border-bottom: 1px solid var(--line); padding-bottom: .15rem;
  }}
  .box {{
    border: 1.5px solid #888; border-radius: 4px; padding: .28rem .4rem;
    margin: 0 0 .28rem; background: #faf8f4; text-align: center;
    font-weight: 600; color: #222;
  }}
  .box-in {{ border-color: #5b9bd5; background: #eef5fb; }}
  .box-enc {{ border-color: #70ad47; background: #f1f8ea; }}
  .box-enc2 {{ border-color: #70ad47; background: #f1f8ea; font-weight: 500; }}
  .arch-row {{
    display: grid; grid-template-columns: 1fr 1fr; gap: .28rem;
    margin-bottom: .28rem;
  }}
  .box-intent {{ border-color: #ed7d31; background: #fff4ec; }}
  .box-slot {{ border-color: #c45911; background: #fceee6; }}
  .box-act {{ border-color: #7030a0; background: #f5eef9; }}
  .box-pi {{ border-color: #548235; background: #eef6e8; }}
  h3 {{
    margin: 0 0 .3rem; font-size: .72rem; letter-spacing: .08em;
    text-transform: uppercase; color: var(--maroon); border-bottom: 1px solid var(--line);
    padding-bottom: .2rem;
  }}
  table {{ width: 100%; border-collapse: collapse; font-size: .82rem; }}
  th, td {{
    border-top: 1px solid #ebe4d8; padding: .28rem .2rem;
    vertical-align: top; text-align: left;
  }}
  th {{ width: 42%; color: var(--muted); font-weight: 600; }}
  .notes {{
    margin: .7rem 0 0; padding: .45rem .6rem; background: #fff;
    border-left: 3px solid var(--maroon); color: var(--muted); font-size: .8rem;
  }}
  strong {{ color: var(--ok); }}
  @media print {{
    body {{ background: #fff; padding: 0; }}
    nav {{ display: none; }}
    .slide {{
      box-shadow: none; page-break-after: always; margin: 0;
      border: none; max-width: none; width: 100%; height: 100vh;
    }}
  }}
</style>
</head>
<body>
  <h1 class="page-title">ME2 · Voice Controlled Smart Device — scoreboard</h1>
  <p class="lead">Slide-style version cards (newest first), matching the class results deck layout.
  Each finished retrain appends a new version. File:
  <code>me2/grammar_asr/runs/SCOREBOARD.html</code></p>
  <nav>
    <strong>Version index</strong>
    <ol>{toc or "<li>No runs yet</li>"}</ol>
  </nav>
  {"".join(body) if body else "<p class='lead'>No scoreboard entries yet.</p>"}
</body>
</html>
"""


def write_scoreboard_html(runs_dir: Optional[Path] = None) -> Path:
    runs_dir = runs_dir or RUNS_DIR
    global HISTORY, HTML_OUT
    HISTORY = runs_dir / "scoreboard_history.jsonl"
    HTML_OUT = runs_dir / "SCOREBOARD.html"
    cards = load_history()
    HTML_OUT.write_text(render_html(cards), encoding="utf-8")
    return HTML_OUT


def append_scoreboard_version(card: Dict[str, Any],
                              runs_dir: Optional[Path] = None) -> Path:
    """Append one versioned card and regenerate HTML."""
    runs_dir = runs_dir or RUNS_DIR
    global HISTORY, HTML_OUT
    HISTORY = runs_dir / "scoreboard_history.jsonl"
    HTML_OUT = runs_dir / "SCOREBOARD.html"
    card = dict(card)
    card["version"] = normalize_version(str(card.get("tag") or ""), card.get("version"))
    # Avoid duplicate consecutive same tag+finished
    prev = load_history()
    if prev and prev[-1].get("tag") == card.get("tag") and prev[-1].get(
        "validation"
    ) == card.get("validation"):
        return write_scoreboard_html(runs_dir)
    append_history(card)
    return write_scoreboard_html(runs_dir)


def _discover_command_runs(runs_dir: Path) -> List[Path]:
    """All finished command runs that have summary.json (sorted by finished_at)."""
    found: List[tuple[str, Path]] = []
    for summary in sorted(runs_dir.glob("*/summary.json")):
        try:
            s = json.loads(summary.read_text(encoding="utf-8"))
        except Exception:
            continue
        key = str(s.get("finished_at") or summary.stat().st_mtime)
        found.append((key, summary))
    found.sort(key=lambda x: x[0])
    return [p for _, p in found]


def _discover_wake_runs(runs_dir: Path) -> List[Path]:
    return sorted(runs_dir.glob("*/best.metrics.json"))


def sync_all_runs(runs_dir: Optional[Path] = None, *, force: bool = False) -> Path:
    """Rebuild (force) or append-missing scoreboard cards from every finished run.

    Call after pulling DGX artifacts so SCOREBOARD.html matches disk.
    """
    runs_dir = runs_dir or RUNS_DIR
    global HISTORY
    HISTORY = runs_dir / "scoreboard_history.jsonl"
    if force and HISTORY.exists():
        HISTORY.write_text("", encoding="utf-8")

    existing_tags = {c.get("tag") for c in load_history()}

    for summary in _discover_command_runs(runs_dir):
        tag = summary.parent.name
        if not force and tag in existing_tags:
            continue
        s = json.loads(summary.read_text(encoding="utf-8"))
        model = s.get("model") or "intent"
        version = normalize_version(tag, s.get("version"))
        experiment = s.get("experiment") or model
        if tag == "ctc_hf_s0":
            title = "ME2 command · CTC grammar · v0"
        else:
            title = f"ME2 command · {experiment} · {version}"
        append_scoreboard_version(card_from_command_summary(
            summary,
            version=version,
            experiment=experiment,
            title=title,
        ), runs_dir)
        existing_tags.add(tag)

    wake_notes = {
        "wake_boss": "Legacy wake (pre-kaki rename).",
        "wake_kaki": "First kaki wake baseline.",
        "wake_kaki_v2": "Multi-speaker positives (hdc/fermina/keith).",
        "wake_kaki_v3": "Added hard negatives (good/cookie/khaki/…).",
        "wake_kaki_v4": "Mac ambient hard-negs (silence/typing/noise/speech).",
        "wake_kaki_v5": "Cleaned pos/neg after manual audit; Mac ambient kept.",
    }
    for metrics in _discover_wake_runs(runs_dir):
        tag = metrics.parent.name
        if not force and tag in existing_tags:
            continue
        note = wake_notes.get(tag, "")
        ver = normalize_version(tag)
        wake_name = "boss" if "boss" in tag else "kaki"
        append_scoreboard_version(card_from_wake_metrics(
            metrics,
            version=ver,
            notes=note,
            title=f"ME2 wake · {wake_name} · {ver}",
        ), runs_dir)
        existing_tags.add(tag)

    return write_scoreboard_html(runs_dir)


def seed_from_disk(runs_dir: Optional[Path] = None, force: bool = False) -> Path:
    """Import finished runs into history if empty (or force rebuild)."""
    runs_dir = runs_dir or RUNS_DIR
    global HISTORY
    HISTORY = runs_dir / "scoreboard_history.jsonl"
    if HISTORY.exists() and HISTORY.read_text().strip() and not force:
        # Still pick up any new runs that finished since last rebuild.
        return sync_all_runs(runs_dir, force=False)
    return sync_all_runs(runs_dir, force=True)


def append_from_run_dir(run_dir: Path, runs_dir: Optional[Path] = None,
                        **extra) -> Path:
    runs_dir = runs_dir or RUNS_DIR
    summary = run_dir / "summary.json"
    wake = run_dir / "best.metrics.json"
    if summary.exists():
        card = card_from_command_summary(summary, **extra)
        return append_scoreboard_version(card, runs_dir)
    if wake.exists():
        card = card_from_wake_metrics(wake, **extra)
        return append_scoreboard_version(card, runs_dir)
    raise FileNotFoundError(f"No summary.json or best.metrics.json in {run_dir}")

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", action="store_true",
                    help="Import existing runs into empty history")
    ap.add_argument("--force-seed", action="store_true",
                    help="Rebuild history from disk (clears prior HTML history)")
    ap.add_argument("--sync", action="store_true",
                    help="Append any finished runs missing from the scoreboard")
    ap.add_argument("--from-run", type=str, default=None,
                    help="Append card from runs/<tag>")
    ap.add_argument("--version", type=str, default=None)
    ap.add_argument("--notes", type=str, default=None)
    args = ap.parse_args(argv)

    if args.force_seed:
        path = seed_from_disk(force=True)
    elif args.seed or args.sync:
        path = seed_from_disk(force=False)
    elif args.from_run:
        extra = {}
        if args.version:
            extra["version"] = args.version
        if args.notes:
            extra["notes"] = args.notes
        path = append_from_run_dir(RUNS_DIR / args.from_run, **extra)
    else:
        # Default: sync missing runs then rewrite HTML
        path = sync_all_runs(force=False)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
