"""Phase 7 sample bank: sanitize, mock publish, optional live GitHub."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from me3.config import ROOT, load_secrets
from me3.mcp.common import RequestMeta
from me3.publication.sanitize import validate_source
from me3.publication.service import PublicationService


def _demo_site(out: Path) -> Path:
    site = out / "demo-site"
    site.mkdir(parents=True, exist_ok=True)
    (site / "index.html").write_text(
        "<!DOCTYPE html><html><body><h1>ME3 Phase 7 Demo</h1>"
        "<p>Published by sample bank.</p></body></html>\n",
        encoding="utf-8",
    )
    (site / "style.css").write_text("body { font-family: sans-serif; }\n", encoding="utf-8")
    return site


def run_structural(out: Path) -> dict:
    site = _demo_site(out)
    # poison file must be rejected
    poison = out / "poison-site"
    poison.mkdir(parents=True, exist_ok=True)
    (poison / "index.html").write_text("<h1>ok</h1>\n", encoding="utf-8")
    (poison / ".env").write_text("GITHUB_PAT=ghp_should_not_publish\n", encoding="utf-8")

    cases: list[dict] = []
    good = validate_source(site)
    cases.append({"case": "validate_clean_site", "ok": good["ok"] and good["file_count"] >= 2})

    bad = validate_source(poison)
    cases.append(
        {
            "case": "reject_env_file",
            "ok": (not bad["ok"]) and any("blocked name" in r for r in bad.get("rejects") or []),
        }
    )

    svc = PublicationService(force_mock=True, db_path=out / "pub.db")
    need = svc.handle("publish.site", source_dir=str(site), confirm=False)
    cases.append(
        {
            "case": "confirm_required",
            "ok": (not need.ok) and "confirm" in (need.error.message if need.error else "").lower(),
        }
    )

    pub = svc.handle(
        "publish.site",
        RequestMeta(idempotency_key="p7-1"),
        source_dir=str(site),
        confirm=True,
        message="ME3 phase7 sample",
        expect_substring="ME3",
    )
    cases.append(
        {
            "case": "mock_publish",
            "ok": pub.ok
            and bool((pub.data.get("publication") or {}).get("commit_sha"))
            and bool((pub.data.get("pages") or {}).get("ok")),
            "provider": svc.provider,
        }
    )

    replay = svc.handle(
        "publish.site",
        RequestMeta(idempotency_key="p7-1"),
        source_dir=str(site),
        confirm=True,
    )
    cases.append(
        {
            "case": "idempotent_publish",
            "ok": replay.ok and replay.data.get("idempotent_replay") is True,
        }
    )

    status = svc.handle("publish.status")
    cases.append(
        {
            "case": "status_lists_recent",
            "ok": status.ok and len(status.data.get("recent") or []) >= 1,
        }
    )
    svc.close()
    return {"structural_ok": all(c["ok"] for c in cases), "cases": cases}


def run_live(out: Path) -> dict:
    secrets = load_secrets()
    if not (
        secrets.github_pat.strip()
        and secrets.github_owner.strip()
        and secrets.github_repo.strip()
    ):
        return {"live_ok": False, "blocked": True, "reason": "GITHUB_PAT/OWNER/REPO not set"}

    site = _demo_site(out / "live")
    marker = f"ME3-live-{datetime.now(timezone.utc).strftime('%H%M%S')}"
    (site / "index.html").write_text(
        f"<!DOCTYPE html><html><body><h1>ME3 Live Publish</h1>"
        f"<p>{marker}</p></body></html>\n",
        encoding="utf-8",
    )
    svc = PublicationService(force_mock=False, db_path=out / "live.db")
    try:
        health = svc.health()
        pub = svc.handle(
            "publish.site",
            RequestMeta(idempotency_key=f"p7-live-{marker}"),
            source_dir=str(site),
            confirm=True,
            message=f"ME3 phase7 live {marker}",
            expect_substring="ME3 Live Publish",
        )
        return {
            "live_ok": bool(pub.ok and health.get("healthy")),
            "blocked": False,
            "provider": svc.provider,
            "health": health,
            "publication": (pub.data.get("publication") if pub.ok else None),
            "pages": (pub.data.get("pages") if pub.ok else None),
            "error": None if pub.ok else (pub.error.message if pub.error else "failed"),
            "marker": marker,
        }
    finally:
        svc.close()


def main() -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "artifacts" / "publication_phase7" / stamp
    out.mkdir(parents=True, exist_ok=True)
    structural = run_structural(out)
    live = run_live(out)
    report = {
        "ts": stamp,
        "structural": structural,
        "live": live,
        "ok": structural["structural_ok"] and (live.get("live_ok") or live.get("blocked")),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not structural["structural_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
