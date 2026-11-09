"""Phase 7: sanitize, mock publish, confirm gate, idempotency."""

from __future__ import annotations

from pathlib import Path

from me3.mcp.common import RequestMeta
from me3.publication.sanitize import validate_source
from me3.publication.service import PublicationService


def _site(tmp: Path, *, poison: bool = False) -> Path:
    site = tmp / ("poison" if poison else "site")
    site.mkdir(parents=True, exist_ok=True)
    (site / "index.html").write_text("<html><body>Hello ME3</body></html>\n", encoding="utf-8")
    if poison:
        (site / ".env").write_text("SECRET=1\n", encoding="utf-8")
    return site


def test_validate_rejects_env(tmp_path: Path):
    good = validate_source(_site(tmp_path))
    assert good["ok"] is True
    bad = validate_source(_site(tmp_path, poison=True))
    assert bad["ok"] is False


def test_confirm_required(tmp_path: Path):
    svc = PublicationService(force_mock=True, db_path=tmp_path / "p.db")
    site = _site(tmp_path)
    r = svc.handle("publish.site", source_dir=str(site), confirm=False)
    assert not r.ok
    assert "confirm" in (r.error.message if r.error else "").lower()
    svc.close()


def test_mock_publish_and_status(tmp_path: Path):
    svc = PublicationService(force_mock=True, db_path=tmp_path / "p.db")
    site = _site(tmp_path)
    r = svc.handle(
        "publish.site",
        RequestMeta(idempotency_key="same"),
        source_dir=str(site),
        confirm=True,
        message="test",
    )
    assert r.ok
    assert r.data["publication"]["commit_sha"]
    assert r.data["pages"]["ok"] is True
    replay = svc.handle(
        "publish.site",
        RequestMeta(idempotency_key="same"),
        source_dir=str(site),
        confirm=True,
    )
    assert replay.data.get("idempotent_replay") is True
    status = svc.handle("publish.status")
    assert status.ok
    assert len(status.data["recent"]) >= 1
    svc.close()
