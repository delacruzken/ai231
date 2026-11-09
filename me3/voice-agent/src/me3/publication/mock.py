"""Mock publisher for structural tests without a GitHub PAT."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from me3.publication.sanitize import collect_publishable_files


class MockPublisher:
    provider = "mock"

    def __init__(self, *, owner: str = "mock-owner", repo: str = "mock-site", site_root: str = "docs"):
        self.owner = owner
        self.repo = repo
        self.site_root = site_root.strip().strip("/")
        self.last_publish: dict[str, Any] | None = None
        self._published_marker = "ME3 mock site"

    def health(self) -> dict[str, Any]:
        return {
            "healthy": True,
            "provider": self.provider,
            "owner": self.owner,
            "repo": self.repo,
        }

    def pages_info(self) -> dict[str, Any]:
        return {
            "configured": True,
            "html_url": f"https://{self.owner}.github.io/{self.repo}",
            "status": "built",
        }

    def default_pages_url(self) -> str:
        return f"https://{self.owner}.github.io/{self.repo}"

    def publish_directory(
        self,
        source_dir: Path,
        *,
        message: str | None = None,
        confirm: bool = False,
    ) -> dict[str, Any]:
        if not confirm:
            raise RuntimeError("confirm=true is required for publication")
        files, rejects = collect_publishable_files(source_dir)
        if not files:
            raise RuntimeError("nothing to publish: " + "; ".join(rejects[:8]))
        source_dir = source_dir.resolve()
        published = [
            f"{self.site_root}/{p.relative_to(source_dir).as_posix()}" if self.site_root else p.relative_to(source_dir).as_posix()
            for p in files
        ]
        sha = uuid4().hex
        result = {
            "provider": self.provider,
            "commit_sha": sha,
            "commit_url": f"https://github.com/{self.owner}/{self.repo}/commit/{sha}",
            "branch": "main",
            "files": published,
            "file_count": len(published),
            "pages": self.pages_info(),
            "pages_url": self.default_pages_url(),
            "message": message or "ME3 mock publish",
            "rejects_skipped": [r for r in rejects if r.startswith("skip") or r.startswith("disallowed")],
        }
        self.last_publish = result
        return result

    def validate_pages(
        self,
        *,
        expect_substring: str | None = None,
        retries: int = 1,
        delay_s: float = 0.0,
    ) -> dict[str, Any]:
        _ = retries, delay_s
        url = self.default_pages_url() + "/"
        body = f"<html><body>{self._published_marker}</body></html>"
        if self.last_publish:
            body = f"<html><body>{self._published_marker} commit={self.last_publish['commit_sha'][:8]}</body></html>"
        ok = True
        if expect_substring:
            ok = expect_substring.lower() in body.lower() or expect_substring.lower() in self._published_marker.lower()
            # Mock treats any expect as ok after a successful publish unless it's a known poison string
            if expect_substring.lower() == "__missing__":
                ok = False
            elif self.last_publish is not None:
                ok = True
        return {
            "ok": ok,
            "url": url,
            "http_status": 200 if ok else 404,
            "pages": self.pages_info(),
            "expect_substring": expect_substring,
            "error": None if ok else "mock pages content missing",
            "snippet": body[:240],
        }
