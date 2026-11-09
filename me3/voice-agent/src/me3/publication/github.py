"""GitHub Contents / Git Data API client for Pages publication."""

from __future__ import annotations

import base64
import json
import logging
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from uuid import uuid4

from me3.publication.sanitize import collect_publishable_files

logger = logging.getLogger("me3.publication.github")

API = "https://api.github.com"


class GitHubPublisher:
    def __init__(
        self,
        *,
        token: str,
        owner: str,
        repo: str,
        branch: str = "main",
        site_root: str = "docs",
        pages_url: str | None = None,
        timeout_s: float = 45.0,
    ):
        self.token = token.strip()
        self.owner = owner.strip()
        self.repo = repo.strip()
        self.branch = branch.strip() or "main"
        self.site_root = (site_root or "docs").strip().strip("/")
        self.pages_url_override = (pages_url or "").strip() or None
        self.timeout_s = timeout_s

    @property
    def provider(self) -> str:
        return "github"

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        raw: bool = False,
    ) -> Any:
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"{API}{path}",
            data=data,
            method=method,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self.token}",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "me3-voice-agent",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw_body = resp.read()
                if raw:
                    return raw_body
                if not raw_body:
                    return {}
                return json.loads(raw_body.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:800]
            logger.warning("github %s %s → %s %s", method, path, exc.code, detail)
            raise RuntimeError(f"GitHub {method} {path} HTTP {exc.code}: {detail}") from exc

    def health(self) -> dict[str, Any]:
        try:
            meta = self._request("GET", f"/repos/{self.owner}/{self.repo}")
            return {
                "healthy": True,
                "provider": self.provider,
                "owner": self.owner,
                "repo": self.repo,
                "default_branch": meta.get("default_branch"),
                "private": meta.get("private"),
            }
        except Exception as exc:
            return {"healthy": False, "provider": self.provider, "error": str(exc)[:300]}

    def pages_info(self) -> dict[str, Any]:
        try:
            info = self._request("GET", f"/repos/{self.owner}/{self.repo}/pages")
        except RuntimeError as exc:
            if "404" in str(exc):
                return {
                    "configured": False,
                    "html_url": self.default_pages_url(),
                    "status": "not_configured",
                }
            raise
        return {
            "configured": True,
            "html_url": info.get("html_url") or self.default_pages_url(),
            "status": info.get("status"),
            "build_type": info.get("build_type"),
            "source": info.get("source"),
        }

    def default_pages_url(self) -> str:
        if self.pages_url_override:
            return self.pages_url_override.rstrip("/")
        # user/org site vs project site
        if self.repo.lower() == f"{self.owner.lower()}.github.io":
            return f"https://{self.owner}.github.io"
        return f"https://{self.owner}.github.io/{self.repo}"

    def publish_directory(
        self,
        source_dir: Path,
        *,
        message: str | None = None,
        confirm: bool = False,
    ) -> dict[str, Any]:
        if not confirm:
            raise RuntimeError("confirm=true is required for live GitHub publication")
        files, rejects = collect_publishable_files(source_dir)
        if not files:
            raise RuntimeError("nothing to publish: " + "; ".join(rejects[:8]))

        source_dir = source_dir.resolve()
        ref = self._request("GET", f"/repos/{self.owner}/{self.repo}/git/ref/heads/{self.branch}")
        head_sha = ref["object"]["sha"]
        commit = self._request("GET", f"/repos/{self.owner}/{self.repo}/git/commits/{head_sha}")
        base_tree = commit["tree"]["sha"]

        tree_items: list[dict[str, Any]] = []
        published: list[str] = []
        for path in files:
            rel = path.relative_to(source_dir).as_posix()
            dest = f"{self.site_root}/{rel}" if self.site_root else rel
            content = path.read_bytes()
            blob = self._request(
                "POST",
                f"/repos/{self.owner}/{self.repo}/git/blobs",
                body={
                    "content": base64.b64encode(content).decode("ascii"),
                    "encoding": "base64",
                },
            )
            tree_items.append(
                {
                    "path": dest,
                    "mode": "100644",
                    "type": "blob",
                    "sha": blob["sha"],
                }
            )
            published.append(dest)

        tree = self._request(
            "POST",
            f"/repos/{self.owner}/{self.repo}/git/trees",
            body={"base_tree": base_tree, "tree": tree_items},
        )
        msg = (message or f"ME3 publish {uuid4().hex[:8]}").strip()
        new_commit = self._request(
            "POST",
            f"/repos/{self.owner}/{self.repo}/git/commits",
            body={
                "message": msg,
                "tree": tree["sha"],
                "parents": [head_sha],
            },
        )
        self._request(
            "PATCH",
            f"/repos/{self.owner}/{self.repo}/git/refs/heads/{self.branch}",
            body={"sha": new_commit["sha"]},
        )
        pages = self.pages_info()
        return {
            "provider": self.provider,
            "commit_sha": new_commit["sha"],
            "commit_url": new_commit.get("html_url")
            or f"https://github.com/{self.owner}/{self.repo}/commit/{new_commit['sha']}",
            "branch": self.branch,
            "files": published,
            "file_count": len(published),
            "pages": pages,
            "pages_url": pages.get("html_url") or self.default_pages_url(),
            "rejects_skipped": [r for r in rejects if r.startswith("skip") or r.startswith("disallowed")],
        }

    def validate_pages(
        self,
        *,
        expect_substring: str | None = None,
        retries: int = 3,
        delay_s: float = 2.0,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        # Keep voice turns snappy: Pages HTML often lags the git commit.
        check_timeout = 8.0 if timeout_s is None else float(timeout_s)
        pages = self.pages_info()
        url = (pages.get("html_url") or self.default_pages_url()).rstrip("/") + "/"
        last_err = ""
        body = ""
        status = 0
        for attempt in range(max(1, retries)):
            try:
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "me3-voice-agent"},
                    method="GET",
                )
                with urllib.request.urlopen(req, timeout=check_timeout) as resp:
                    status = resp.status
                    body = resp.read().decode("utf-8", errors="replace")
                if status < 400:
                    break
            except Exception as exc:
                last_err = str(exc)[:300]
                status = 0
            if attempt + 1 < retries:
                time.sleep(delay_s)

        ok = 200 <= status < 400
        if ok and expect_substring:
            ok = expect_substring.lower() in body.lower()
        return {
            "ok": ok,
            "url": url,
            "http_status": status,
            "pages": pages,
            "expect_substring": expect_substring,
            "error": last_err if not ok and not status else None,
            "snippet": body[:240] if body else "",
        }
