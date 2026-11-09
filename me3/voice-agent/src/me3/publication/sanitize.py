"""Reject private / sensitive files before GitHub publication."""

from __future__ import annotations

import re
from pathlib import Path

# Only these extensions may be published to the public Pages site.
ALLOWED_SUFFIXES = {
    ".html",
    ".htm",
    ".css",
    ".js",
    ".mjs",
    ".json",
    ".md",
    ".txt",
    ".svg",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
    ".ico",
    ".woff",
    ".woff2",
    ".map",
}

BLOCKED_NAME_PATTERNS = (
    re.compile(r"^\.env", re.I),
    re.compile(r"^\.git", re.I),
    re.compile(r"credentials", re.I),
    re.compile(r"secret", re.I),
    re.compile(r"id_rsa", re.I),
    re.compile(r"\.pem$", re.I),
    re.compile(r"\.key$", re.I),
    re.compile(r"\.sqlite3?$", re.I),
    re.compile(r"\.db$", re.I),
    re.compile(r"^token", re.I),
)

# Content heuristics — fail closed if matched in text-like files.
BLOCKED_CONTENT = (
    re.compile(r"sk-ant-", re.I),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"tvly-[A-Za-z0-9_-]{10,}"),
    re.compile(r"OPENCODE_SERVER_PASSWORD\s*="),
    re.compile(r"ME3_UI_TOKEN\s*="),
    re.compile(r"ANTHROPIC_API_KEY\s*="),
    re.compile(r"TAVILY_API_KEY\s*="),
    re.compile(r"GITHUB_PAT\s*="),
    re.compile(r"-----BEGIN (RSA |OPENSSH )?PRIVATE KEY-----"),
)

MAX_FILE_BYTES = 1_500_000
TEXT_SUFFIXES = {".html", ".htm", ".css", ".js", ".mjs", ".json", ".md", ".txt", ".svg", ".map"}


def is_blocked_name(name: str) -> bool:
    base = Path(name).name
    return any(p.search(base) for p in BLOCKED_NAME_PATTERNS)


def collect_publishable_files(source_dir: Path) -> tuple[list[Path], list[str]]:
    """Return (files, rejection reasons). Rejects private names/extensions/content."""
    source_dir = source_dir.resolve()
    if not source_dir.is_dir():
        return [], [f"source_dir is not a directory: {source_dir}"]

    files: list[Path] = []
    rejects: list[str] = []
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(source_dir).as_posix()
        # Blocked names first — including hidden secrets like .env
        if is_blocked_name(path.name) or any(is_blocked_name(part) for part in path.relative_to(source_dir).parts):
            rejects.append(f"blocked name: {rel}")
            continue
        if any(part.startswith(".") for part in path.relative_to(source_dir).parts):
            rejects.append(f"skip hidden path: {rel}")
            continue
        if path.suffix.lower() not in ALLOWED_SUFFIXES:
            rejects.append(f"disallowed extension: {rel}")
            continue
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            rejects.append(f"too large ({size} B): {rel}")
            continue
        if path.suffix.lower() in TEXT_SUFFIXES:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                rejects.append(f"unreadable: {rel} ({exc})")
                continue
            hit = next((p.pattern for p in BLOCKED_CONTENT if p.search(text)), None)
            if hit:
                rejects.append(f"private content matched /{hit}/ in {rel}")
                continue
        files.append(path)

    # Hard fail if any blocked-name/content reject (not merely hidden/extension skips)
    hard = [r for r in rejects if r.startswith(("blocked name:", "private content", "too large"))]
    if hard:
        return [], rejects
    if not files:
        rejects.append("no publishable files found")
        return [], rejects
    return files, rejects


def validate_source(source_dir: Path) -> dict:
    files, rejects = collect_publishable_files(source_dir)
    hard = any(
        r.startswith(("blocked name:", "private content", "too large", "source_dir")) for r in rejects
    )
    return {
        "ok": bool(files) and not hard,
        "file_count": len(files),
        "files": [p.relative_to(source_dir.resolve()).as_posix() for p in files],
        "rejects": rejects,
        "source_dir": str(source_dir.resolve()),
    }
