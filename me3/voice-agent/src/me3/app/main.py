"""Application entry — session UI backend + Phase 8 fdbench launcher pointer."""

from __future__ import annotations

import argparse
import copy
import logging
import os
import sys

from me3 import __version__
from me3.config import load_config, load_secrets, resolve_path

logger = logging.getLogger("me3.app")

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _uvicorn_log_config() -> dict:
    """Uvicorn access/error logs with timestamps (default uvicorn format has none)."""
    from uvicorn.config import LOGGING_CONFIG

    cfg = copy.deepcopy(LOGGING_CONFIG)
    cfg["formatters"]["default"]["fmt"] = "%(asctime)s %(levelprefix)s %(message)s"
    cfg["formatters"]["default"]["datefmt"] = LOG_DATEFMT
    cfg["formatters"]["access"]["fmt"] = (
        '%(asctime)s %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'
    )
    cfg["formatters"]["access"]["datefmt"] = LOG_DATEFMT
    return cfg


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        datefmt=LOG_DATEFMT,
        force=True,
    )
    parser = argparse.ArgumentParser(description="ME3 voice-agent")
    parser.add_argument("--serve", action="store_true", help="Run session UI backend")
    parser.add_argument(
        "--benchmark-check",
        action="store_true",
        help="Print fd-benchmark launcher check command for Phase 8",
    )
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = load_config()
    secrets = load_secrets()
    fd_bench = resolve_path(cfg.paths.fd_benchmark_dir)
    fdbench_url = secrets.fdbench_url or os.environ.get("FDBENCH_URL", "")

    logger.info("me3-voice-agent %s", __version__)
    logger.info("pending flags: %s", cfg.pending.model_dump())
    logger.info("fd-benchmark dir: %s (exists=%s)", fd_bench, fd_bench.is_dir())

    if args.benchmark_check:
        from me3.config import ROOT

        entry = ROOT / "scripts" / "fdbench_assistant_entry.py"
        launch = fd_bench / "launch_assistant.py"
        py = sys.executable
        print("Phase 8 fd-benchmark entry")
        print(f"  project: {ROOT}")
        print(f"  entry:   {entry} (exists={entry.is_file()})")
        print(f"  launch:  {launch} (exists={launch.is_file()})")
        print(f"  FDBENCH_URL default: {fdbench_url or 'ws://127.0.0.1:8791'}")
        print("\nCheck:")
        print(
            f"  FDBENCH_PROJECT={ROOT} FDBENCH_ENTRY={entry} \\\n"
            f"    {py} {launch} --check"
        )
        return 0 if entry.is_file() and launch.is_file() else 1

    if not args.serve:
        logger.info("Pass --serve for UI, or --benchmark-check for Phase 8 launcher help.")
        return 0

    if cfg.ui.require_token and not secrets.me3_ui_token:
        logger.error("Set ME3_UI_TOKEN in .env before --serve")
        return 1

    import uvicorn

    from me3.app.server import create_app

    host = args.host or cfg.ui.host
    port = args.port or cfg.ui.port
    app = create_app()
    logger.info("serving session UI backend on http://%s:%s", host, port)
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        log_config=_uvicorn_log_config(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
