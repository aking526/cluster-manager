"""Command-line entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import ConfigError, load_settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only GPU availability TUI")
    parser.add_argument("--env-file", type=Path, default=Path(".env"), help="configuration file (default: .env)")
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.env_file)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    from .tui import GPUTrackerApp

    GPUTrackerApp(settings).run()
    return 0
