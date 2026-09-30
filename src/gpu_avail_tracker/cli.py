"""Command-line entry point."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from .config import ConfigError, load_settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local GPU monitoring web dashboard")
    parser.add_argument("--config", "--env-file", dest="config_file", type=Path, default=Path("env.yml"), help="YAML configuration file (default: env.yml)")
    parser.add_argument("--port", type=int, default=3000, help="dashboard port (default: 3000)")
    parser.add_argument("--api-port", type=int, default=8000, help="local API port (default: 8000)")
    parser.add_argument("--api-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535 or not 1 <= args.api_port <= 65535 or args.port == args.api_port:
        parser.error("ports must be distinct integers from 1 to 65535")
    try:
        settings = load_settings(args.config_file)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    if args.api_only:
        from .web import serve

        try:
            serve(settings, args.api_port)
        except OSError as exc:
            print(f"Could not start API: {exc}", file=sys.stderr)
            return 1
        return 0

    root = Path(__file__).resolve().parents[2]
    runner = root / "scripts" / "run.mjs"
    if not runner.is_file() or not (root / "node_modules" / "next").is_dir():
        print("Dashboard dependencies are missing. From the project root run: npm install", file=sys.stderr)
        return 1
    env = os.environ.copy()
    env.update({
        "GPU_TRACKER_CONFIG_FILE": str(args.config_file.resolve()),
        "GPU_TRACKER_API_PORT": str(args.api_port),
        "GPU_TRACKER_PYTHON": sys.executable,
        "WEB_PORT": str(args.port),
    })
    try:
        return subprocess.call(["node", str(runner), "dev"], cwd=root, env=env)
    except FileNotFoundError:
        print("Node.js is required to run the dashboard.", file=sys.stderr)
        return 1
