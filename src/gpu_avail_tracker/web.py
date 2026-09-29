"""Local HTTP dashboard for GPU and checkpoint snapshots."""

from __future__ import annotations

import asyncio
import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import PurePosixPath
from urllib.parse import urlsplit

from .config import Settings, Target
from .probe import ProbeError, ProjectSnapshot, Snapshot, query_project_files, query_target


@dataclass
class HostState:
    snapshot: Snapshot | None = None
    error: str | None = None
    projects: ProjectSnapshot | None = None
    project_error: str | None = None


class Dashboard:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.states = {target.key: HostState() for target in settings.targets}
        self.last_refresh_at: datetime | None = None
        self.refreshing = False
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._wake.set()
        self._thread = threading.Thread(target=self._run, name="cluster-probes", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=35)

    def request_refresh(self) -> bool:
        with self._lock:
            if self.refreshing or self._wake.is_set():
                return False
            self._wake.set()
            return True

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(self.settings.refresh_seconds or None)
            if self._stop.is_set():
                break
            with self._lock:
                self._wake.clear()
                self.refreshing = True
            try:
                asyncio.run(self._refresh())
            finally:
                with self._lock:
                    self.last_refresh_at = datetime.now(timezone.utc)
                    self.refreshing = False

    async def _refresh(self) -> None:
        semaphore = asyncio.Semaphore(4)

        async def fetch(kind: str, target: Target) -> tuple[str, Target, Snapshot | ProjectSnapshot | None, str | None]:
            async with semaphore:
                try:
                    if kind == "gpu":
                        return kind, target, await query_target(target), None
                    return kind, target, await query_project_files(target), None
                except ProbeError as exc:
                    return kind, target, None, str(exc)

        tasks = [asyncio.create_task(fetch("gpu", target)) for target in self.settings.targets]
        tasks.extend(
            asyncio.create_task(fetch("files", target))
            for target in self.settings.targets if target.project_dirs
        )
        try:
            for task in asyncio.as_completed(tasks):
                kind, target, result, error = await task
                with self._lock:
                    state = self.states[target.key]
                    if kind == "gpu":
                        if isinstance(result, Snapshot):
                            state.snapshot, state.error = result, None
                        else:
                            state.error = error or "Query failed"
                    elif isinstance(result, ProjectSnapshot):
                        state.projects, state.project_error = result, None
                    else:
                        state.project_error = error or "Scan failed"
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    def payload(self) -> dict:
        with self._lock:
            hosts = []
            idle = busy = unknown = checkpoints = 0
            for target in self.settings.targets:
                state = self.states[target.key]
                snapshot = state.snapshot
                projects = state.projects
                if snapshot is None or state.error:
                    unknown += 1
                else:
                    idle += sum(not gpu.processes for gpu in snapshot.gpus)
                    busy += sum(bool(gpu.processes) for gpu in snapshot.gpus)
                if projects:
                    checkpoints += len(projects.files)
                hosts.append({
                    "cluster": target.cluster,
                    "name": target.name,
                    "host": target.host,
                    "gpu_error": state.error,
                    "project_error": state.project_error,
                    "gpu_checked_at": snapshot.checked_at.isoformat() if snapshot else None,
                    "project_checked_at": projects.checked_at.isoformat() if projects else None,
                    "project_dirs": target.project_dirs,
                    "gpus": [
                        {
                            "index": gpu.index,
                            "uuid": gpu.uuid,
                            "name": gpu.name,
                            "memory_total_mib": gpu.memory_total_mib,
                            "memory_used_mib": gpu.memory_used_mib,
                            "processes": [
                                {
                                    "pid": process.pid,
                                    "name": process.name,
                                    "user": process.user,
                                    "used_memory_mib": process.memory_mib,
                                }
                                for process in gpu.processes
                            ],
                        }
                        for gpu in snapshot.gpus
                    ] if snapshot else [],
                    "files": [
                        {
                            "name": PurePosixPath(file.path).name,
                            "path": file.path,
                            "size_bytes": file.size_bytes,
                            "modified_at": file.modified_at.isoformat(),
                        }
                        for file in projects.files
                    ] if projects else [],
                })
            return {
                "hosts": hosts,
                "summary": {
                    "idle": idle, "busy": busy, "unknown_hosts": unknown,
                    "checkpoints": checkpoints, "total_hosts": len(hosts),
                },
                "refreshing": self.refreshing,
                "refresh_seconds": self.settings.refresh_seconds,
                "last_refresh_at": self.last_refresh_at.isoformat() if self.last_refresh_at else None,
            }


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], dashboard: Dashboard) -> None:
        super().__init__(address, DashboardHandler)
        self.dashboard = dashboard


class DashboardHandler(BaseHTTPRequestHandler):
    server: DashboardServer

    def _trusted_origin(self, origin: str) -> bool:
        try:
            parsed = urlsplit(origin)
            return (
                parsed.scheme == "http"
                and parsed.hostname in ("localhost", "127.0.0.1")
                and parsed.port in (self.server.server_port, int(os.environ.get("WEB_PORT", "3000")))
            )
        except ValueError:
            return False

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; object-src 'none'; base-uri 'none'")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, data: dict) -> None:
        self._send(status, json.dumps(data).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/api/state":
            self._json(200, self.server.dashboard.payload())
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if urlsplit(self.path).path != "/api/refresh":
            self.send_error(404)
            return
        origin = self.headers.get("Origin")
        if origin and not self._trusted_origin(origin):
            self.send_error(403)
            return
        if self.headers.get("Content-Length", "0") != "0" or self.headers.get("Transfer-Encoding"):
            self.send_error(400)
            return
        accepted = self.server.dashboard.request_refresh()
        self._json(202, {"accepted": accepted})


def serve(settings: Settings, port: int = 8000) -> None:
    dashboard = Dashboard(settings)
    with DashboardServer(("127.0.0.1", port), dashboard) as server:
        dashboard.start()
        print(f"Cluster Manager API is running at http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        finally:
            dashboard.close()
