"""Local HTTP dashboard for GPU snapshots."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from .config import Settings, Target
from .probe import ProbeError, Snapshot, StorageUsage, query_storage, query_target


@dataclass
class StorageState:
    usage: StorageUsage | None = None
    checked_at: datetime | None = None
    attempted_at: datetime | None = None
    attempted_monotonic: float | None = None
    error: str | None = None


@dataclass
class HostState:
    snapshot: Snapshot | None = None
    error: str | None = None
    storage: dict[tuple[str, str], StorageState] = field(default_factory=dict)


class Dashboard:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.states = {target.key: HostState() for target in settings.targets}
        for target in settings.targets:
            self.states[target.key].storage = {
                (kind, path): StorageState()
                for kind, paths in (("disk", target.disk_paths), ("folder", target.folder_paths))
                for path in paths
            }
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

        async def fetch(target: Target) -> None:
            async with semaphore:
                try:
                    snapshot = await query_target(target)
                except ProbeError as exc:
                    with self._lock:
                        self.states[target.key].error = str(exc)
                else:
                    with self._lock:
                        state = self.states[target.key]
                        state.snapshot, state.error = snapshot, None

        async def fetch_storage(target: Target, kind: str, path: str) -> None:
            async with semaphore:
                try:
                    usage = await query_storage(target, kind, path)
                    error = None
                except ProbeError as exc:
                    usage, error = None, str(exc)
                with self._lock:
                    state = self.states[target.key].storage[kind, path]
                    state.attempted_at = datetime.now(timezone.utc)
                    state.attempted_monotonic = time.monotonic()
                    state.error = error
                    if usage is not None:
                        state.usage, state.checked_at = usage, state.attempted_at

        tasks = [asyncio.create_task(fetch(target)) for target in self.settings.targets]
        for target in self.settings.targets:
            for (kind, path), state in self.states[target.key].storage.items():
                if (kind == "folder" and state.attempted_monotonic is not None
                        and time.monotonic() - state.attempted_monotonic < self.settings.folder_refresh_seconds):
                    continue
                tasks.append(asyncio.create_task(fetch_storage(target, kind, path)))
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    def payload(self) -> dict:
        with self._lock:
            hosts = []
            idle = busy = unknown = 0
            for target in self.settings.targets:
                state = self.states[target.key]
                snapshot = state.snapshot
                if snapshot is None or state.error:
                    unknown += 1
                else:
                    idle += sum(not gpu.processes for gpu in snapshot.gpus)
                    busy += sum(bool(gpu.processes) for gpu in snapshot.gpus)
                hosts.append({
                    "cluster": target.cluster,
                    "name": target.name,
                    "host": target.host,
                    "gpu_error": state.error,
                    "gpu_checked_at": snapshot.checked_at.isoformat() if snapshot else None,
                    "memory_total_kib": snapshot.memory_total_kib if snapshot else None,
                    "memory_available_kib": snapshot.memory_available_kib if snapshot else None,
                    "storage": [
                        {
                            "kind": kind, "path": path, "error": reading.error,
                            "checked_at": reading.checked_at.isoformat() if reading.checked_at else None,
                            "attempted_at": reading.attempted_at.isoformat() if reading.attempted_at else None,
                            "used_kib": reading.usage.used_kib if reading.usage else None,
                            "total_kib": reading.usage.total_kib if reading.usage else None,
                            "available_kib": reading.usage.available_kib if reading.usage else None,
                        }
                        for (kind, path), reading in state.storage.items()
                    ],
                    "gpus": [
                        {
                            "index": gpu.index,
                            "uuid": gpu.uuid,
                            "name": gpu.name,
                            "memory_total_mib": gpu.memory_total_mib,
                            "memory_used_mib": gpu.memory_used_mib,
                            "utilization_percent": gpu.utilization_percent,
                            "processes": [
                                {
                                    "pid": process.pid,
                                    "name": process.name,
                                    "user": process.user,
                                    "used_memory_mib": process.memory_mib,
                                    "elapsed_seconds": process.elapsed_seconds,
                                }
                                for process in gpu.processes
                            ],
                        }
                        for gpu in snapshot.gpus
                    ] if snapshot else [],
                })
            return {
                "hosts": hosts,
                "summary": {
                    "idle": idle, "busy": busy, "unknown_hosts": unknown,
                    "total_hosts": len(hosts),
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
