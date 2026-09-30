from __future__ import annotations

import asyncio
import json
import threading
import time
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from gpu_avail_tracker.config import Settings, Target
from gpu_avail_tracker.probe import Gpu, GpuProcess, ProbeError, Snapshot, StorageUsage
from gpu_avail_tracker.web import Dashboard, DashboardServer


class DashboardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.target = Target(
            "lab", "gpu01", "gpu.example", "alice", Path("/tmp/key"),
        )
        self.checked_at = datetime.now(timezone.utc)
        self.snapshot = Snapshot(self.target, (
            Gpu(0, "GPU-aaa", "NVIDIA A100", 81920, 2048, (
                GpuProcess(123, "python", "alice", 2000),
            )),
            Gpu(1, "GPU-bbb", "NVIDIA A100", 81920, 0, ()),
        ), self.checked_at)
        self.dashboard = Dashboard(Settings((self.target,), 0))

    def test_refresh_preserves_stale_gpu_snapshot(self) -> None:
        with patch("gpu_avail_tracker.web.query_target", return_value=self.snapshot) as query:
            asyncio.run(self.dashboard._refresh())
        query.assert_awaited_once_with(self.target)
        initial = self.dashboard.payload()
        self.assertEqual(initial["summary"], {
            "idle": 1, "busy": 1, "unknown_hosts": 0, "total_hosts": 1,
        })
        self.assertNotIn("files", initial["hosts"][0])
        self.assertNotIn("project_error", initial["hosts"][0])
        self.assertEqual(initial["hosts"][0]["gpus"][0]["processes"][0]["used_memory_mib"], 2000)

        with patch("gpu_avail_tracker.web.query_target", side_effect=ProbeError("SSH unavailable")):
            asyncio.run(self.dashboard._refresh())
        stale = self.dashboard.payload()
        self.assertEqual(stale["summary"]["unknown_hosts"], 1)
        self.assertEqual(stale["summary"]["idle"], 0)
        self.assertEqual(stale["summary"]["busy"], 0)
        self.assertEqual(stale["hosts"][0]["gpu_error"], "SSH unavailable")
        self.assertEqual(stale["hosts"][0]["gpus"], initial["hosts"][0]["gpus"])

    def test_http_api_returns_state_and_rejects_cross_origin_refresh(self) -> None:
        with DashboardServer(("127.0.0.1", 0), self.dashboard) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                with urlopen(base + "/api/state") as response:
                    state = json.load(response)
                    self.assertEqual(response.headers["Cache-Control"], "no-store")
                self.assertEqual(state["summary"]["total_hosts"], 1)
                self.assertNotIn("identity_file", state["hosts"][0])

                with self.assertRaises(HTTPError) as rejected:
                    urlopen(Request(base + "/api/refresh", method="POST", headers={
                        "Origin": "https://attacker.example",
                    }))
                self.assertEqual(rejected.exception.code, 403)
                with urlopen(Request(base + "/api/refresh", method="POST")) as response:
                    self.assertEqual(json.load(response), {"accepted": True})
                with urlopen(Request(base + "/api/refresh", method="POST")) as response:
                    self.assertEqual(json.load(response), {"accepted": False})
            finally:
                server.shutdown()
                thread.join()

    def test_new_metrics_are_serialized(self) -> None:
        snapshot = replace(self.snapshot, memory_total_kib=1000, memory_available_kib=400,
                           gpus=(replace(self.snapshot.gpus[0], utilization_percent=82,
                                         processes=(GpuProcess(123, "python", "alice", 2000, 3600),)),))
        with patch("gpu_avail_tracker.web.query_target", return_value=snapshot):
            asyncio.run(self.dashboard._refresh())
        host = self.dashboard.payload()["hosts"][0]
        self.assertEqual(host["memory_available_kib"], 400)
        self.assertEqual(host["gpus"][0]["utilization_percent"], 82)
        self.assertEqual(host["gpus"][0]["processes"][0]["elapsed_seconds"], 3600)

    def test_storage_failures_are_independent_and_folders_are_cached(self) -> None:
        target = replace(self.target, disk_paths=("/scratch",), folder_paths=("/scratch/runs",))
        dashboard = Dashboard(Settings((target,)))
        with patch("gpu_avail_tracker.web.query_target", return_value=self.snapshot), patch(
            "gpu_avail_tracker.web.query_storage", return_value=StorageUsage(100, 1000, 800),
        ) as storage:
            asyncio.run(dashboard._refresh())
            self.assertEqual(storage.await_count, 2)
            original = dashboard.payload()["hosts"][0]["storage"]
            storage.reset_mock()
            storage.side_effect = ProbeError("Permission denied")
            asyncio.run(dashboard._refresh())
            storage.assert_awaited_once_with(target, "disk", "/scratch")
            host = dashboard.payload()["hosts"][0]
            self.assertIsNone(host["gpu_error"])
            self.assertEqual(host["storage"][0]["error"], "Permission denied")
            self.assertEqual(host["storage"][0]["used_kib"], 100)
            self.assertEqual(host["storage"][0]["checked_at"], original[0]["checked_at"])
            self.assertEqual(host["storage"][1], original[1])
            dashboard.states[target.key].storage["folder", "/scratch/runs"].attempted_monotonic -= 301
            storage.reset_mock()
            asyncio.run(dashboard._refresh())
            self.assertEqual(storage.await_count, 2)
            self.assertEqual(dashboard.payload()["hosts"][0]["storage"][1]["error"], "Permission denied")

    def test_storage_can_succeed_when_gpu_query_fails(self) -> None:
        target = replace(self.target, disk_paths=("/", "/missing"))
        dashboard = Dashboard(Settings((target,)))

        async def query(_target: Target, _kind: str, path: str) -> StorageUsage:
            if path == "/missing":
                raise ProbeError("No such file or directory")
            return StorageUsage(0, 1000, 1000)

        with patch("gpu_avail_tracker.web.query_target", side_effect=ProbeError("GPU unavailable")), patch(
            "gpu_avail_tracker.web.query_storage", side_effect=query,
        ):
            asyncio.run(dashboard._refresh())
        host = dashboard.payload()["hosts"][0]
        self.assertEqual(host["gpu_error"], "GPU unavailable")
        self.assertEqual(host["storage"][0]["available_kib"], 1000)
        self.assertIsNone(host["storage"][1]["used_kib"])
        self.assertEqual(host["storage"][1]["error"], "No such file or directory")

    def test_launch_refresh_does_not_overlap_manual_requests(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        calls = 0

        async def query(_target: Target) -> Snapshot:
            nonlocal calls
            calls += 1
            entered.set()
            await asyncio.to_thread(release.wait, 5)
            return self.snapshot

        with patch("gpu_avail_tracker.web.query_target", side_effect=query):
            self.dashboard.start()
            try:
                self.assertTrue(entered.wait(2))
                self.assertTrue(self.dashboard.payload()["refreshing"])
                self.assertFalse(self.dashboard.request_refresh())
                self.assertEqual(calls, 1)
                release.set()
                deadline = time.monotonic() + 2
                while self.dashboard.payload()["refreshing"] and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertFalse(self.dashboard.payload()["refreshing"])
                self.assertTrue(self.dashboard.request_refresh())
                deadline = time.monotonic() + 2
                while calls < 2 and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertEqual(calls, 2)
            finally:
                release.set()
                self.dashboard.close()
