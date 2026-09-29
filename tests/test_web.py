from __future__ import annotations

import asyncio
import json
import threading
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from gpu_avail_tracker.config import Settings, Target
from gpu_avail_tracker.probe import Checkpoint, Gpu, GpuProcess, ProbeError, ProjectSnapshot, Snapshot
from gpu_avail_tracker.web import Dashboard, DashboardServer


class DashboardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.target = Target(
            "lab", "gpu01", "gpu.example", "alice", Path("/tmp/key"),
            project_dirs=("/work/models",),
        )
        self.checked_at = datetime.now(timezone.utc)
        self.snapshot = Snapshot(self.target, (
            Gpu(0, "GPU-aaa", "NVIDIA A100", 81920, 2048, (
                GpuProcess(123, "python", "alice", 2000),
            )),
            Gpu(1, "GPU-bbb", "NVIDIA A100", 81920, 0, ()),
        ), self.checked_at)
        self.projects = ProjectSnapshot((
            Checkpoint("/work/models/run/weights.safetensors", 1024, self.checked_at),
        ), self.checked_at)
        self.dashboard = Dashboard(Settings((self.target,), 0))

    def test_refresh_preserves_independent_stale_snapshots(self) -> None:
        with (
            patch("gpu_avail_tracker.web.query_target", return_value=self.snapshot),
            patch("gpu_avail_tracker.web.query_project_files", return_value=self.projects),
        ):
            asyncio.run(self.dashboard._refresh())
        initial = self.dashboard.payload()
        self.assertEqual(initial["summary"], {
            "idle": 1, "busy": 1, "unknown_hosts": 0, "checkpoints": 1, "total_hosts": 1,
        })
        self.assertEqual(initial["hosts"][0]["gpus"][0]["processes"][0]["used_memory_mib"], 2000)
        self.assertEqual(initial["hosts"][0]["files"][0]["name"], "weights.safetensors")

        with (
            patch("gpu_avail_tracker.web.query_target", side_effect=ProbeError("SSH unavailable")),
            patch("gpu_avail_tracker.web.query_project_files", side_effect=ProbeError("Permission denied")),
        ):
            asyncio.run(self.dashboard._refresh())
        stale = self.dashboard.payload()
        self.assertEqual(stale["summary"]["unknown_hosts"], 1)
        self.assertEqual(stale["summary"]["idle"], 0)
        self.assertEqual(stale["summary"]["busy"], 0)
        self.assertEqual(stale["hosts"][0]["gpu_error"], "SSH unavailable")
        self.assertEqual(stale["hosts"][0]["project_error"], "Permission denied")
        self.assertEqual(stale["hosts"][0]["gpus"], initial["hosts"][0]["gpus"])
        self.assertEqual(stale["hosts"][0]["files"], initial["hosts"][0]["files"])

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
