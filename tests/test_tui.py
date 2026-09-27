from __future__ import annotations

import importlib.util
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from gpu_avail_tracker.config import Settings, Target
from gpu_avail_tracker.probe import Gpu, GpuProcess, ProbeError, Snapshot


@unittest.skipUnless(importlib.util.find_spec("textual"), "Textual is not installed")
class TuiTests(unittest.IsolatedAsyncioTestCase):
    async def test_launch_manual_refresh_and_stale_error(self) -> None:
        from textual.widgets import DataTable

        from gpu_avail_tracker.tui import GPUTrackerApp

        target = Target("lab", "gpu01", "gpu.example", "alice", Path("/tmp/key"))
        snapshot = Snapshot(
            target,
            (
                Gpu(0, "GPU-aaa", "NVIDIA A100", 81920, 2048, (GpuProcess(123, "python", "alice", 2000),)),
                Gpu(1, "GPU-bbb", "NVIDIA A100", 81920, 0, ()),
            ),
            datetime.now(timezone.utc),
        )
        calls = 0

        async def fake_query(_target: Target) -> Snapshot:
            nonlocal calls
            calls += 1
            if calls == 1:
                return snapshot
            raise ProbeError("Permission denied (publickey).")

        with patch("gpu_avail_tracker.tui.query_target", side_effect=fake_query):
            app = GPUTrackerApp(Settings((target,), 0))
            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause()
                table = app.query_one("#gpus", DataTable)
                self.assertEqual(calls, 1)
                self.assertEqual(table.row_count, 3)
                self.assertEqual(
                    [column.label.plain for column in table.columns.values()],
                    ["GPU", "Model", "Status", "Memory (MiB)", "Procs", "Checked"],
                )
                self.assertEqual(table.get_row_at(0)[0].plain, "lab / gpu01")
                self.assertEqual(table.get_row_at(0)[1:], ["", "", "", "", ""])
                self.assertEqual(table.get_row_at(1)[2].plain, "In use")
                self.assertEqual(table.get_row_at(1)[3].plain, "2048/81920")
                self.assertEqual(table.get_row_at(2)[3].plain, "   0/81920")
                self.assertEqual(
                    table.get_row_at(1)[3].plain.index("/"),
                    table.get_row_at(2)[3].plain.index("/"),
                )
                self.assertEqual(table.get_row_at(2)[2].plain, "Idle")

                table.move_cursor(row=2)
                await pilot.pause()

                await pilot.press("r")
                await pilot.pause()
                self.assertEqual(calls, 2)
                self.assertEqual(table.row_count, 3)
                self.assertEqual(table.get_row_at(1)[2].plain, "Unknown")
                self.assertEqual(table.cursor_row, 2)
                self.assertIs(app.states[target.key].snapshot, snapshot)

    async def test_each_host_has_its_own_heading(self) -> None:
        from textual.widgets import DataTable

        from gpu_avail_tracker.tui import GPUTrackerApp

        targets = (
            Target("lab-a", "gpu01", "one.example", "alice", Path("/tmp/key")),
            Target("lab-b", "gpu02", "two.example", "alice", Path("/tmp/key")),
        )

        async def fake_query(target: Target) -> Snapshot:
            used = 40000 if target.name == "gpu01" else 0
            return Snapshot(
                target,
                (Gpu(0, f"GPU-{target.name}", "NVIDIA A100", 81920, used, ()),),
                datetime.now(timezone.utc),
            )

        with patch("gpu_avail_tracker.tui.query_target", side_effect=fake_query):
            app = GPUTrackerApp(Settings(targets, 0))
            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause()
                table = app.query_one("#gpus", DataTable)
                self.assertEqual(table.row_count, 4)
                self.assertEqual(table.get_row_at(0)[0].plain, "lab-a / gpu01")
                self.assertEqual(table.get_row_at(1)[0], "0")
                self.assertEqual(table.get_row_at(2)[0].plain, "lab-b / gpu02")
                self.assertEqual(table.get_row_at(3)[0], "0")
                self.assertEqual(table.get_row_at(1)[3].plain, "40000/81920")
                self.assertEqual(table.get_row_at(3)[3].plain, "    0/81920")
