from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import subprocess

from gpu_avail_tracker.config import Target
from gpu_avail_tracker.probe import (
    ProbeError,
    REMOTE_SCRIPT,
    parse_snapshot,
    query_target,
    query_storage,
    storage_script,
)


def frame(*blocks: str) -> bytes:
    payload = b"GPU_AVAIL_V2\n"
    if len(blocks) == 3:
        blocks = (*blocks, "MemTotal:       131072 kB\nMemAvailable:   65536 kB")
    for block in blocks:
        data = block.encode("utf-8")
        payload += str(len(data)).encode("ascii") + b"\n" + data
    return payload


GPU_ROWS = "0, GPU-aaa, NVIDIA A100, 81920, 2048, 75\n1, GPU-bbb, NVIDIA A100, 81920, 0, N/A"
APP_ROWS = 'GPU-aaa, 123, "python, trainer.py", 2000\nGPU-aaa, 456, python, 48'
OWNER_ROWS = "123 alice 13320\n456 bob 0\n999 root 86400"


class ProbeParsingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.target = Target("lab", "gpu01", "gpu.example", "alice", Path("/tmp/key"))

    def test_processes_are_joined_by_gpu_uuid_and_pid(self) -> None:
        snapshot = parse_snapshot(frame(GPU_ROWS, APP_ROWS, OWNER_ROWS), self.target)
        self.assertEqual(len(snapshot.gpus), 2)
        self.assertEqual(snapshot.gpus[0].processes[0].name, "python, trainer.py")
        self.assertEqual(snapshot.gpus[0].processes[0].user, "alice")
        self.assertEqual(len(snapshot.gpus[1].processes), 0)
        self.assertEqual(snapshot.gpus[0].processes[0].elapsed_seconds, 13320)
        self.assertEqual(snapshot.gpus[0].processes[1].elapsed_seconds, 0)
        self.assertEqual(snapshot.gpus[0].utilization_percent, 75)
        self.assertIsNone(snapshot.gpus[1].utilization_percent)
        self.assertEqual(snapshot.memory_available_kib, 65536)
        self.assertEqual(snapshot.memory_total_kib, 131072)

    def test_missing_owner_does_not_make_busy_gpu_idle(self) -> None:
        snapshot = parse_snapshot(frame(GPU_ROWS, APP_ROWS, ""), self.target)
        self.assertEqual(len(snapshot.gpus[0].processes), 2)
        self.assertIsNone(snapshot.gpus[0].processes[0].user)
        self.assertIsNone(snapshot.gpus[0].processes[0].elapsed_seconds)

    def test_partial_and_malformed_data_are_not_snapshots(self) -> None:
        with self.assertRaisesRegex(ProbeError, "ended early"):
            parse_snapshot(frame(GPU_ROWS, APP_ROWS, OWNER_ROWS)[:-3], self.target)
        with self.assertRaisesRegex(ProbeError, "wrong number of fields"):
            parse_snapshot(frame("0, GPU-aaa", "", ""), self.target)

    def test_unmatched_process_gpu_is_rejected(self) -> None:
        with self.assertRaisesRegex(ProbeError, "unknown GPU"):
            parse_snapshot(frame(GPU_ROWS, "GPU-other, 123, python, 200", OWNER_ROWS), self.target)

    def test_optional_memory_and_age_are_not_required(self) -> None:
        snapshot = parse_snapshot(frame(GPU_ROWS, APP_ROWS, "123 alice", ""), self.target)
        self.assertIsNone(snapshot.memory_available_kib)
        self.assertIsNone(snapshot.gpus[0].processes[0].elapsed_seconds)

    def test_invalid_utilization_and_age_are_rejected(self) -> None:
        with self.assertRaisesRegex(ProbeError, "utilization"):
            parse_snapshot(frame(GPU_ROWS.replace(", 75", ", 101"), APP_ROWS, OWNER_ROWS), self.target)
        with self.assertRaisesRegex(ProbeError, "process age"):
            parse_snapshot(frame(GPU_ROWS, APP_ROWS, "123 alice -1"), self.target)

    def test_storage_paths_remain_literal_shell_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "injected"
            path = f"/data/it's a folder; $(touch {marker}) `touch {marker}`"
            timeout = root / "timeout"
            timeout.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n")
            timeout.chmod(0o755)
            result = subprocess.run(["sh", "-s"], input=storage_script("disk", path),
                                    text=True, capture_output=True, check=True,
                                    env={**os.environ, "PATH": f"{root}:{os.environ['PATH']}"})
            self.assertEqual(result.stdout.splitlines()[-1], path)
            self.assertFalse(marker.exists())

    def test_folder_scan_uses_same_filesystem_and_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "a folder's name"
            folder.mkdir()
            timeout = root / "timeout"
            timeout.write_text('#!/bin/sh\nshift 3\nexec "$@"\n')
            timeout.chmod(0o755)
            result = subprocess.run(["sh", "-s"], input=storage_script("folder", str(folder)),
                                    text=True, capture_output=True, check=True,
                                    env={**os.environ, "PATH": f"{root}:{os.environ['PATH']}"})
            self.assertTrue(result.stdout.split()[0].isdigit())
            self.assertIn("timeout -k 1s 15s sh -c", storage_script("folder", str(folder)))
            missing = subprocess.run(["sh", "-s"], input=storage_script("folder", str(root / "missing")),
                                     text=True, capture_output=True,
                                     env={**os.environ, "PATH": f"{root}:{os.environ['PATH']}"})
            self.assertNotEqual(missing.returncode, 0)
            self.assertIn("not an accessible directory", missing.stderr)

    def test_remote_shell_script_and_framing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nvidia = root / "nvidia-smi"
            nvidia.write_text(
                "#!/bin/sh\n"
                "case \"$1\" in\n"
                "  --query-gpu=*) printf '%s\\n' '0, GPU-aaa, NVIDIA A100, 81920, 2048, 75' ;;\n"
                "  --query-compute-apps=*) printf '%s\\n' 'GPU-aaa, 123, python, 2000' ;;\n"
                "  *) exit 2 ;;\n"
                "esac\n",
                encoding="utf-8",
            )
            nvidia.chmod(0o755)
            ps = root / "ps"
            ps.write_text("#!/bin/sh\nprintf '%s\\n' '123 alice 13320'\n", encoding="utf-8")
            ps.chmod(0o755)
            result = subprocess.run(
                ["sh", "-s"],
                input=REMOTE_SCRIPT.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={**os.environ, "PATH": f"{root}:{os.environ['PATH']}"},
                check=True,
            )
            snapshot = parse_snapshot(result.stdout, self.target)
            self.assertEqual(snapshot.gpus[0].processes[0].user, "alice")



class FakeSshTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.key = self.root / "key"
        self.key.write_text("test", encoding="utf-8")
        self.target = Target("lab", "gpu01", "gpu.example", "alice", self.key, 2222)
        self.args_file = self.root / "args.json"
        self.script_file = self.root / "remote-script.txt"
        self.ssh = self.root / "fake-ssh"
        self.ssh.write_text(
            f"#!{sys.executable}\n"
            "import base64, json, os, sys, time\n"
            "open(os.environ['FAKE_ARGS_FILE'], 'w').write(json.dumps(sys.argv[1:]))\n"
            "open(os.environ['FAKE_SCRIPT_FILE'], 'w').write(sys.stdin.read())\n"
            "if os.environ.get('FAKE_SLEEP'): time.sleep(1)\n"
            "if os.environ.get('FAKE_FAIL'):\n"
            "    sys.stderr.write('Permission denied (publickey).\\n')\n"
            "    sys.exit(255)\n"
            "sys.stdout.buffer.write(base64.b64decode(os.environ['FAKE_PAYLOAD']))\n",
            encoding="utf-8",
        )
        self.ssh.chmod(0o755)
        self.base_env = {
            "FAKE_ARGS_FILE": str(self.args_file),
            "FAKE_SCRIPT_FILE": str(self.script_file),
            "FAKE_PAYLOAD": base64.b64encode(frame(GPU_ROWS, APP_ROWS, OWNER_ROWS)).decode("ascii"),
        }

    async def test_one_safe_ssh_session(self) -> None:
        with patch.dict(os.environ, self.base_env):
            snapshot = await query_target(self.target, ssh_binary=str(self.ssh))
        self.assertEqual(len(snapshot.gpus), 2)
        args = json.loads(self.args_file.read_text(encoding="utf-8"))
        self.assertIn("BatchMode=yes", args)
        self.assertIn("StrictHostKeyChecking=yes", args)
        self.assertIn(str(self.key), args)
        self.assertEqual(args[-4:], ["--", "alice@gpu.example", "sh", "-s"])
        script = self.script_file.read_text(encoding="utf-8")
        self.assertIn("nvidia-smi --query-gpu", script)
        self.assertIn("nvidia-smi --query-compute-apps", script)
        self.assertIn("ps -eo", script)
        self.assertNotIn("gpu.example", script)

    async def test_ssh_error_is_reported(self) -> None:
        with patch.dict(os.environ, {**self.base_env, "FAKE_FAIL": "1"}):
            with self.assertRaisesRegex(ProbeError, "Permission denied"):
                await query_target(self.target, ssh_binary=str(self.ssh))

    async def test_storage_readings_and_malformed_output(self) -> None:
        for kind, payload, expected in (
            ("disk", "Filesystem 1024-blocks Used Available Capacity Mounted on\n/dev/my disk 1000 200 750 22% /data path\n", (200, 1000, 750)),
            ("folder", "4096\t/data/my folder\n", (4096, None, None)),
        ):
            with self.subTest(kind=kind), patch.dict(os.environ, {
                **self.base_env, "FAKE_PAYLOAD": base64.b64encode(payload.encode()).decode(),
            }):
                usage = await query_storage(self.target, kind, "/data/my folder", ssh_binary=str(self.ssh))
                self.assertEqual((usage.used_kib, usage.total_kib, usage.available_kib), expected)
        for kind in ("disk", "folder"):
            with patch.dict(os.environ, {**self.base_env, "FAKE_PAYLOAD": ""}):
                with self.assertRaises(ProbeError):
                    await query_storage(self.target, kind, "/missing", ssh_binary=str(self.ssh))

    async def test_timeout_is_reported(self) -> None:
        with patch.dict(os.environ, {**self.base_env, "FAKE_SLEEP": "1"}):
            with self.assertRaisesRegex(ProbeError, "timed out"):
                await query_target(self.target, ssh_binary=str(self.ssh), timeout_seconds=0.05)
