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
    REMOTE_FILES_SCRIPT,
    REMOTE_SCRIPT,
    parse_project_files,
    parse_snapshot,
    query_project_files,
    query_target,
)


def frame(*blocks: str) -> bytes:
    payload = b"GPU_AVAIL_V1\n"
    for block in blocks:
        data = block.encode("utf-8")
        payload += str(len(data)).encode("ascii") + b"\n" + data
    return payload


GPU_ROWS = "0, GPU-aaa, NVIDIA A100, 81920, 2048\n1, GPU-bbb, NVIDIA A100, 81920, 0"
APP_ROWS = 'GPU-aaa, 123, "python, trainer.py", 2000\nGPU-aaa, 456, python, 48'
OWNER_ROWS = "123 alice\n456 bob\n999 root"


class ProbeParsingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.target = Target("lab", "gpu01", "gpu.example", "alice", Path("/tmp/key"))

    def test_processes_are_joined_by_gpu_uuid_and_pid(self) -> None:
        snapshot = parse_snapshot(frame(GPU_ROWS, APP_ROWS, OWNER_ROWS), self.target)
        self.assertEqual(len(snapshot.gpus), 2)
        self.assertEqual(snapshot.gpus[0].processes[0].name, "python, trainer.py")
        self.assertEqual(snapshot.gpus[0].processes[0].user, "alice")
        self.assertEqual(len(snapshot.gpus[1].processes), 0)

    def test_missing_owner_does_not_make_busy_gpu_idle(self) -> None:
        snapshot = parse_snapshot(frame(GPU_ROWS, APP_ROWS, ""), self.target)
        self.assertEqual(len(snapshot.gpus[0].processes), 2)
        self.assertIsNone(snapshot.gpus[0].processes[0].user)

    def test_partial_and_malformed_data_are_not_snapshots(self) -> None:
        with self.assertRaisesRegex(ProbeError, "ended early"):
            parse_snapshot(frame(GPU_ROWS, APP_ROWS, OWNER_ROWS)[:-3], self.target)
        with self.assertRaisesRegex(ProbeError, "wrong number of fields"):
            parse_snapshot(frame("0, GPU-aaa", "", ""), self.target)

    def test_unmatched_process_gpu_is_rejected(self) -> None:
        with self.assertRaisesRegex(ProbeError, "unknown GPU"):
            parse_snapshot(frame(GPU_ROWS, "GPU-other, 123, python, 200", OWNER_ROWS), self.target)

    def test_remote_shell_script_and_framing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nvidia = root / "nvidia-smi"
            nvidia.write_text(
                "#!/bin/sh\n"
                "case \"$1\" in\n"
                "  --query-gpu=*) printf '%s\\n' '0, GPU-aaa, NVIDIA A100, 81920, 2048' ;;\n"
                "  --query-compute-apps=*) printf '%s\\n' 'GPU-aaa, 123, python, 2000' ;;\n"
                "  *) exit 2 ;;\n"
                "esac\n",
                encoding="utf-8",
            )
            nvidia.chmod(0o755)
            ps = root / "ps"
            ps.write_text("#!/bin/sh\nprintf '%s\\n' '123 alice'\n", encoding="utf-8")
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

    def test_remote_checkpoint_scan_recurses_and_reports_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "model $(echo injected)"
            nested = root / "run 1"
            nested.mkdir(parents=True)
            (nested / "weights.SAFETENSORS").write_bytes(b"weights")
            (nested / "notes.txt").write_text("ignore", encoding="utf-8")
            target = Target("lab", "gpu01", "gpu.example", "alice", Path("/tmp/key"), project_dirs=(str(root),))
            result = subprocess.run(
                ["sh", "-s"],
                input=(REMOTE_FILES_SCRIPT + str(root) + "\nEND_PROJECT_DIRS\n").encode("utf-8"),
                capture_output=True,
                check=True,
            )
            snapshot = parse_project_files(result.stdout, target)
            self.assertEqual([file.path for file in snapshot.files], [str(nested / "weights.SAFETENSORS")])
            self.assertEqual(snapshot.files[0].size_bytes, 7)

    def test_invalid_checkpoint_response_is_rejected(self) -> None:
        target = Target("lab", "gpu", "gpu.example", "alice", Path("/tmp/key"), project_dirs=("/work",))
        for payload, error in (
            (b"CHECKPOINTS_V1\0/work/a.pt\0", "Incomplete"),
            (b"CHECKPOINTS_V1\0/elsewhere/a.pt\0" b"1\0" b"1.0\0", "outside"),
            (b"CHECKPOINTS_V1\0/work/a.pt\0" b"-1\0" b"1.0\0", "metadata"),
        ):
            with self.subTest(payload=payload), self.assertRaisesRegex(ProbeError, error):
                parse_project_files(payload, target)


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

    async def test_timeout_is_reported(self) -> None:
        with patch.dict(os.environ, {**self.base_env, "FAKE_SLEEP": "1"}):
            with self.assertRaisesRegex(ProbeError, "timed out"):
                await query_target(self.target, ssh_binary=str(self.ssh), timeout_seconds=0.05)

    async def test_project_paths_are_sent_as_data_to_ssh(self) -> None:
        target = Target(
            "lab", "gpu01", "gpu.example", "alice", self.key, 2222,
            ("/work/my project", "/work/$(touch /tmp/never-run)"),
        )
        payload = b"CHECKPOINTS_V1\0/work/my project/run/checkpoint.pt\0" b"123\0" b"1790000000.0\0"
        with patch.dict(os.environ, {**self.base_env, "FAKE_PAYLOAD": base64.b64encode(payload).decode()}):
            snapshot = await query_project_files(target, ssh_binary=str(self.ssh))
        self.assertEqual(snapshot.files[0].size_bytes, 123)
        script = self.script_file.read_text(encoding="utf-8")
        self.assertIn('find "$directory"', script)
        self.assertTrue(script.endswith("/work/my project\n/work/$(touch /tmp/never-run)\nEND_PROJECT_DIRS\n"))

    async def test_project_scan_error_is_reported(self) -> None:
        target = Target(
            "lab", "gpu01", "gpu.example", "alice", self.key, 2222, ("/missing",)
        )
        with patch.dict(os.environ, {**self.base_env, "FAKE_FAIL": "1"}):
            with self.assertRaisesRegex(ProbeError, "Permission denied"):
                await query_project_files(target, ssh_binary=str(self.ssh))
