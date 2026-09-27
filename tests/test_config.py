from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gpu_avail_tracker.config import ConfigError, load_settings


class ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.key = self.root / "id_ed25519"
        self.key.write_text("test key", encoding="utf-8")
        self.env_file = self.root / ".env"

    def _env(self, targets: list[dict], refresh: int = 0) -> None:
        self.env_file.write_text(
            f"GPU_TRACKER_SSH_USER=alice\n"
            f"GPU_TRACKER_SSH_KEY={self.key}\n"
            f"GPU_TRACKER_TARGETS='{json.dumps(targets)}'\n"
            f"GPU_TRACKER_REFRESH_SECONDS={refresh}\n",
            encoding="utf-8",
        )

    def test_multiple_hosts_with_override(self) -> None:
        self._env([
            {"cluster": "lab-a", "name": "gpu01", "host": "gpu01.example"},
            {"cluster": "lab-b", "name": "gpu01", "host": "gpu02.example", "port": 2222, "user": "bob"},
        ])
        settings = load_settings(self.env_file, environ={})
        self.assertEqual(len(settings.targets), 2)
        self.assertEqual(settings.targets[0].user, "alice")
        self.assertEqual(settings.targets[1].user, "bob")
        self.assertEqual(settings.targets[1].port, 2222)
        self.assertEqual(settings.refresh_seconds, 0)

    def test_environment_override_and_timer(self) -> None:
        self._env([{"cluster": "lab", "name": "gpu", "host": "gpu.example"}])
        settings = load_settings(self.env_file, environ={"GPU_TRACKER_REFRESH_SECONDS": "30"})
        self.assertEqual(settings.refresh_seconds, 30)

    def test_duplicate_target_and_unsafe_host_are_rejected(self) -> None:
        target = {"cluster": "lab", "name": "gpu", "host": "gpu.example"}
        self._env([target, target])
        with self.assertRaisesRegex(ConfigError, "Duplicate target"):
            load_settings(self.env_file, environ={})
        self._env([{**target, "host": "-oProxyCommand=evil"}])
        with self.assertRaisesRegex(ConfigError, "hostname"):
            load_settings(self.env_file, environ={})
        self._env([{**target, "host": "gpu.example;evil"}])
        with self.assertRaisesRegex(ConfigError, "hostname"):
            load_settings(self.env_file, environ={})

    def test_refresh_interval_below_minimum_is_rejected(self) -> None:
        self._env([{"cluster": "lab", "name": "gpu", "host": "gpu.example"}], refresh=5)
        with self.assertRaisesRegex(ConfigError, "at least 10"):
            load_settings(self.env_file, environ={})

    def test_missing_key_is_rejected(self) -> None:
        self._env([{"cluster": "lab", "name": "gpu", "host": "gpu.example"}])
        self.key.unlink()
        with self.assertRaisesRegex(ConfigError, "does not exist"):
            load_settings(self.env_file, environ={})

    def test_missing_file_explains_first_setup(self) -> None:
        with self.assertRaisesRegex(ConfigError, "Copy .env.example"):
            load_settings(self.env_file, environ={})
