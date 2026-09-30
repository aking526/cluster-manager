from __future__ import annotations

import json
import yaml
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gpu_avail_tracker.config import ConfigError, load_settings


class ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.key = self.root / "id_ed25519"
        self.key.write_text("test key", encoding="utf-8")
        self.config_file = self.root / "env.yml"

    def _config(self, targets: list[dict], refresh: int = 0) -> None:
        self.config_file.write_text(yaml.safe_dump({
            "ssh_user": "alice", "ssh_key": str(self.key),
            "targets": targets, "refresh_seconds": refresh,
        }, sort_keys=False), encoding="utf-8")

    def test_multiple_hosts_with_override(self) -> None:
        self._config([
            {"cluster": "lab-a", "name": "gpu01", "host": "gpu01.example"},
            {"cluster": "lab-b", "name": "gpu01", "host": "gpu02.example", "port": 2222, "user": "bob"},
        ])
        settings = load_settings(self.config_file, environ={})
        self.assertEqual(len(settings.targets), 2)
        self.assertEqual(settings.targets[0].user, "alice")
        self.assertEqual(settings.targets[1].user, "bob")
        self.assertEqual(settings.targets[1].port, 2222)
        self.assertEqual(settings.refresh_seconds, 0)

    def test_environment_override_and_timer(self) -> None:
        self._config([{"cluster": "lab", "name": "gpu", "host": "gpu.example"}])
        settings = load_settings(self.config_file, environ={"GPU_TRACKER_REFRESH_SECONDS": "30"})
        self.assertEqual(settings.refresh_seconds, 30)

    def test_per_host_storage_paths_and_folder_interval(self) -> None:
        self._config([
            {"cluster": "lab", "name": "gpu1", "host": "gpu1.example", "disk_paths": ["/", "/scratch"], "folder_paths": ["/scratch/alice's runs"]},
            {"cluster": "lab", "name": "gpu2", "host": "gpu2.example"},
        ])
        settings = load_settings(self.config_file, environ={"GPU_TRACKER_FOLDER_REFRESH_SECONDS": "600"})
        self.assertEqual(settings.targets[0].disk_paths, ("/", "/scratch"))
        self.assertEqual(settings.targets[0].folder_paths, ("/scratch/alice's runs",))
        self.assertEqual(settings.targets[1].disk_paths, ())
        self.assertEqual(settings.folder_refresh_seconds, 600)

    def test_invalid_storage_configuration_is_rejected(self) -> None:
        for field in ("disk_paths", "folder_paths"):
            for paths in ("/data", ["~/data"], ["relative"], ["/data", "/data"], ["/data\nother"], [42]):
                with self.subTest(field=field, paths=paths):
                    self._config([{"cluster": "lab", "name": "gpu", "host": "gpu.example", field: paths}])
                    with self.assertRaises(ConfigError):
                        load_settings(self.config_file, environ={})
        self._config([{"cluster": "lab", "name": "gpu", "host": "gpu.example"}])
        for interval in ("0", "59", "bad"):
            with self.assertRaisesRegex(ConfigError, "at least 60"):
                load_settings(self.config_file, environ={"GPU_TRACKER_FOLDER_REFRESH_SECONDS": interval})

    def test_legacy_project_directory_setting_is_ignored(self) -> None:
        self._config([{"cluster": "lab", "name": "gpu", "host": "gpu.example"}])
        baseline = load_settings(self.config_file, environ={})
        settings = load_settings(
            self.config_file, environ={"GPU_TRACKER_PROJECT_DIRS": "obsolete configuration"},
        )
        self.assertEqual(settings, baseline)

    def test_duplicate_target_and_unsafe_host_are_rejected(self) -> None:
        target = {"cluster": "lab", "name": "gpu", "host": "gpu.example"}
        self._config([target, target])
        with self.assertRaisesRegex(ConfigError, "Duplicate target"):
            load_settings(self.config_file, environ={})
        self._config([{**target, "host": "-oProxyCommand=evil"}])
        with self.assertRaisesRegex(ConfigError, "hostname"):
            load_settings(self.config_file, environ={})
        self._config([{**target, "host": "gpu.example;evil"}])
        with self.assertRaisesRegex(ConfigError, "hostname"):
            load_settings(self.config_file, environ={})

    def test_refresh_interval_below_minimum_is_rejected(self) -> None:
        self._config([{"cluster": "lab", "name": "gpu", "host": "gpu.example"}], refresh=5)
        with self.assertRaisesRegex(ConfigError, "at least 10"):
            load_settings(self.config_file, environ={})

    def test_missing_key_is_rejected(self) -> None:
        self._config([{"cluster": "lab", "name": "gpu", "host": "gpu.example"}])
        self.key.unlink()
        with self.assertRaisesRegex(ConfigError, "does not exist"):
            load_settings(self.config_file, environ={})

    def test_missing_file_explains_first_setup(self) -> None:
        with self.assertRaisesRegex(ConfigError, "Copy env.example.yml"):
            load_settings(self.config_file, environ={})

    def test_yaml_comments_and_relative_key_path(self) -> None:
        self.config_file.write_text('''# A readable host list
ssh_user: alice
ssh_key: id_ed25519
targets:
  - cluster: lab
    name: "001"
    host: gpu.example
    disk_paths:
      - /scratch
    folder_paths:
      - "/scratch/alice's runs"
''')
        settings = load_settings(self.config_file, environ={})
        self.assertEqual(settings.targets[0].identity_file, self.key.resolve())
        self.assertEqual(settings.targets[0].name, "001")
        self.assertEqual(settings.targets[0].folder_paths, ("/scratch/alice's runs",))
        self.assertEqual(settings.folder_refresh_seconds, 300)

    def test_invalid_yaml_and_duplicate_fields_are_rejected(self) -> None:
        for content, error in (
            ("targets: [", "Invalid YAML"),
            ("[]", "YAML mapping"),
            ("", "YAML mapping"),
            ("ssh_user: alice\nssh_user: bob", "Duplicate YAML field"),
            ("targets:\n  - name: first\n    name: second", "Duplicate YAML field"),
            ("refresh_second: 30", "Unknown configuration fields"),
            ("true: value", "field names must be strings"),
            ("!!python/object/apply:builtins.str [unsafe]", "Invalid YAML"),
        ):
            with self.subTest(content=content):
                self.config_file.write_text(content)
                with self.assertRaisesRegex(ConfigError, error):
                    load_settings(self.config_file, environ={})

    def test_yaml_intervals_reject_boolean_float_and_null(self) -> None:
        for field in ("refresh_seconds", "folder_refresh_seconds"):
            for value in (True, False, 60.5, None):
                with self.subTest(field=field, value=value):
                    self._config([{"cluster": "lab", "name": "gpu", "host": "gpu.example"}])
                    data = yaml.safe_load(self.config_file.read_text())
                    data[field] = value
                    self.config_file.write_text(yaml.safe_dump(data))
                    with self.assertRaisesRegex(ConfigError, field):
                        load_settings(self.config_file, environ={})

    def test_target_environment_override(self) -> None:
        self._config([{"cluster": "lab", "name": "gpu", "host": "gpu.example"}])
        settings = load_settings(self.config_file, environ={
            "GPU_TRACKER_TARGETS": json.dumps([{"cluster": "override", "name": "gpu2", "host": "gpu2.example"}]),
        })
        self.assertEqual(settings.targets[0].cluster, "override")
        with self.assertRaisesRegex(ConfigError, "must be JSON"):
            load_settings(self.config_file, environ={"GPU_TRACKER_TARGETS": "not json"})

    def test_config_loader_does_not_read_key_contents(self) -> None:
        self._config([{"cluster": "lab", "name": "gpu", "host": "gpu.example"}])
        original_open = Path.open

        def guarded_open(path, *args, **kwargs):
            self.assertNotEqual(path.resolve(), self.key.resolve(), "Loader must not open the SSH key")
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", guarded_open):
            settings = load_settings(self.config_file, environ={})
        self.assertEqual(settings.targets[0].identity_file, self.key.resolve())
