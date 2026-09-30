from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from gpu_avail_tracker.cli import main
from gpu_avail_tracker.config import Settings


class CliConfigTests(unittest.TestCase):
    def test_default_yaml_and_custom_config_reach_api(self) -> None:
        settings = Settings(())
        for args, expected in (([], Path("env.yml")), (["--config", "custom.yml"], Path("custom.yml"))):
            with self.subTest(args=args), patch("gpu_avail_tracker.cli.load_settings", return_value=settings) as load, patch(
                "gpu_avail_tracker.web.serve",
            ) as serve:
                self.assertEqual(main(["--api-only", *args]), 0)
                load.assert_called_once_with(expected)
                serve.assert_called_once_with(settings, 8000)

    def test_custom_config_is_forwarded_to_npm_runner(self) -> None:
        with patch("gpu_avail_tracker.cli.load_settings", return_value=Settings(())), patch(
            "gpu_avail_tracker.cli.subprocess.call", return_value=0,
        ) as launch, patch.object(Path, "is_file", return_value=True), patch.object(Path, "is_dir", return_value=True):
            self.assertEqual(main(["--config", "custom.yml"]), 0)
        self.assertEqual(launch.call_args.kwargs["env"]["GPU_TRACKER_CONFIG_FILE"], str(Path("custom.yml").resolve()))
