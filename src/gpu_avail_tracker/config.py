"""Load and validate the intentionally small .env configuration format."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


class ConfigError(ValueError):
    """Configuration cannot be used safely."""


@dataclass(frozen=True)
class Target:
    cluster: str
    name: str
    host: str
    user: str
    identity_file: Path
    port: int = 22
    project_dirs: tuple[str, ...] = ()

    @property
    def key(self) -> tuple[str, str]:
        return self.cluster, self.name


@dataclass(frozen=True)
class Settings:
    targets: tuple[Target, ...]
    refresh_seconds: int = 0


_ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_USERNAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*[$]?$")


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ConfigError(f"Cannot read {path}: {exc}") from exc

    values: dict[str, str] = {}
    for line_number, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ConfigError(f"{path}:{line_number}: expected KEY=value")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not _ENV_KEY.fullmatch(key):
            raise ConfigError(f"{path}:{line_number}: invalid variable name")
        if key in values:
            raise ConfigError(f"{path}:{line_number}: duplicate {key}")
        if value.startswith(("'", '"')):
            quote = value[0]
            if len(value) < 2 or value[-1] != quote:
                raise ConfigError(f"{path}:{line_number}: unmatched quote")
            value = value[1:-1]
        values[key] = value
    return values


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{label} must be a nonempty string")
    text = value.strip()
    if any(ord(char) < 32 or ord(char) == 127 for char in text):
        raise ConfigError(f"{label} contains a control character")
    return text


def _ssh_host(value: object, label: str) -> str:
    host = _text(value, label)
    if host.startswith("-") or not re.fullmatch(r"[A-Za-z0-9._:%\[\]-]+", host):
        raise ConfigError(f"{label} must be a hostname, IP address, or SSH alias")
    return host


def _identity(value: object, label: str, base: Path) -> Path:
    name = _text(value, label)
    path = Path(name).expanduser()
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if not path.is_file():
        raise ConfigError(f"{label} does not exist: {path}")
    return path


def load_settings(
    env_file: Path = Path(".env"), environ: Mapping[str, str] | None = None
) -> Settings:
    """Read documented .env values, with process environment taking precedence."""
    env_file = env_file.expanduser().resolve()
    values = _read_env_file(env_file)
    values.update(os.environ if environ is None else environ)
    if not env_file.exists() and not values.get("GPU_TRACKER_TARGETS"):
        raise ConfigError(f"No configuration found. Copy .env.example to {env_file} and edit it")

    user = _text(values.get("GPU_TRACKER_SSH_USER"), "GPU_TRACKER_SSH_USER")
    if not _USERNAME.fullmatch(user):
        raise ConfigError("GPU_TRACKER_SSH_USER is not a valid SSH username")
    identity = _identity(
        values.get("GPU_TRACKER_SSH_KEY"), "GPU_TRACKER_SSH_KEY", env_file.parent
    )

    raw_targets = values.get("GPU_TRACKER_TARGETS")
    if not raw_targets:
        raise ConfigError("Set GPU_TRACKER_TARGETS in .env (see .env.example)")
    try:
        target_data = json.loads(raw_targets)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"GPU_TRACKER_TARGETS must be JSON: {exc.msg}") from exc
    if not isinstance(target_data, list) or not target_data:
        raise ConfigError("GPU_TRACKER_TARGETS must be a nonempty JSON array")

    try:
        project_data = json.loads(values.get("GPU_TRACKER_PROJECT_DIRS", "{}"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"GPU_TRACKER_PROJECT_DIRS must be JSON: {exc.msg}") from exc
    if not isinstance(project_data, dict):
        raise ConfigError("GPU_TRACKER_PROJECT_DIRS must be a JSON object")
    project_dirs: dict[str, tuple[str, ...]] = {}
    for cluster, directories in project_data.items():
        label = f"GPU_TRACKER_PROJECT_DIRS[{cluster!r}]"
        if not isinstance(cluster, str) or not cluster.strip():
            raise ConfigError("GPU_TRACKER_PROJECT_DIRS keys must be cluster names")
        if not isinstance(directories, list):
            raise ConfigError(f"{label} must be an array of absolute directories")
        paths: list[str] = []
        for directory in directories:
            path = _text(directory, label)
            if not path.startswith("/"):
                raise ConfigError(f"{label} must contain absolute POSIX paths")
            if path not in paths:
                paths.append(path)
        project_dirs[cluster] = tuple(paths)

    targets: list[Target] = []
    seen: set[tuple[str, str]] = set()
    for index, item in enumerate(target_data, 1):
        label = f"GPU_TRACKER_TARGETS[{index}]"
        if not isinstance(item, dict):
            raise ConfigError(f"{label} must be an object")
        unknown = set(item) - {"cluster", "name", "host", "user", "identity_file", "port"}
        if unknown:
            raise ConfigError(f"{label} has unknown fields: {', '.join(sorted(unknown))}")
        cluster = _text(item.get("cluster"), f"{label}.cluster")
        name = _text(item.get("name"), f"{label}.name")
        host = _ssh_host(item.get("host"), f"{label}.host")
        target_user = _text(item.get("user", user), f"{label}.user")
        if not _USERNAME.fullmatch(target_user):
            raise ConfigError(f"{label}.user is not a valid SSH username")
        target_identity = _identity(
            item.get("identity_file", str(identity)), f"{label}.identity_file", env_file.parent
        )
        port = item.get("port", 22)
        if type(port) is not int or not 1 <= port <= 65535:
            raise ConfigError(f"{label}.port must be an integer from 1 to 65535")
        key = (cluster, name)
        if key in seen:
            raise ConfigError(f"Duplicate target {cluster}/{name}")
        seen.add(key)
        targets.append(
            Target(cluster, name, host, target_user, target_identity, port, project_dirs.get(cluster, ()))
        )

    unknown_clusters = set(project_dirs) - {target.cluster for target in targets}
    if unknown_clusters:
        raise ConfigError(
            f"GPU_TRACKER_PROJECT_DIRS has unknown clusters: {', '.join(sorted(unknown_clusters))}"
        )

    refresh_value = values.get("GPU_TRACKER_REFRESH_SECONDS", "0")
    try:
        refresh_seconds = int(refresh_value)
    except (TypeError, ValueError) as exc:
        raise ConfigError("GPU_TRACKER_REFRESH_SECONDS must be 0 or at least 10") from exc
    if refresh_seconds != 0 and refresh_seconds < 10:
        raise ConfigError("GPU_TRACKER_REFRESH_SECONDS must be 0 or at least 10")
    return Settings(tuple(targets), refresh_seconds)
