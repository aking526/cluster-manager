"""Load and validate local YAML configuration and environment overrides."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml


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
    disk_paths: tuple[str, ...] = ()
    folder_paths: tuple[str, ...] = ()

    @property
    def key(self) -> tuple[str, str]:
        return self.cluster, self.name


@dataclass(frozen=True)
class Settings:
    targets: tuple[Target, ...]
    refresh_seconds: int = 0
    folder_refresh_seconds: int = 300


_USERNAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*[$]?$")
_ENV_FIELDS = {
    "GPU_TRACKER_SSH_USER": "ssh_user",
    "GPU_TRACKER_SSH_KEY": "ssh_key",
    "GPU_TRACKER_REFRESH_SECONDS": "refresh_seconds",
    "GPU_TRACKER_FOLDER_REFRESH_SECONDS": "folder_refresh_seconds",
    "GPU_TRACKER_TARGETS": "targets",
}


class _ConfigLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        seen = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ConfigError("YAML field names must be strings")
            if key in seen:
                raise ConfigError(f"Duplicate YAML field: {key}")
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def _read_yaml_file(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        values = yaml.load(path.read_text(encoding="utf-8"), Loader=_ConfigLoader)
    except (OSError, UnicodeError) as exc:
        raise ConfigError(f"Cannot read {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        location = f":{mark.line + 1}:{mark.column + 1}" if mark else ""
        raise ConfigError(f"Invalid YAML in {path}{location}") from exc
    if not isinstance(values, dict):
        raise ConfigError(f"{path} must contain a YAML mapping (see env.example.yml)")
    unknown = set(values) - set(_ENV_FIELDS.values())
    if unknown:
        raise ConfigError(f"Unknown configuration fields: {', '.join(sorted(unknown))}")
    return values


def _interval(value: object, label: str, minimum: int, allow_zero: bool = False) -> int:
    if isinstance(value, str) and re.fullmatch(r"[0-9]+", value):
        value = int(value)
    if type(value) is not int or (value < minimum and not (allow_zero and value == 0)):
        allowed = f"0 or at least {minimum}" if allow_zero else f"at least {minimum}"
        raise ConfigError(f"{label} must be an integer {allowed}")
    return value


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


def _paths(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ConfigError(f"{label} must be an array of absolute remote paths")
    paths = tuple(_text(path, label) for path in value)
    if any(not path.startswith("/") for path in paths):
        raise ConfigError(f"{label} must contain absolute remote paths (no ~ expansion)")
    if len(set(paths)) != len(paths):
        raise ConfigError(f"{label} contains duplicate paths")
    return paths


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
    config_file: Path = Path("env.yml"), environ: Mapping[str, str] | None = None
) -> Settings:
    """Read YAML values, with explicitly set process environment taking precedence."""
    config_file = config_file.expanduser().resolve()
    values = _read_yaml_file(config_file)
    environment = os.environ if environ is None else environ
    for variable, field in _ENV_FIELDS.items():
        if variable in environment:
            value = environment[variable]
            if field == "targets":
                try:
                    value = json.loads(value)
                except json.JSONDecodeError as exc:
                    raise ConfigError(f"{variable} must be JSON: {exc.msg}") from exc
            values[field] = value
    if not config_file.exists() and "targets" not in values:
        raise ConfigError(f"No configuration found. Copy env.example.yml to {config_file} and edit it")

    user = _text(values.get("ssh_user"), "ssh_user")
    if not _USERNAME.fullmatch(user):
        raise ConfigError("ssh_user is not a valid SSH username")
    identity = _identity(values.get("ssh_key"), "ssh_key", config_file.parent)

    target_data = values.get("targets")
    if not isinstance(target_data, list) or not target_data:
        raise ConfigError("targets must be a nonempty YAML list")

    targets: list[Target] = []
    seen: set[tuple[str, str]] = set()
    for index, item in enumerate(target_data, 1):
        label = f"targets[{index}]"
        if not isinstance(item, dict):
            raise ConfigError(f"{label} must be an object")
        unknown = set(item) - {"cluster", "name", "host", "user", "identity_file", "port", "disk_paths", "folder_paths"}
        if unknown:
            raise ConfigError(f"{label} has unknown fields: {', '.join(sorted(unknown))}")
        cluster = _text(item.get("cluster"), f"{label}.cluster")
        name = _text(item.get("name"), f"{label}.name")
        host = _ssh_host(item.get("host"), f"{label}.host")
        target_user = _text(item.get("user", user), f"{label}.user")
        if not _USERNAME.fullmatch(target_user):
            raise ConfigError(f"{label}.user is not a valid SSH username")
        target_identity = _identity(
            item.get("identity_file", str(identity)), f"{label}.identity_file", config_file.parent
        )
        port = item.get("port", 22)
        if type(port) is not int or not 1 <= port <= 65535:
            raise ConfigError(f"{label}.port must be an integer from 1 to 65535")
        key = (cluster, name)
        if key in seen:
            raise ConfigError(f"Duplicate target {cluster}/{name}")
        seen.add(key)
        targets.append(
            Target(cluster, name, host, target_user, target_identity, port,
                   _paths(item.get("disk_paths", []), f"{label}.disk_paths"),
                   _paths(item.get("folder_paths", []), f"{label}.folder_paths"))
        )

    refresh_seconds = _interval(values.get("refresh_seconds", 0), "refresh_seconds", 10, allow_zero=True)
    folder_refresh_seconds = _interval(values.get("folder_refresh_seconds", 300), "folder_refresh_seconds", 60)
    return Settings(tuple(targets), refresh_seconds, folder_refresh_seconds)
