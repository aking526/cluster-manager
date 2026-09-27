"""One read-only SSH probe and strict parsing of its result."""

from __future__ import annotations

import asyncio
import csv
import io
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from .config import Target


class ProbeError(RuntimeError):
    """A host could not provide a complete, trustworthy snapshot."""


@dataclass(frozen=True)
class GpuProcess:
    pid: int
    name: str
    user: str | None
    memory_mib: int | None


@dataclass(frozen=True)
class Gpu:
    index: int
    uuid: str
    name: str
    memory_total_mib: int | None
    memory_used_mib: int | None
    processes: tuple[GpuProcess, ...]


@dataclass(frozen=True)
class Snapshot:
    target: Target
    gpus: tuple[Gpu, ...]
    checked_at: datetime


# The script is constant: configuration values are SSH argv entries, never shell code.
# Length-prefixed blocks keep process names from masquerading as protocol delimiters.
REMOTE_SCRIPT = """set -eu
export LC_ALL=C
gpus="$(nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used --format=csv,noheader,nounits)"
apps="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits)"
owners="$(ps -eo pid=,user:64=)"
printf 'GPU_AVAIL_V1\\n'
for block in "$gpus" "$apps" "$owners"; do
    length="$(printf '%s' "$block" | wc -c)"
    printf '%s\\n' "$length"
    printf '%s' "$block"
done
"""


def _blocks(payload: bytes) -> tuple[str, str, str]:
    if len(payload) > 2_000_000:
        raise ProbeError("SSH response is too large")
    prefix = b"GPU_AVAIL_V1\n"
    if not payload.startswith(prefix):
        raise ProbeError("SSH response has an invalid header")
    position = len(prefix)
    blocks: list[str] = []
    for _ in range(3):
        end = payload.find(b"\n", position)
        if end < 0 or end - position > 12:
            raise ProbeError("SSH response has an invalid block length")
        try:
            size = int(payload[position:end].strip())
        except ValueError as exc:
            raise ProbeError("SSH response has an invalid block length") from exc
        if size < 0 or size > 2_000_000:
            raise ProbeError("SSH response has an invalid block length")
        position = end + 1
        if position + size > len(payload):
            raise ProbeError("SSH response ended early")
        blocks.append(payload[position : position + size].decode("utf-8", "replace"))
        position += size
    if position != len(payload):
        raise ProbeError("SSH response has extra data")
    return blocks[0], blocks[1], blocks[2]


def _csv_rows(data: str, expected_fields: int, label: str) -> list[list[str]]:
    try:
        rows = list(csv.reader(io.StringIO(data, newline=""), skipinitialspace=True, strict=True))
    except csv.Error as exc:
        raise ProbeError(f"Invalid {label} CSV: {exc}") from exc
    if any(len(row) != expected_fields for row in rows):
        raise ProbeError(f"Invalid {label} CSV: wrong number of fields")
    return rows


def _integer(value: str, label: str, *, optional: bool = False) -> int | None:
    if optional and value.strip().upper() in {"N/A", "[NOT SUPPORTED]"}:
        return None
    try:
        number = int(value.strip())
    except ValueError as exc:
        raise ProbeError(f"Invalid {label}: {value!r}") from exc
    if number < 0:
        raise ProbeError(f"Invalid {label}: {value!r}")
    return number


def parse_snapshot(payload: bytes, target: Target) -> Snapshot:
    """Only a complete, valid response becomes an availability snapshot."""
    gpu_csv, app_csv, owner_text = _blocks(payload)
    owners: dict[int, str] = {}
    for line in owner_text.splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or not fields[0].isdigit():
            raise ProbeError("Invalid process owner output")
        owners[int(fields[0])] = fields[1]

    gpu_rows = _csv_rows(gpu_csv, 5, "GPU")
    if not gpu_rows:
        raise ProbeError("No GPUs were returned by nvidia-smi")
    gpu_info: dict[str, tuple[int, str, int | None, int | None]] = {}
    indices: set[int] = set()
    for row in gpu_rows:
        index = _integer(row[0], "GPU index")
        assert index is not None
        uuid = row[1].strip()
        name = row[2].strip()
        if not uuid.startswith("GPU-") or not name or uuid in gpu_info or index in indices:
            raise ProbeError("Invalid or duplicate GPU identity")
        indices.add(index)
        gpu_info[uuid] = (
            index,
            name,
            _integer(row[3], "GPU total memory", optional=True),
            _integer(row[4], "GPU used memory", optional=True),
        )

    apps: dict[str, list[GpuProcess]] = {uuid: [] for uuid in gpu_info}
    for row in _csv_rows(app_csv, 4, "compute process"):
        uuid = row[0].strip()
        if uuid not in apps:
            raise ProbeError("A compute process references an unknown GPU")
        pid = _integer(row[1], "process PID")
        assert pid is not None
        if pid == 0:
            raise ProbeError("Invalid process PID: 0")
        name = row[2].strip()
        if not name:
            raise ProbeError("Compute process has no name")
        apps[uuid].append(
            GpuProcess(pid, name, owners.get(pid), _integer(row[3], "process memory", optional=True))
        )

    gpus = tuple(
        Gpu(index, uuid, name, total, used, tuple(apps[uuid]))
        for uuid, (index, name, total, used) in sorted(gpu_info.items(), key=lambda item: item[1][0])
    )
    return Snapshot(target, gpus, datetime.now(timezone.utc))


def _error_text(stderr: bytes) -> str:
    message = stderr.decode("utf-8", "replace").strip().splitlines()
    first = message[0] if message else "remote command failed"
    first = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", first)
    return first[:180]


async def query_target(
    target: Target, *, ssh_binary: str = "ssh", timeout_seconds: float = 15
) -> Snapshot:
    """Query one Linux GPU host without allowing interactive SSH prompts."""
    command = [
        ssh_binary,
        "-T",
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=yes",
        "-o", "IdentitiesOnly=yes",
        "-o", "NumberOfPasswordPrompts=0",
        "-o", "ConnectionAttempts=1",
        "-o", "ConnectTimeout=5",
        "-i", str(target.identity_file),
        "-p", str(target.port),
        "--", f"{target.user}@{target.host}", "sh", "-s",
    ]
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        raise ProbeError(f"Could not start SSH: {exc}") from exc
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(REMOTE_SCRIPT.encode("utf-8")), timeout=timeout_seconds
        )
    except (asyncio.TimeoutError, asyncio.CancelledError) as exc:
        process.kill()
        await process.communicate()
        if isinstance(exc, asyncio.CancelledError):
            raise
        raise ProbeError(f"SSH query timed out after {timeout_seconds:g}s") from exc
    if process.returncode != 0:
        raise ProbeError(_error_text(stderr))
    return parse_snapshot(stdout, target)
