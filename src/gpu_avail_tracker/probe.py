"""One read-only SSH probe and strict parsing of its result."""

from __future__ import annotations

import asyncio
import csv
import io
import re
import shlex
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
    elapsed_seconds: int | None = None


@dataclass(frozen=True)
class Gpu:
    index: int
    uuid: str
    name: str
    memory_total_mib: int | None
    memory_used_mib: int | None
    processes: tuple[GpuProcess, ...]
    utilization_percent: int | None = None


@dataclass(frozen=True)
class Snapshot:
    target: Target
    gpus: tuple[Gpu, ...]
    checked_at: datetime
    memory_total_kib: int | None = None
    memory_available_kib: int | None = None


# GPU probe is constant; storage paths below are shell-quoted literal arguments.
# Length-prefixed blocks keep process names from masquerading as protocol delimiters.
REMOTE_SCRIPT = """set -eu
export LC_ALL=C
gpus="$(nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits)"
apps="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits)"
owners="$(ps -eo pid=,user:64=,etimes= 2>/dev/null || ps -eo pid=,user:64=)"
memory="$(cat /proc/meminfo 2>/dev/null || true)"
printf 'GPU_AVAIL_V2\\n'
for block in "$gpus" "$apps" "$owners" "$memory"; do
    length="$(printf '%s' "$block" | wc -c)"
    printf '%s\\n' "$length"
    printf '%s' "$block"
done
"""

def _blocks(payload: bytes) -> tuple[str, ...]:
    if len(payload) > 2_000_000:
        raise ProbeError("SSH response is too large")
    prefix = b"GPU_AVAIL_V2\n"
    if not payload.startswith(prefix):
        raise ProbeError("SSH response has an invalid header")
    position = len(prefix)
    blocks: list[str] = []
    for _ in range(4):
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
    return tuple(blocks)


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
    gpu_csv, app_csv, owner_text, memory_text = _blocks(payload)
    owners: dict[int, tuple[str, int | None]] = {}
    for line in owner_text.splitlines():
        fields = line.split()
        if len(fields) not in (2, 3) or not fields[0].isdigit():
            raise ProbeError("Invalid process owner output")
        owners[int(fields[0])] = (fields[1], _integer(fields[2], "process age") if len(fields) == 3 else None)

    gpu_rows = _csv_rows(gpu_csv, 6, "GPU")
    if not gpu_rows:
        raise ProbeError("No GPUs were returned by nvidia-smi")
    gpu_info: dict[str, tuple[int, str, int | None, int | None, int | None]] = {}
    indices: set[int] = set()
    for row in gpu_rows:
        index = _integer(row[0], "GPU index")
        assert index is not None
        uuid = row[1].strip()
        name = row[2].strip()
        if not uuid.startswith("GPU-") or not name or uuid in gpu_info or index in indices:
            raise ProbeError("Invalid or duplicate GPU identity")
        indices.add(index)
        utilization = _integer(row[5], "GPU utilization", optional=True)
        if utilization is not None and utilization > 100:
            raise ProbeError("Invalid GPU utilization: exceeds 100")
        gpu_info[uuid] = (
            index,
            name,
            _integer(row[3], "GPU total memory", optional=True),
            _integer(row[4], "GPU used memory", optional=True),
            utilization,
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
        user, elapsed = owners.get(pid, (None, None))
        apps[uuid].append(GpuProcess(pid, name, user, _integer(row[3], "process memory", optional=True), elapsed))

    gpus = tuple(
        Gpu(index, uuid, name, total, used, tuple(apps[uuid]), utilization)
        for uuid, (index, name, total, used, utilization) in sorted(gpu_info.items(), key=lambda item: item[1][0])
    )
    memory = dict(re.findall(r"^(MemTotal|MemAvailable):\s+(\d+) kB$", memory_text, re.MULTILINE))
    return Snapshot(target, gpus, datetime.now(timezone.utc),
                    int(memory["MemTotal"]) if "MemTotal" in memory else None,
                    int(memory["MemAvailable"]) if "MemAvailable" in memory else None)


@dataclass(frozen=True)
class StorageUsage:
    used_kib: int
    total_kib: int | None = None
    available_kib: int | None = None


def storage_script(kind: str, path: str) -> str:
    """Bound remote work too, so a disconnected client does not leave du running."""
    quoted = shlex.quote(path)
    if kind == "disk":
        command = f"timeout -k 1s 8s df -Pk -- {quoted}"
    elif kind == "folder":
        check = ('test -d "$1" || { echo "Path is not an accessible directory" >&2; exit 1; }; '
                 'exec du -skx -- "$1"/.')
        command = f"timeout -k 1s 15s sh -c {shlex.quote(check)} sh {quoted}"
    else:
        raise ValueError(f"Unknown storage kind: {kind}")
    return f"export LC_ALL=C\n{command}\n"


async def query_storage(target: Target, kind: str, path: str, *, ssh_binary: str = "ssh") -> StorageUsage:
    payload = await _run_ssh(target, storage_script(kind, path).encode(), ssh_binary, 22)
    lines = payload.decode("utf-8", "replace").strip().splitlines()
    if kind == "folder":
        if len(lines) != 1:
            raise ProbeError("Invalid folder size output")
        used = _integer(lines[0].split(maxsplit=1)[0], "folder size")
        assert used is not None
        return StorageUsage(used)
    # Parse from the numeric columns, allowing whitespace in the filesystem name.
    if len(lines) != 2:
        raise ProbeError("Invalid filesystem output")
    match = re.fullmatch(r".+?\s+(\d+)\s+(\d+)\s+(\d+)\s+\d+%\s+.+", lines[1])
    if match is None:
        raise ProbeError("Invalid filesystem output")
    total, used, available = map(int, match.groups())
    return StorageUsage(used, total, available)


def _error_text(stderr: bytes) -> str:
    message = stderr.decode("utf-8", "replace").strip().splitlines()
    first = message[0] if message else "remote command failed"
    first = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", first)
    return first[:180]


def _ssh_command(target: Target, ssh_binary: str) -> list[str]:
    return [
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


async def _run_ssh(target: Target, script: bytes, ssh_binary: str, timeout_seconds: float) -> bytes:
    try:
        process = await asyncio.create_subprocess_exec(
            *_ssh_command(target, ssh_binary),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        raise ProbeError(f"Could not start SSH: {exc}") from exc
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(script), timeout=timeout_seconds
        )
    except (asyncio.TimeoutError, asyncio.CancelledError) as exc:
        process.kill()
        await process.communicate()
        if isinstance(exc, asyncio.CancelledError):
            raise
        raise ProbeError(f"SSH query timed out after {timeout_seconds:g}s") from exc
    if process.returncode != 0:
        if process.returncode in (124, 137):
            raise ProbeError("Remote storage check timed out")
        raise ProbeError(_error_text(stderr))
    return stdout


async def query_target(
    target: Target, *, ssh_binary: str = "ssh", timeout_seconds: float = 15
) -> Snapshot:
    """Query one Linux GPU host without allowing interactive SSH prompts."""
    return parse_snapshot(
        await _run_ssh(target, REMOTE_SCRIPT.encode("utf-8"), ssh_binary, timeout_seconds), target
    )
