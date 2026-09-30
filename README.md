# Cluster Manager

A local, read-only web dashboard for NVIDIA GPU hosts reachable over SSH. The single-page Next.js interface shows GPU availability, active compute processes, and their owners. A Python service queries hosts and keeps the last successful snapshot when a query fails.

## Requirements

- Python 3.11+ and Node.js 20+
- OpenSSH on your computer
- Direct SSH access to Linux GPU hosts with `sh`, `nvidia-smi`, and `ps`
- Optional storage tracking requires GNU-compatible `timeout`, `df`, and `du` on the remote hosts
- A trusted host key in your local `known_hosts` file for each host

The service runs commands on the remote hosts without installing an agent or using sudo. Both the web and API servers listen on `127.0.0.1` by default; the dashboard is intended for local use.

## Setup

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .
npm install
cp env.example.yml env.yml
```

Edit `env.yml` with your real hosts, SSH username, and **private** key path. Your public key must already be authorized on each host. Local `env.yml` and `env.yaml` are gitignored; the committed `env.example.yml` contains only fake values. The configuration loader checks that the key path exists without reading its contents. OpenSSH uses the key when connecting.

For example:

```yaml
ssh_user: alice
ssh_key: ~/.ssh/id_ed25519
refresh_seconds: 0
folder_refresh_seconds: 300

targets:
  - cluster: lab-a
    name: gpu01
    host: gpu01.lab.example
  - cluster: lab-a
    name: gpu02
    host: gpu02.lab.example
  - cluster: lab-b
    name: gpu01
    host: gpu01.other.example
    port: 2222
```

Each target needs `cluster`, `name`, and `host`:

- `cluster` is a grouping label displayed in the dashboard, such as `lab-a`.
- `name` is a short label for one GPU machine, such as `gpu01`. The pair of `cluster` and `name` must be unique.
- `host` is the SSH destination for that machine: a DNS name, IP address, or SSH config alias.

Each target may override `user`, `identity_file`, and `port`. The default user and key are required. A relative key path is resolved from the YAML file's directory. Duplicate fields and unknown configuration fields are rejected. Quote values that YAML might interpret as booleans or numbers when you intend strings (for example, `name: "001"`).

Explicit process environment variables still override the corresponding file values: `GPU_TRACKER_SSH_USER`, `GPU_TRACKER_SSH_KEY`, `GPU_TRACKER_REFRESH_SECONDS`, `GPU_TRACKER_FOLDER_REFRESH_SECONDS`, and `GPU_TRACKER_TARGETS` (a JSON array that replaces the entire target list). The service no longer reads `.env` for configuration.

### Disk and folder tracking

Add `disk_paths` and/or `folder_paths` to individual entries under `targets`:

```yaml
targets:
  - cluster: lab-a
    name: gpu01
    host: gpu01.lab.example
    disk_paths:
      - /
      - /scratch
      - /mnt/datasets
    folder_paths:
      - /scratch/alice/runs
      - /scratch/alice/checkpoints
```

Both lists default to empty. Paths are absolute paths **on the remote host**; spaces and quotes are supported, but `~`, environment variables, and wildcards are not expanded. Restart the service after changing configuration.

- `disk_paths` reports total, used, and available capacity of the filesystem containing each path (`df -Pk`). Two paths on the same filesystem report the same capacity. Available space can be less than total minus used because of reserved blocks; it does not account for user quotas.
- `folder_paths` reports allocated disk space for each directory and its descendants (`du -skx`). It resolves the configured directory path, stays on that filesystem, and does not follow symlinks encountered within the directory. These are disk usage measurements, not apparent file sizes; overlapping folders should not be summed. Unreadable or missing directories produce an error rather than an incomplete total.

Disk checks run on each refresh. Folder scans run on the first refresh and then no more often than `folder_refresh_seconds` (default `300`, minimum `60`), including manual refreshes. This setting limits scan frequency; it does not enable automatic refreshing. With manual mode, a new folder scan happens on the next click after the interval. Each filesystem check has an 8-second remote timeout, and each folder scan has a 15-second remote timeout. A very large or slow directory may time out; choose narrower paths in that case.

Expand **Storage** on a host card to see measurements and their timestamps. Each path keeps its last successful reading on failure, marked stale. Storage checks share the four-SSH-query concurrency limit and cannot invalidate a successful GPU reading.

Before running the dashboard, verify each host's fingerprint through your lab's trusted source and connect once using SSH so OpenSSH can add it to `known_hosts`. The service requires an existing trusted host key and never accepts one automatically.

## Run

```sh
.venv/bin/cluster-manager
```

Open `http://127.0.0.1:3000` in your browser. The command starts both the Next.js frontend and local Python API. For development, `npm run dev` starts the same pair of servers. Use `--config path/to/env.yml` to load a different YAML file (`--env-file` remains an alias for the flag); `--port` changes the dashboard port and `--api-port` changes the API port. For a built frontend, run `npm run build` followed by `npm start`. Both commands require an installed Python package and a valid `env.yml`. For npm commands, set `GPU_TRACKER_CONFIG_FILE=/absolute/path/to/env.yml` to select another file.

The dashboard queries once at launch. Use the refresh icon to refresh GPU data. Each node has a compact card with GPU memory usage, process counts, and process owners. Available GPUs are blue and in-use GPUs are amber; selecting a GPU shows details. Set `refresh_seconds` to `10` or higher to add automatic refresh; `0` keeps manual mode. Refreshes never overlap, and at most four SSH queries run at once. Failed GPU queries retain the last successful snapshot and mark it stale.

**Available** means `nvidia-smi` returned no active compute processes for that GPU at the time of the query. It does not mean that a scheduler has released the GPU or that no graphics or other GPU work exists. A failed host is **Unknown**, and any previously displayed process list is marked stale. If a process exits between the NVIDIA query and `ps`, its owner appears as **Unknown** while its GPU remains **In use**.

The dashboard displays physical GPUs. NVIDIA MIG partitions and scheduler reservations are not tracked.

GPU rows also show utilization reported by `nvidia-smi` (`N/A` when unsupported). Expand a GPU to see each process's age at the last successful check, using `ps etimes`. This includes waiting time and may predate the experiment, for example with a persistent notebook kernel. If the process exits or elapsed time is unavailable, its age is unknown. Host cards show available and total system RAM from Linux `/proc/meminfo` (`MemAvailable` and `MemTotal`); available RAM includes memory the kernel estimates it can reclaim. RAM and process ages retain the GPU snapshot's timestamp and stale status. No history store or remote monitoring agent is required.

## Checks

```sh
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
npm run lint
npm run typecheck
npm run build
```
