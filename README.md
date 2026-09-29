# Cluster Manager

A local, read-only web dashboard for NVIDIA GPU hosts reachable over SSH. The single-page Next.js interface shows GPU availability, active compute processes and owners, and project checkpoint files. A Python service queries hosts and keeps the last successful snapshot when a query fails.

## Requirements

- Python 3.11+ and Node.js 20+
- OpenSSH on your computer
- Direct SSH access to Linux GPU hosts with `sh`, `nvidia-smi`, and `ps`
- A trusted host key in your local `known_hosts` file for each host

The service runs commands on the remote hosts without installing an agent or using sudo. Both the web and API servers listen on `127.0.0.1` by default; the dashboard is intended for local use.

## Setup

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .
npm install
cp .env.example .env
```

Edit `.env` with your real hosts, SSH username, and **private** key path. Your public key must already be authorized on each host. The local `.env` is gitignored. The committed `.env.example` contains only fake values.

For example:

```dotenv
GPU_TRACKER_SSH_USER=alice
GPU_TRACKER_SSH_KEY=~/.ssh/id_ed25519
GPU_TRACKER_TARGETS='[{"cluster":"lab-a","name":"gpu01","host":"gpu01.lab.example"},{"cluster":"lab-a","name":"gpu02","host":"gpu02.lab.example"},{"cluster":"lab-b","name":"gpu01","host":"gpu01.other.example","port":2222}]'
GPU_TRACKER_PROJECT_DIRS='{"lab-a":["/home/alice/projects","/scratch/models"],"lab-b":["/work/checkpoints"]}'
GPU_TRACKER_REFRESH_SECONDS=0
```

Each target needs `cluster`, `name`, and `host`:

- `cluster` is a grouping label displayed in the dashboard, such as `lab-a`.
- `name` is a short label for one GPU machine, such as `gpu01`. The pair of `cluster` and `name` must be unique.
- `host` is the SSH destination for that machine: a DNS name, IP address, or SSH config alias.

Each target may override `user`, `identity_file`, and `port`. The default user and key are required. A relative key path is resolved from the `.env` directory. Environment variables override file values.

`GPU_TRACKER_PROJECT_DIRS` is an optional JSON object mapping cluster labels to arrays of absolute directories on the remote Linux hosts. Every host in a cluster is scanned under those paths; omit the value or use `{}` to disable checkpoint discovery. Paths may contain spaces, but must exist and be readable by the configured SSH user on each host. The scan recurses into subdirectories, excludes symlink targets, and lists filenames ending in `.ckpt`, `.pt`, `.pth`, `.bin`, `.safetensors`, `.onnx`, `.h5`, `.hdf5`, `.gguf`, or `.weights` (case insensitive). Only file paths, byte sizes, and modification times are read; model contents are not transferred.

Before running the dashboard, verify each host's fingerprint through your lab's trusted source and connect once using SSH so OpenSSH can add it to `known_hosts`. The service requires an existing trusted host key and never accepts one automatically.

## Run

```sh
.venv/bin/cluster-manager
```

Open `http://127.0.0.1:3000` in your browser. The command starts both the Next.js frontend and local Python API. For development, `npm run dev` starts the same pair of servers. Use `--env-file path/to/.env` to load a different file; `--port` changes the dashboard port and `--api-port` changes the API port. For a built frontend, run `npm run build` followed by `npm start`. Both commands require an installed Python package and a valid `.env`.

The dashboard queries once at launch. Use the refresh icon to refresh GPU and checkpoint views. Each node has a compact card with GPU memory usage, process counts, and process owners. Available GPUs are blue and in-use GPUs are amber; selecting a GPU or checkpoint shows details. Set `GPU_TRACKER_REFRESH_SECONDS` to `10` or higher to add automatic refresh; `0` keeps manual mode. Refreshes never overlap, and at most four SSH queries run at once. Checkpoint scan errors are shown separately from GPU query errors; the last successful list remains visible and is marked stale.

**Available** means `nvidia-smi` returned no active compute processes for that GPU at the time of the query. It does not mean that a scheduler has released the GPU or that no graphics or other GPU work exists. A failed host is **Unknown**, and any previously displayed process list is marked stale. If a process exits between the NVIDIA query and `ps`, its owner appears as **Unknown** while its GPU remains **In use**.

The dashboard displays physical GPUs. NVIDIA MIG partitions and scheduler reservations are not tracked.

## Checks

```sh
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
npm run lint
npm run typecheck
npm run build
```
