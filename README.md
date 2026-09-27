# GPU Availability Tracker

A local, read-only terminal dashboard for NVIDIA GPU hosts that you can reach over SSH. It shows each physical GPU, active compute processes, the process owners, and the time of the last successful query.

## Requirements

- Python 3.11 or newer and a terminal
- OpenSSH on your computer
- Direct SSH access to Linux GPU hosts with `sh`, `nvidia-smi`, and `ps`
- A trusted host key in your local `known_hosts` file for each host

The tracker runs commands on the remote hosts without installing an agent or using sudo.

## Setup

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .
cp .env.example .env
```

Edit `.env` with your real hosts, SSH username, and **private** key path. Your public key must already be authorized on each host. The local `.env` is gitignored. The committed `.env.example` contains only fake values.

For example:

```dotenv
GPU_TRACKER_SSH_USER=alice
GPU_TRACKER_SSH_KEY=~/.ssh/id_ed25519
GPU_TRACKER_TARGETS='[{"cluster":"lab-a","name":"gpu01","host":"gpu01.lab.example"},{"cluster":"lab-a","name":"gpu02","host":"gpu02.lab.example"},{"cluster":"lab-b","name":"gpu01","host":"gpu01.other.example","port":2222}]'
GPU_TRACKER_REFRESH_SECONDS=0
```

Each target needs `cluster`, `name`, and `host`:

- `cluster` is a grouping label displayed in the TUI, such as `lab-a`.
- `name` is a short label for one GPU machine, such as `gpu01`. The pair of `cluster` and `name` must be unique.
- `host` is the SSH destination for that machine: a DNS name, IP address, or SSH config alias.

Each target may override `user`, `identity_file`, and `port`. The default user and key are required. A relative key path is resolved from the `.env` directory. Environment variables override file values.

Before running the TUI, verify each host's fingerprint through your lab's trusted source and connect once using SSH so OpenSSH can add it to `known_hosts`. The tracker requires an existing trusted host key and never accepts one automatically.

## Run

```sh
.venv/bin/gpu-avail
```

Use `--env-file path/to/.env` to load a different file. Each host has a heading above its GPU rows. Press `r` to refresh, arrow keys to select a host or GPU, and `q` to quit. The app queries once at launch. Set `GPU_TRACKER_REFRESH_SECONDS` to `10` or higher to add automatic refresh; `0` keeps manual mode. Refreshes never overlap, and at most four hosts are queried at once.

**Idle** means `nvidia-smi` returned no active compute processes for that GPU at the time of the query. It does not mean that a scheduler has released the GPU or that no graphics or other GPU work exists. A failed host is **Unknown**, and any previously displayed process list is marked stale. If a process exits between the NVIDIA query and `ps`, its owner appears as **Unknown** while its GPU remains **In use**.

V1 displays physical GPUs. NVIDIA MIG partitions and scheduler reservations are not tracked.

## Tests

The tests use a fake SSH executable and Textual's headless TUI runner:

```sh
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```
