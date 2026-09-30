"use client";

import { useCallback, useEffect, useState } from "react";

type Process = { pid: number; name: string; user: string | null; used_memory_mib: number | null; elapsed_seconds: number | null };
type Gpu = {
  index: number; uuid: string; name: string;
  memory_total_mib: number | null; memory_used_mib: number | null; processes: Process[];
  utilization_percent: number | null;
};
type Storage = {
  kind: "disk" | "folder"; path: string; error: string | null;
  checked_at: string | null; attempted_at: string | null;
  used_kib: number | null; total_kib: number | null; available_kib: number | null;
};
type Host = {
  cluster: string; name: string; host: string; gpu_error: string | null;
  gpu_checked_at: string | null;
  gpus: Gpu[];
  memory_total_kib: number | null; memory_available_kib: number | null;
  storage: Storage[];
};
type State = {
  hosts: Host[];
  refreshing: boolean; refresh_seconds: number; last_refresh_at: string | null;
};
type Status = "idle" | "busy" | "unknown" | "waiting";

function timestamp(value: string | null) {
  return value ? new Date(value).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }) : "Not checked yet";
}

function memory(value: number | null) {
  return value === null ? "N/A" : `${value.toLocaleString()} MiB`;
}

function size(kib: number | null) {
  if (kib === null) return "N/A";
  if (kib >= 1024 ** 3) return `${(kib / 1024 ** 3).toLocaleString(undefined, { maximumFractionDigits: 1 })} TiB`;
  if (kib >= 1024 ** 2) return `${(kib / 1024 ** 2).toLocaleString(undefined, { maximumFractionDigits: 1 })} GiB`;
  if (kib >= 1024) return `${(kib / 1024).toLocaleString(undefined, { maximumFractionDigits: 1 })} MiB`;
  return `${kib.toLocaleString()} KiB`;
}

function duration(seconds: number | null) {
  if (seconds === null) return "Age unknown";
  const days = Math.floor(seconds / 86400);
  const hours = Math.floor(seconds / 3600) % 24;
  const minutes = Math.floor(seconds / 60) % 60;
  if (days) return `${days}d ${hours}h`;
  if (hours) return `${hours}h ${minutes}m`;
  if (minutes) return `${minutes}m ${seconds % 60}s`;
  return `${seconds}s`;
}

function HostResources({ host }: { host: Host }) {
  return <>
    <div className="host-resources"><span>Host RAM available{host.gpu_error && host.gpu_checked_at ? " (stale)" : ""}</span>
      <strong>{size(host.memory_available_kib)} <span>/ {size(host.memory_total_kib)}</span></strong>
    </div>
    {host.storage.length > 0 && <details className="storage-panel">
      <summary>Storage <span>{host.storage.length} tracked {host.storage.length === 1 ? "path" : "paths"}</span></summary>
      {host.storage.map((entry) => <div className="storage-entry" key={`${entry.kind}:${entry.path}`}>
        <div className="storage-path"><span>{entry.kind === "disk" ? "Filesystem" : "Folder"}</span><code>{entry.path}</code></div>
        {entry.checked_at ? <>
          <div className="storage-usage">{entry.kind === "disk"
            ? <><strong>{size(entry.available_kib)} available</strong><span>{size(entry.used_kib)} used · {size(entry.total_kib)} total</span></>
            : <strong>{size(entry.used_kib)} on disk</strong>}</div>
          <div className="storage-note">{entry.error ? "Stale · Last success " : "Checked "}{timestamp(entry.checked_at)}</div>
        </> : !entry.error && <div className="storage-note">Waiting for measurement…</div>}
        {entry.error && <div className="storage-error" role="status">{entry.error} · Attempted {timestamp(entry.attempted_at)}</div>}
      </div>)}
    </details>}
  </>;
}

function gpuStatus(host: Host, gpu: Gpu): Status {
  return host.gpu_error ? "unknown" : gpu.processes.length ? "busy" : "idle";
}

function badge(status: Status) {
  return <span className={`status-badge ${status}`}>{{
    idle: "Available", busy: "In use", unknown: "Unknown", waiting: "Waiting",
  }[status]}</span>;
}

function GpuDetail({ gpu, host }: { gpu: Gpu; host: Host }) {
  return <div className="detail-panel">
    <div className="detail-kicker">GPU DETAILS {host.gpu_error ? "· STALE SNAPSHOT" : ""}</div>
    <div className="detail-grid">
      <div className="detail-cell"><small>GPU ID</small><span>{gpu.uuid}</span></div>
      <div className="detail-cell"><small>Total memory</small><span>{memory(gpu.memory_total_mib)}</span></div>
      <div className="detail-cell"><small>Last checked</small><span>{timestamp(host.gpu_checked_at)}</span></div>
      <div className="detail-cell"><small>Compute processes</small><span>{gpu.processes.length}</span></div>
      <div className="detail-cell"><small>GPU utilization</small><span>{gpu.utilization_percent === null ? "N/A" : `${gpu.utilization_percent}%`}</span></div>
    </div>
    {gpu.processes.length ? <table className="process-table"><thead><tr><th>PID</th><th>Process</th><th>Owner</th><th>GPU memory</th></tr></thead><tbody>
      {gpu.processes.map((process, i) => <tr key={`${process.pid}-${i}`}><td>{process.pid}</td><td>{process.name}<span className="process-age" title="Process age at last check; includes waiting time and may predate the experiment">{process.elapsed_seconds === null ? "Age unknown" : `Running ${duration(process.elapsed_seconds)}`}</span></td><td>{process.user || "Unknown"}</td><td>{memory(process.used_memory_mib)}</td></tr>)}
    </tbody></table> : <p className="detail-text">No active compute processes observed on this GPU.</p>}
  </div>;
}

function HostCard({ host, selected, onSelect }: {
  host: Host; selected: string | null; onSelect: (key: string) => void;
}) {
  const hostKey = `${host.cluster}/${host.name}`;
  const error = host.gpu_error;
  const checked = host.gpu_checked_at;
  const hostStatus: Status = error ? "unknown" : !checked ? "waiting" : host.gpus.some((gpu) => gpu.processes.length) ? "busy" : "idle";
  return <article className="host-card">
    <div className="host-heading">
      <div className="host-identity"><div className="host-title">
        <h2>{host.name}</h2><small>{host.cluster} · {host.host}</small>
      </div></div>
      {badge(hostStatus)}
    </div>
    {error && <div className="error-banner">GPU query failed: {error}{checked ? " · Showing last successful data" : ""}</div>}
    <HostResources host={host} />
    {host.gpus.map((gpu) => {
      const key = `${hostKey}/gpu/${gpu.uuid}`;
      const expanded = selected === key;
      const owners = [...new Set(gpu.processes.map((process) => process.user || "Unknown user"))];
      const percent = gpu.memory_total_mib ? Math.min(100, Math.max(0, (gpu.memory_used_mib || 0) / gpu.memory_total_mib * 100)) : 0;
      return <div key={key}><button className="gpu-row" type="button" aria-expanded={expanded} onClick={() => onSelect(key)}>
        <span className="gpu-model"><span className="gpu-name"><span className="gpu-index">GPU {gpu.index}</span>{gpu.name}</span></span>
        <span className="gpu-status">{badge(gpuStatus(host, gpu))}</span>
        <span className="gpu-memory"><span className="memory-label">{memory(gpu.memory_used_mib)} / {memory(gpu.memory_total_mib)}</span><span className="memory-track"><span className={`memory-fill ${percent > 75 ? "heavy" : ""}`} style={{ display: "block", width: `${percent}%` }} /></span></span>
        <span className="gpu-processes"><span className="gpu-utilization">{gpu.utilization_percent === null ? "Utilization N/A" : `${gpu.utilization_percent}% utilization`}{host.gpu_error ? " (stale)" : ""}</span><span className="process-count">{`${gpu.processes.length} ${gpu.processes.length === 1 ? "process" : "processes"}`}{host.gpu_error ? " (stale)" : ""}</span>{owners.length > 0 && <span className="process-owners"><span className="owner-label">{owners.length === 1 ? "User" : "Users"}</span> {owners.join(", ")}</span>}</span><span className="row-chevron" aria-hidden="true">›</span>
      </button>{expanded && <GpuDetail gpu={gpu} host={host} />}</div>;
    })}
    {!host.gpus.length && <div className="empty-host">{error ? "No successful GPU snapshot yet." : "Waiting for GPU data…"}</div>}
    <div className="host-footer"><span>{`${host.gpus.length} ${host.gpus.length === 1 ? "GPU" : "GPUs"}`}</span><span>Checked {timestamp(checked)}</span></div>
  </article>;
}

export default function Home() {
  const [data, setData] = useState<State | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const response = await fetch("/api/state", { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      setData(await response.json() as State);
      setError(null);
    } catch {
      setError("Cannot reach the local API. Check that the Python service is running.");
    }
  }, []);
  useEffect(() => {
    void load();
    const poll = window.setInterval(() => { if (!document.hidden) void load(); }, 3000);
    return () => window.clearInterval(poll);
  }, [load]);

  async function refresh() {
    try {
      const response = await fetch("/api/refresh", { method: "POST" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      setData((previous) => previous ? { ...previous, refreshing: true } : previous);
      void load();
    } catch {
      setError("Refresh request failed. Check that the Python service is running.");
    }
  }

  const cards = (data?.hosts || []).map((host) =>
    <HostCard key={`${host.cluster}/${host.name}`} host={host} selected={selected} onSelect={(key) => setSelected(selected === key ? null : key)} />
  );
  return <main className="content" aria-label="Cluster manager">
    <div className="view-controls">
      <button className="refresh-button" onClick={() => void refresh()} disabled={data?.refreshing} type="button" aria-label={data?.refreshing ? "Refreshing data" : "Refresh data"} title={data?.refreshing ? "Refreshing data" : "Refresh data"}>
        <svg className="refresh-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M20 7v5h-5M4 17v-5h5" /><path d="M6.1 7a7 7 0 0 1 11.6-1L20 9M4 15l2.3 3A7 7 0 0 0 17.9 17" /></svg>
      </button>
    </div>
    {error && <div className="api-error" role="alert">{error}</div>}
    <section className="inventory-list" aria-label="Nodes" aria-live="polite">{cards.length ? cards : <div className="empty-state"><strong>{data ? "No nodes configured" : "Waiting for cluster data"}</strong>{data ? "Add nodes to your configuration to see them here." : "Your nodes will appear after the first query."}</div>}</section>
  </main>;
}
