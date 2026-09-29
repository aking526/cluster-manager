"use client";

import { useCallback, useEffect, useState } from "react";

type Process = { pid: number; name: string; user: string | null; used_memory_mib: number | null };
type Gpu = {
  index: number; uuid: string; name: string;
  memory_total_mib: number | null; memory_used_mib: number | null; processes: Process[];
};
type Checkpoint = { name: string; path: string; size_bytes: number; modified_at: string };
type Host = {
  cluster: string; name: string; host: string; gpu_error: string | null; project_error: string | null;
  gpu_checked_at: string | null; project_checked_at: string | null; project_dirs: string[];
  gpus: Gpu[]; files: Checkpoint[];
};
type State = {
  hosts: Host[];
  refreshing: boolean; refresh_seconds: number; last_refresh_at: string | null;
};
type View = "gpus" | "files";
type Status = "idle" | "busy" | "unknown" | "waiting";

function timestamp(value: string | null) {
  return value ? new Date(value).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }) : "Not checked yet";
}

function bytes(value: number) {
  if (value < 1024) return `${value} B`;
  const unit = Math.floor(Math.log(value) / Math.log(1024));
  return `${(value / 1024 ** unit).toFixed(1)} ${["B", "KB", "MB", "GB", "TB"][unit] || "PB"}`;
}

function memory(value: number | null) {
  return value === null ? "N/A" : `${value.toLocaleString()} MiB`;
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
    </div>
    {gpu.processes.length ? <table className="process-table"><thead><tr><th>PID</th><th>Process</th><th>Owner</th><th>GPU memory</th></tr></thead><tbody>
      {gpu.processes.map((process, i) => <tr key={`${process.pid}-${i}`}><td>{process.pid}</td><td>{process.name}</td><td>{process.user || "Unknown"}</td><td>{memory(process.used_memory_mib)}</td></tr>)}
    </tbody></table> : <p className="detail-text">No active compute processes observed on this GPU.</p>}
  </div>;
}

function HostCard({ host, view, selected, onSelect }: {
  host: Host; view: View; selected: string | null; onSelect: (key: string) => void;
}) {
  const hostKey = `${host.cluster}/${host.name}`;
  const error = view === "gpus" ? host.gpu_error : host.project_error;
  const checked = view === "gpus" ? host.gpu_checked_at : host.project_checked_at;
  const hostStatus: Status = error ? "unknown" : !checked ? "waiting" : view === "gpus" && host.gpus.some((gpu) => gpu.processes.length) ? "busy" : "idle";
  return <article className="host-card">
    <div className="host-heading">
      <div className="host-identity"><div className="host-title">
        <h2>{host.name}</h2><small>{host.cluster} · {host.host}</small>
      </div></div>
      {badge(hostStatus)}
    </div>
    {error && <div className="error-banner">{view === "gpus" ? "GPU query" : "Checkpoint scan"} failed: {error}{checked ? " · Showing last successful data" : ""}</div>}
    {view === "gpus" ? <>
      {host.gpus.map((gpu) => {
        const key = `${hostKey}/gpu/${gpu.uuid}`;
        const expanded = selected === key;
        const owners = [...new Set(gpu.processes.map((process) => process.user || "Unknown user"))];
        const percent = gpu.memory_total_mib ? Math.min(100, Math.max(0, (gpu.memory_used_mib || 0) / gpu.memory_total_mib * 100)) : 0;
        return <div key={key}><button className="gpu-row" type="button" aria-expanded={expanded} onClick={() => onSelect(key)}>
          <span className="gpu-model"><span className="gpu-name"><span className="gpu-index">GPU {gpu.index}</span>{gpu.name}</span></span>
          <span className="gpu-status">{badge(gpuStatus(host, gpu))}</span>
          <span className="gpu-memory"><span className="memory-label">{memory(gpu.memory_used_mib)} / {memory(gpu.memory_total_mib)}</span><span className="memory-track"><span className={`memory-fill ${percent > 75 ? "heavy" : ""}`} style={{ display: "block", width: `${percent}%` }} /></span></span>
          <span className="gpu-processes"><span className="process-count">{`${gpu.processes.length} ${gpu.processes.length === 1 ? "process" : "processes"}`}{host.gpu_error ? " (stale)" : ""}</span>{owners.length > 0 && <span className="process-owners"><span className="owner-label">{owners.length === 1 ? "User" : "Users"}</span> {owners.join(", ")}</span>}</span><span className="row-chevron" aria-hidden="true">›</span>
        </button>{expanded && <GpuDetail gpu={gpu} host={host} />}</div>;
      })}
      {!host.gpus.length && <div className="empty-host">{error ? "No successful GPU snapshot yet." : "Waiting for GPU data…"}</div>}
    </> : <>
      {host.files.map((file) => {
        const key = `${hostKey}/file/${file.path}`;
        const expanded = selected === key;
        return <div key={key}><button className="file-row" type="button" aria-expanded={expanded} onClick={() => onSelect(key)}>
          <span className="file-name"><span className="file-icon" aria-hidden="true">▤</span><span>{file.name}</span></span>
          <span className="file-size">{bytes(file.size_bytes)}</span><span className="file-modified">Modified {timestamp(file.modified_at)}</span><span className="row-chevron">›</span>
        </button>{expanded && <div className="detail-panel"><div className="detail-kicker">CHECKPOINT DETAILS {error ? "· STALE SCAN" : ""}</div><div className="file-path">{file.path}</div>
          <div className="detail-grid" style={{ marginTop: 16 }}><div className="detail-cell"><small>Size</small><span>{file.size_bytes.toLocaleString()} bytes</span></div><div className="detail-cell"><small>Modified</small><span>{timestamp(file.modified_at)}</span></div><div className="detail-cell"><small>Last scanned</small><span>{timestamp(checked)}</span></div></div>
        </div>}</div>;
      })}
      {!host.project_dirs.length && <div className="empty-host">No project directories configured for this cluster.</div>}
      {!!host.project_dirs.length && !host.files.length && <div className="empty-host">{error ? "No successful checkpoint scan yet." : checked ? "No checkpoints found in configured directories." : "Waiting for checkpoint scan…"}</div>}
    </>}
    <div className="host-footer"><span>{view === "gpus" ? `${host.gpus.length} ${host.gpus.length === 1 ? "GPU" : "GPUs"}` : `${host.files.length} checkpoints`}</span><span>Checked {timestamp(checked)}</span></div>
  </article>;
}

export default function Home() {
  const [data, setData] = useState<State | null>(null);
  const [view, setView] = useState<View>("gpus");
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
    <HostCard key={`${host.cluster}/${host.name}`} host={host} view={view} selected={selected} onSelect={(key) => setSelected(selected === key ? null : key)} />
  );
  function changeView(next: View) { setView(next); setSelected(null); }
  return <main className="content" aria-label="Cluster manager">
    <div className="view-controls">
      <div className="view-switch" role="group" aria-label="Inventory view">{(["gpus", "files"] as const).map((tab) => <button key={tab} className={`view-tab ${view === tab ? "active" : ""}`} onClick={() => changeView(tab)} aria-pressed={view === tab} type="button">{tab === "gpus" ? "GPUs" : "Checkpoints"}</button>)}</div>
      <button className="refresh-button" onClick={() => void refresh()} disabled={data?.refreshing} type="button" aria-label={data?.refreshing ? "Refreshing data" : "Refresh data"} title={data?.refreshing ? "Refreshing data" : "Refresh data"}>
        <svg className="refresh-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M20 7v5h-5M4 17v-5h5" /><path d="M6.1 7a7 7 0 0 1 11.6-1L20 9M4 15l2.3 3A7 7 0 0 0 17.9 17" /></svg>
      </button>
    </div>
    {error && <div className="api-error" role="alert">{error}</div>}
    <section className="inventory-list" aria-label="Nodes" aria-live="polite">{cards.length ? cards : <div className="empty-state"><strong>{data ? "No nodes configured" : "Waiting for cluster data"}</strong>{data ? "Add nodes to your configuration to see them here." : "Your nodes will appear after the first query."}</div>}</section>
  </main>;
}
