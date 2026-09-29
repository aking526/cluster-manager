"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";

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
  summary: { idle: number; busy: number; unknown_hosts: number; checkpoints: number; total_hosts: number };
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

function hostVisible(host: Host, view: View, search: string, filter: string) {
  const matches = (text: string) => text.toLowerCase().includes(search);
  const hostMatch = [host.cluster, host.name, host.host].some(matches);
  if (view === "files") return hostMatch || host.files.some((file) => matches(file.path));
  if (filter === "unknown" && !host.gpu_error && host.gpu_checked_at) return false;
  if (filter === "unknown" && !host.gpu_checked_at && hostMatch) return true;
  return host.gpus.some((gpu) =>
    (!filter || gpuStatus(host, gpu) === filter) &&
    (hostMatch || [gpu.name, gpu.uuid, ...gpu.processes.flatMap((p) => [p.name, p.user || "", String(p.pid)])].some(matches))
  ) || (!filter && hostMatch);
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

function HostCard({ host, view, search, filter, selected, onSelect }: {
  host: Host; view: View; search: string; filter: string; selected: string | null; onSelect: (key: string) => void;
}) {
  const hostKey = `${host.cluster}/${host.name}`;
  const matches = (text: string) => text.toLowerCase().includes(search);
  const hostMatch = [host.cluster, host.name, host.host].some(matches);
  const gpus = host.gpus.filter((gpu) =>
    (!filter || gpuStatus(host, gpu) === filter) &&
    (hostMatch || [gpu.name, gpu.uuid, ...gpu.processes.flatMap((p) => [p.name, p.user || "", String(p.pid)])].some(matches))
  );
  const fileList = host.files.filter((file) => hostMatch || matches(file.path));
  const error = view === "gpus" ? host.gpu_error : host.project_error;
  const checked = view === "gpus" ? host.gpu_checked_at : host.project_checked_at;
  const hostStatus: Status = error ? "unknown" : !checked ? "waiting" : view === "gpus" && host.gpus.some((gpu) => gpu.processes.length) ? "busy" : "idle";
  return <article className="host-card">
    <div className="host-heading">
      <div className="host-identity"><span className="host-avatar" aria-hidden="true">▦</span><div className="host-title">
        <strong>{host.name}</strong><small>{host.cluster} · {host.host}</small>
      </div></div>
      <div className="host-meta"><span>Checked {timestamp(checked)}</span>{badge(hostStatus)}</div>
    </div>
    {error && <div className="error-banner">{view === "gpus" ? "GPU query" : "Checkpoint scan"} failed: {error}{checked ? " · Showing last successful data" : ""}</div>}
    {view === "gpus" ? <>
      {gpus.length > 0 && <div className="gpu-columns"><span>GPU / MODEL</span><span>STATUS</span><span>MEMORY USAGE</span><span>PROCESSES</span><span></span></div>}
      {gpus.map((gpu) => {
        const key = `${hostKey}/gpu/${gpu.uuid}`;
        const expanded = selected === key;
        const percent = gpu.memory_total_mib ? Math.min(100, Math.max(0, (gpu.memory_used_mib || 0) / gpu.memory_total_mib * 100)) : 0;
        return <div key={key}><button className="gpu-row" type="button" aria-expanded={expanded} onClick={() => onSelect(key)}>
          <span><span className="gpu-name"><span className="gpu-index">#{gpu.index}</span>{gpu.name}</span><span className="gpu-sub">{gpu.uuid}</span></span>
          <span className="gpu-status">{badge(gpuStatus(host, gpu))}</span>
          <span><span className="memory-label">{memory(gpu.memory_used_mib)} / {memory(gpu.memory_total_mib)}</span><span className="memory-track"><span className={`memory-fill ${percent > 75 ? "heavy" : ""}`} style={{ display: "block", width: `${percent}%` }} /></span></span>
          <span className="process-count">{host.gpu_error ? "—" : gpu.processes.length}</span><span className="row-chevron">›</span>
        </button>{expanded && <GpuDetail gpu={gpu} host={host} />}</div>;
      })}
      {!host.gpus.length && <div className="empty-host">{error ? "No successful GPU snapshot yet." : "Waiting for GPU data…"}</div>}
    </> : <>
      {fileList.length > 0 && <div className="gpu-columns file-columns"><span>CHECKPOINT FILE</span><span>SIZE</span><span>MODIFIED</span><span></span></div>}
      {fileList.map((file) => {
        const key = `${hostKey}/file/${file.path}`;
        const expanded = selected === key;
        return <div key={key}><button className="file-row" type="button" aria-expanded={expanded} onClick={() => onSelect(key)}>
          <span className="file-name"><span className="file-icon" aria-hidden="true">▤</span><span>{file.name}</span></span>
          <span>{bytes(file.size_bytes)}</span><span className="file-modified">{timestamp(file.modified_at)}</span><span className="row-chevron">›</span>
        </button>{expanded && <div className="detail-panel"><div className="detail-kicker">CHECKPOINT DETAILS {error ? "· STALE SCAN" : ""}</div><div className="file-path">{file.path}</div>
          <div className="detail-grid" style={{ marginTop: 16 }}><div className="detail-cell"><small>Size</small><span>{file.size_bytes.toLocaleString()} bytes</span></div><div className="detail-cell"><small>Modified</small><span>{timestamp(file.modified_at)}</span></div><div className="detail-cell"><small>Last scanned</small><span>{timestamp(checked)}</span></div></div>
        </div>}</div>;
      })}
      {!host.project_dirs.length && <div className="empty-host">No project directories configured for this cluster.</div>}
      {!!host.project_dirs.length && !host.files.length && <div className="empty-host">{error ? "No successful checkpoint scan yet." : checked ? "No checkpoints found in configured directories." : "Waiting for checkpoint scan…"}</div>}
    </>}
  </article>;
}

export default function Home() {
  const [data, setData] = useState<State | null>(null);
  const [view, setView] = useState<View>("gpus");
  const [search, setSearch] = useState("");
  const [cluster, setCluster] = useState("");
  const [filter, setFilter] = useState("");
  const [selected, setSelected] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [clock, setClock] = useState("");

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
    const tick = () => setClock(new Date().toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" }));
    tick();
    const timer = window.setInterval(tick, 30000);
    return () => { window.clearInterval(poll); window.clearInterval(timer); };
  }, [load]);
  useEffect(() => {
    const key = (event: KeyboardEvent) => {
      if (event.key === "/" && !(event.target instanceof HTMLInputElement) && !(event.target instanceof HTMLSelectElement)) {
        event.preventDefault();
        document.getElementById("search")?.focus();
      }
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, []);

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

  const clusters = useMemo(() => [...new Set(data?.hosts.map((host) => host.cluster) || [])].sort(), [data]);
  const cards = (data?.hosts || []).filter((host) =>
    (!cluster || host.cluster === cluster) && hostVisible(host, view, search.toLowerCase().trim(), filter)
  ).map((host) =>
    <HostCard key={`${host.cluster}/${host.name}`} host={host} view={view} search={search.toLowerCase().trim()} filter={filter} selected={selected} onSelect={(key) => setSelected(selected === key ? null : key)} />
  );
  function changeView(next: View) { setView(next); setSelected(null); setFilter(""); }
  return <div className="app-shell">
    <aside className="sidebar">
      <Link className="brand" href="/" aria-label="Cluster Manager home"><span className="brand-mark"><span /><span /><span /><span /></span><span>cluster<span className="brand-light">manager</span><small>INFRASTRUCTURE CONSOLE</small></span></Link>
      <div className="nav-label">WORKSPACE</div>
      <nav aria-label="Main navigation">
        <button className={`nav-item ${view === "gpus" ? "active" : ""}`} onClick={() => changeView("gpus")} type="button"><span className="nav-icon">▦</span>Overview</button>
        <button className={`nav-item ${view === "files" ? "active" : ""}`} onClick={() => changeView("files")} type="button"><span className="nav-icon">◫</span>Checkpoints</button>
      </nav>
      <div className="sidebar-bottom"><div className="sidebar-status"><span className="live-dot" /> LOCAL CONNECTION <span className="status-arrow">↗</span></div><p>Read-only visibility into your GPU fleet. Data is queried directly over SSH.</p><div className="sidebar-footer">CLUSTER MANAGER <span>v0.1</span></div></div>
    </aside>
    <main className="main">
      <header className="topbar"><div className="breadcrumb">Workspace <span>/</span> <strong>{view === "gpus" ? "Overview" : "Checkpoints"}</strong></div><div className="topbar-right"><span className="topbar-clock">{clock}</span><span className="topbar-divider" /><span className="local-badge"><span /> Local dashboard</span></div></header>
      <div className="content">
        <div className="page-heading"><div><div className="eyebrow"><span className="eyebrow-line" /> INFRASTRUCTURE / MONITORING</div><h1>Cluster overview<span className="heading-dot">.</span></h1><p>Real-time visibility into your GPU hosts and project checkpoints.</p></div><button className="refresh-button" onClick={() => void refresh()} disabled={data?.refreshing} type="button"><span className="refresh-icon">↻</span> Refresh data</button></div>
        <div className="status-strip"><span className="status-pulse" /><span id="refresh-label">{error || (data?.refreshing ? "Refreshing hosts…" : data?.last_refresh_at ? "Monitoring active" : "Connecting to hosts…")}</span><span className="status-separator">·</span><span>{data?.last_refresh_at ? `Last refresh ${timestamp(data.last_refresh_at)}` : "Awaiting first refresh"}</span><span className="status-separator">·</span><span>{data?.refresh_seconds ? `Auto refresh every ${data.refresh_seconds}s` : "Manual refresh"}</span></div>
        <section className="metrics" aria-label="Cluster metrics">
          {([
            ["GPUs available", data?.summary.idle, "No compute processes", "green", "↗"],
            ["GPUs in use", data?.summary.busy, "Active compute processes", "violet", "◈"],
            ["Unknown hosts", data?.summary.unknown_hosts, `Of ${data?.summary.total_hosts ?? "—"} configured hosts`, "amber", "◇"],
            ["Checkpoints", data?.summary.checkpoints, "Last successful scans", "blue", "▤"],
          ] as const).map(([title, value, description, color, icon]) => <div className="metric-card" key={title}><div className="metric-top"><span>{title}</span><span className={`metric-symbol ${color}`}>{icon}</span></div><div className="metric-value">{value ?? "—"}</div><div className="metric-bottom"><span className={`metric-indicator ${color}`} />{description}</div></div>)}
        </section>
        <section className="inventory" aria-labelledby="inventory-title">
          <div className="inventory-header"><div><div className="section-kicker">FLEET INVENTORY</div><h2 id="inventory-title">Your infrastructure <span className="count-badge">{cards.length}</span></h2></div><div className="view-switch" role="tablist" aria-label="Inventory view">{(["gpus", "files"] as const).map((tab) => <button key={tab} className={`view-tab ${view === tab ? "active" : ""}`} onClick={() => changeView(tab)} role="tab" aria-selected={view === tab} type="button">{tab === "gpus" ? "GPU hosts" : "Checkpoints"}</button>)}</div></div>
          <div className="toolbar">
            <label className="search"><span aria-hidden="true">⌕</span><input type="search" id="search" placeholder="Search hosts, GPUs, files..." aria-label="Search inventory" value={search} onChange={(event) => setSearch(event.target.value)} /><kbd>/</kbd></label>
            <label className="select-wrap"><span>Cluster</span><select aria-label="Filter by cluster" value={cluster} onChange={(event) => setCluster(event.target.value)}><option value="">All clusters</option>{clusters.map((name) => <option key={name} value={name}>{name}</option>)}</select></label>
            {view === "gpus" && <label className="select-wrap"><span>Status</span><select aria-label="Filter by GPU status" value={filter} onChange={(event) => setFilter(event.target.value)}><option value="">All statuses</option><option value="idle">Available</option><option value="busy">In use</option><option value="unknown">Unknown</option></select></label>}
          </div>
          <div className="inventory-list" aria-live="polite">{cards.length ? cards : <div className="empty-state"><span className="empty-icon">◇</span><strong>{data ? "No matching hosts" : "Waiting for cluster data"}</strong>{data ? "Try a different search or filter." : "The inventory will appear after the first query."}</div>}</div>
        </section>
        <footer className="footer-note"><span>ⓘ</span> Available means no compute processes observed. Scheduler reservations and NVIDIA MIG partitions are not tracked.</footer>
      </div>
    </main>
  </div>;
}
