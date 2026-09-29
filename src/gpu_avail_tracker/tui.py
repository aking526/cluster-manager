"""Textual dashboard for current and stale GPU snapshots."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from rich.text import Text
from textual.app import App, ComposeResult
from textual.widgets import DataTable, Footer, Header, Static

from .config import Settings, Target
from .probe import Gpu, ProbeError, Snapshot, query_target


@dataclass
class HostState:
    snapshot: Snapshot | None = None
    error: str | None = None


def _safe_display(value: str, limit: int = 120) -> str:
    value = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", value).strip()
    return value[: limit - 1] + "…" if len(value) > limit else value


def _memory(used: int | None, total: int | None, used_width: int) -> Text:
    if used is None or total is None:
        return Text("N/A")
    return Text(f"{used:>{used_width}} / {total}")


class GPUTrackerApp(App[None]):
    TITLE = "GPU Availability"
    SUB_TITLE = "Observed compute processes"
    BINDINGS = [("r", "refresh_gpus", "Refresh"), ("q", "quit", "Quit")]
    CSS = """
    Screen { layout: vertical; }
    #summary { height: 5; padding: 1 2; color: $text; }
    #gpus { height: 1fr; margin: 0 1; border: round $primary; }
    #details { height: 8; margin: 1; padding: 1 2; border: round $secondary; overflow-y: auto; }
    """

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.settings = settings
        self.states: dict[tuple[str, str], HostState] = {
            target.key: HostState() for target in settings.targets
        }
        self._rows: list[tuple[Target, Gpu | None]] = []
        self._row_ids: list[tuple[str, str, str]] = []
        self._refreshing = False
        self._last_refresh_at: datetime | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static(id="summary")
        yield DataTable(id="gpus")
        yield Static(id="details")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#gpus", DataTable)
        table.add_columns("GPU", "Model", "Status", "Memory (MiB)", "Procs")
        table.cursor_type = "row"
        table.zebra_stripes = True
        self._render()
        self._start_refresh()
        if self.settings.refresh_seconds:
            self.set_interval(self.settings.refresh_seconds, self._start_refresh)

    def action_refresh_gpus(self) -> None:
        self._start_refresh()

    def _start_refresh(self) -> None:
        if self._refreshing:
            return
        self._refreshing = True
        self._render_summary()
        self.run_worker(self._refresh_all(), group="gpu-refresh")

    async def _refresh_all(self) -> None:
        semaphore = asyncio.Semaphore(4)

        async def fetch(target: Target) -> tuple[Target, Snapshot | None, str | None]:
            async with semaphore:
                try:
                    return target, await query_target(target), None
                except ProbeError as exc:
                    return target, None, str(exc)

        tasks = [asyncio.create_task(fetch(target)) for target in self.settings.targets]
        try:
            for completed in asyncio.as_completed(tasks):
                target, snapshot, error = await completed
                state = self.states[target.key]
                if snapshot is not None:
                    state.snapshot = snapshot
                    state.error = None
                else:
                    state.error = error or "Query failed"
                self._render()
            self._last_refresh_at = datetime.now(timezone.utc)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self._refreshing = False
            self._render_summary()

    def _render_summary(self) -> None:
        idle = busy = unknown_hosts = 0
        for state in self.states.values():
            if state.snapshot is None or state.error is not None:
                unknown_hosts += 1
                continue
            for gpu in state.snapshot.gpus:
                if gpu.processes:
                    busy += 1
                else:
                    idle += 1
        mode = "manual" if self.settings.refresh_seconds == 0 else f"every {self.settings.refresh_seconds}s"
        last_refresh = (
            self._last_refresh_at.astimezone().strftime("%Y-%m-%d %H:%M:%S")
            if self._last_refresh_at else "—"
        )
        progress = "  •  Refreshing…" if self._refreshing else ""
        text = Text(
            f"Idle {idle}   In use {busy}   Unknown hosts {unknown_hosts}\n"
            f"Refresh: {mode}  •  Last refresh: {last_refresh}{progress}\n"
            "Idle means no compute process observed; cluster reservations are not checked."
        )
        self.query_one("#summary", Static).update(text)

    def _render(self) -> None:
        table = self.query_one("#gpus", DataTable)
        used_width = max(
            (
                len(str(gpu.memory_used_mib))
                for state in self.states.values()
                for gpu in (state.snapshot.gpus if state.snapshot else ())
                if gpu.memory_used_mib is not None
            ),
            default=1,
        )
        selected_id = (
            self._row_ids[table.cursor_row]
            if 0 <= table.cursor_row < len(self._row_ids)
            else None
        )
        table.clear()
        self._rows.clear()
        self._row_ids.clear()
        for target in self.settings.targets:
            state = self.states[target.key]
            snapshot = state.snapshot
            table.add_row(
                Text(f"{target.cluster} / {target.name}", style="bold cyan"),
                "",
                "",
                "",
                "",
            )
            self._rows.append((target, None))
            self._row_ids.append((*target.key, "host"))
            if snapshot is None:
                status = "Unknown" if state.error else "Waiting"
                table.add_row("—", "No GPU data", Text(status, style="yellow"), "—", "—")
                self._rows.append((target, None))
                self._row_ids.append((*target.key, "placeholder"))
                continue
            for gpu in snapshot.gpus:
                status = "Unknown" if state.error else ("In use" if gpu.processes else "Idle")
                color = "yellow" if state.error else ("red" if gpu.processes else "green")
                table.add_row(
                    str(gpu.index),
                    _safe_display(gpu.name, 50),
                    Text(status, style=color),
                    _memory(gpu.memory_used_mib, gpu.memory_total_mib, used_width),
                    str(len(gpu.processes)) if not state.error else "—",
                )
                self._rows.append((target, gpu))
                self._row_ids.append((*target.key, gpu.uuid))
        if self._rows:
            selected = self._row_ids.index(selected_id) if selected_id in self._row_ids else 0
            table.move_cursor(row=selected)
            self._render_details(selected)
        self._render_summary()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        self._render_details(event.cursor_row)

    def _render_details(self, row: int) -> None:
        if row < 0 or row >= len(self._rows):
            return
        target, gpu = self._rows[row]
        state = self.states[target.key]
        detail = Text()
        detail.append(f"{_safe_display(target.cluster)} / {_safe_display(target.name)}", style="bold")
        detail.append(f" · {_safe_display(target.host)}\n")
        if state.error:
            detail.append(f"Query error: {_safe_display(state.error)}\n", style="yellow")
        if gpu is None:
            if state.snapshot is None:
                detail.append("No successful GPU snapshot yet.")
            else:
                detail.append(f"{len(state.snapshot.gpus)} GPUs in the last successful snapshot.\n")
                detail.append(
                    f"Last checked: {state.snapshot.checked_at.astimezone().strftime('%Y-%m-%d %H:%M:%S')}"
                )
                if state.error:
                    detail.append("  (stale)", style="yellow")
        else:
            detail.append(f"GPU {gpu.index}: {_safe_display(gpu.name)}  •  {gpu.uuid}\n")
            if state.error:
                detail.append("Process list below is stale.\n", style="yellow")
            if not gpu.processes:
                detail.append("No compute processes in the last successful snapshot.")
            else:
                for process in gpu.processes:
                    detail.append(
                        f"PID {process.pid:<8}  User {_safe_display(process.user or 'Unknown', 32):<32}  "
                        f"GPU memory {str(process.memory_mib) + ' MiB' if process.memory_mib is not None else 'N/A'}\n"
                        f"    {_safe_display(process.name)}\n"
                    )
        self.query_one("#details", Static).update(detail)
