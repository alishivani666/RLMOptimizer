from __future__ import annotations

import sys
from typing import Any, Protocol

import tqdm


class ProgressReporter(Protocol):
    def handle_event(self, event: dict[str, Any]) -> None: ...

    def close(self) -> None: ...


class DspyStyleProgressReporter:
    """Render evaluator progress similarly to DSPy Evaluate."""

    def __init__(self) -> None:
        self._bar: Any | None = None
        self._score_sum = 0.0
        self._finished_count = 0
        self._completed = 0
        self._total = 0

    def handle_event(self, event: dict[str, Any]) -> None:
        stage = event.get("stage")
        if stage == "evaluation_started":
            self._start(total=_coerce_int(event.get("total")) or 0)
            return
        if stage != "example_finished":
            return

        score = _coerce_float(event.get("score")) or 0.0
        completed = _coerce_int(event.get("completed"))
        self._finished_count += 1
        self._score_sum += score

        if completed is None:
            target_completed = self._completed + 1
        else:
            target_completed = max(completed, self._completed)
        if self._total > 0:
            target_completed = min(target_completed, self._total)

        delta = max(0, target_completed - self._completed)
        self._completed = target_completed

        if self._bar is None:
            return

        self._set_description()
        if delta > 0:
            self._bar.update(delta)

    def close(self) -> None:
        if self._bar is None:
            return
        self._bar.close()
        self._bar = None

    def _start(self, *, total: int) -> None:
        self.close()
        self._score_sum = 0.0
        self._finished_count = 0
        self._completed = 0
        self._total = max(total, 0)

        try:
            tqdm.tqdm._instances.clear()
        except Exception:
            pass

        self._bar = tqdm.tqdm(
            total=self._total,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        self._set_description()

    def _set_description(self) -> None:
        if self._bar is None:
            return
        if self._finished_count > 0:
            pct = round(100 * self._score_sum / self._finished_count, 1)
        else:
            pct = 0.0
        self._bar.set_description(
            f"Average Metric: {self._score_sum:.2f} / {self._finished_count} ({pct}%)"
        )


class RichProgressReporter:
    """Progress reporter using rich.progress instead of tqdm."""

    def __init__(self, *, console: Any | None = None) -> None:
        self._console = console
        self._progress = self._new_progress(console=console)
        self._live: Any | None = None
        self._task_id: Any | None = None
        self._score_sum = 0.0
        self._finished_count = 0

    def handle_event(self, event: dict[str, Any]) -> None:
        stage = event.get("stage")
        if stage == "evaluation_started":
            total = _coerce_int(event.get("total")) or 0
            self._start(total=total)
            return
        if stage != "example_finished":
            return

        score = _coerce_float(event.get("score")) or 0.0
        self._finished_count += 1
        self._score_sum += score

        if self._task_id is not None:
            if self._finished_count > 0:
                pct = round(100 * self._score_sum / self._finished_count, 1)
            else:
                pct = 0.0
            self._progress.update(
                self._task_id,
                advance=1,
                description=f"Average Metric: {self._score_sum:.2f} / {self._finished_count} ({pct}%)",
            )
            self._refresh_live()

    def close(self) -> None:
        try:
            if self._live is not None:
                self._live.stop()
                self._live = None
        except Exception:
            pass
        self._task_id = None

    def _start(self, *, total: int) -> None:
        from rich.live import Live

        self.close()
        self._score_sum = 0.0
        self._finished_count = 0
        self._progress = self._new_progress(console=self._console)
        self._task_id = self._progress.add_task("Evaluating...", total=max(total, 0))
        self._live = Live(
            self._build_panel(),
            console=self._console,
            refresh_per_second=8,
            transient=False,
        )
        self._live.start()
        self._refresh_live()

    def _new_progress(self, *, console: Any | None) -> Any:
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            TextColumn,
            TimeRemainingColumn,
        )

        return Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=console,
        )

    def _build_panel(self) -> Any:
        from rich.panel import Panel

        return Panel(
            self._progress,
            title="[bold]Evaluation[/bold]",
            border_style="bright_blue",
            padding=(0, 1),
        )

    def _refresh_live(self) -> None:
        if self._live is None:
            return
        self._live.update(self._build_panel(), refresh=True)


def create_progress_reporter(
    *,
    use_rich: bool = False,
    console: Any | None = None,
) -> ProgressReporter:
    if use_rich:
        try:
            import rich  # noqa: F401

            return RichProgressReporter(console=console)
        except ImportError:
            pass
    return DspyStyleProgressReporter()


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
