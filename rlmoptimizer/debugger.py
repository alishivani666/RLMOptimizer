"""Rich-based real-time debugger for RLMOptimizer optimization runs."""

from __future__ import annotations

from typing import Protocol


class DebugDisplay(Protocol):
    """Protocol for optimization display backends."""

    def show_header(
        self, *, model: str, train_size: int, val_size: int
    ) -> None: ...

    def show_baseline(
        self, *, score: float, budget: int, steps: list[str]
    ) -> None: ...

    def show_iteration(
        self,
        iteration: int,
        max_iterations: int,
        reasoning: str,
        code: str,
        output: str,
    ) -> None: ...

    def show_final_summary(
        self,
        *,
        baseline_score: float,
        best_score: float,
        budget_used: int,
        total_budget: int,
        iterations: int,
    ) -> None: ...

    def close(self) -> None: ...


class RichDebugDisplay:
    """Rich-based display that renders panels, syntax-highlighted code, and tables."""

    def __init__(self) -> None:
        from rich.console import Console

        self._console = Console()

    def show_header(
        self, *, model: str, train_size: int, val_size: int
    ) -> None:
        from rich.panel import Panel

        parts = []
        if model:
            parts.append(f"Model: {model}")
        parts.append(f"Train: {train_size}")
        if val_size > 0:
            parts.append(f"Val: {val_size}")
        detail_line = "  |  ".join(parts)

        self._console.print()
        self._console.print(
            Panel(
                f"[bold]RLMOptimizer[/bold] — Optimization Run\n"
                f"[dim]{detail_line}[/dim]",
                border_style="bright_blue",
                padding=(1, 2),
            )
        )

    def show_baseline(
        self, *, score: float, budget: int, steps: list[str]
    ) -> None:
        from rich import box
        from rich.panel import Panel
        from rich.table import Table

        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        table.add_column(style="bold")
        table.add_column()
        table.add_row("Baseline Score", f"[red]{score}%[/red]")
        table.add_row("Budget", f"{budget:,}")
        table.add_row("Steps", " -> ".join(steps))
        self._console.print()
        self._console.print(
            Panel(table, title="[bold]Baseline[/bold]", border_style="red")
        )

    def show_iteration(
        self,
        iteration: int,
        max_iterations: int,
        reasoning: str,
        code: str,
        output: str,
    ) -> None:
        from rich.panel import Panel
        from rich.syntax import Syntax
        from rich.text import Text

        self._console.print()
        self._console.print(
            Panel(
                Text(reasoning, style="italic"),
                title=(
                    f"[bold cyan]Iteration {iteration + 1}[/bold cyan] "
                    f"[dim]/ {max_iterations}[/dim]  "
                    f"[bold yellow]Reasoning[/bold yellow]"
                ),
                border_style="cyan",
                padding=(0, 1),
            )
        )

        self._console.print(
            Panel(
                Syntax(
                    code,
                    "python",
                    theme="monokai",
                    line_numbers=False,
                    word_wrap=True,
                ),
                title="[bold green]Code[/bold green]",
                border_style="green",
                padding=(0, 1),
            )
        )

        self._console.print(
            Panel(
                Text(output),
                title="[bold yellow]Output[/bold yellow]",
                border_style="yellow",
                padding=(0, 1),
            )
        )

    def show_final_summary(
        self,
        *,
        baseline_score: float,
        best_score: float,
        budget_used: int,
        total_budget: int,
        iterations: int,
    ) -> None:
        from rich import box
        from rich.table import Table

        delta = best_score - baseline_score
        delta_str = f"+{delta:.2f}" if delta >= 0 else f"{delta:.2f}"
        budget_pct = (
            f"{100 * budget_used / total_budget:.1f}%"
            if total_budget > 0
            else "--"
        )

        table = Table(
            box=box.ROUNDED,
            title="Optimization Summary",
            title_style="bold bright_blue",
        )
        table.add_column("Metric", style="bold")
        table.add_column("Baseline", justify="center")
        table.add_column("Optimized", justify="center")
        table.add_column("Delta", justify="center")
        table.add_row(
            "Train Score",
            f"[red]{baseline_score:.2f}%[/red]",
            f"[green]{best_score:.2f}%[/green]",
            f"[bold green]{delta_str}[/bold green]",
        )
        table.add_row(
            "Budget Used",
            "[dim]--[/dim]",
            f"{budget_used:,} / {total_budget:,}",
            f"[dim]{budget_pct}[/dim]",
        )
        table.add_row(
            "Iterations",
            "[dim]--[/dim]",
            str(iterations),
            "[dim]--[/dim]",
        )
        self._console.print()
        self._console.print(table)
        self._console.print()

    def close(self) -> None:
        pass


def create_debug_display(*, verbose: bool) -> DebugDisplay:
    """Return RichDebugDisplay when verbose and rich is installed, else None."""
    if not verbose:
        return None
    try:
        import rich  # noqa: F401

        return RichDebugDisplay()
    except ImportError:
        return None
