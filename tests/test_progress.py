from __future__ import annotations

import io

import dspy
import pytest

from rlmoptimizer import RLMDocstringOptimizer
from rlmoptimizer.kernel import OptimizationKernel
from rlmoptimizer.progress import RichProgressReporter

from ._helpers import RuleProgram, build_trainset, exact_metric


class _RecorderReporter:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []
        self.closed = False

    def handle_event(self, event: dict[str, object]) -> None:
        self.events.append(dict(event))

    def close(self) -> None:
        self.closed = True


def test_kernel_progress_dispatches_internal_and_external_callbacks(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    reporter = _RecorderReporter()
    create_kwargs: list[dict[str, object]] = []

    def _fake_create(**kwargs):
        create_kwargs.append(dict(kwargs))
        return reporter

    monkeypatch.setattr("rlmoptimizer.kernel.create_progress_reporter", _fake_create)

    kernel = OptimizationKernel(
        program=RuleProgram(),
        trainset=build_trainset(4),
        valset=None,
        metric=exact_metric,
        eval_lm=None,
        num_threads=2,
        max_iterations=2,
        max_output_chars=20_000,
        run_storage_dir=tmp_path / "runs",
    )
    external_events: list[dict[str, object]] = []
    payload = kernel.evaluate_program(
        split="train",
        limit=2,
        _progress_callback=lambda event: external_events.append(dict(event)),
    )

    assert payload["evaluated_count"] == 2
    assert create_kwargs == [{"use_rich": False, "console": None}]
    assert reporter.closed is True
    assert reporter.events[0]["stage"] == "evaluation_started"
    assert sum(1 for event in reporter.events if event.get("stage") == "example_finished") == 2
    assert len(external_events) == len(reporter.events)

    kernel.close()


def test_kernel_progress_passes_debug_console_to_reporter(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    class _DebugDisplay:
        def __init__(self, console: object) -> None:
            self._console = console

        def progress_console(self) -> object:
            return self._console

    reporter = _RecorderReporter()
    create_kwargs: list[dict[str, object]] = []
    console = object()

    def _fake_create(**kwargs):
        create_kwargs.append(dict(kwargs))
        return reporter

    monkeypatch.setattr("rlmoptimizer.kernel.create_progress_reporter", _fake_create)

    kernel = OptimizationKernel(
        program=RuleProgram(),
        trainset=build_trainset(3),
        valset=None,
        metric=exact_metric,
        eval_lm=None,
        num_threads=1,
        max_iterations=2,
        max_output_chars=20_000,
        run_storage_dir=tmp_path / "runs",
        debug_display=_DebugDisplay(console=console),
    )
    payload = kernel.evaluate_program(split="train", limit=1)

    assert payload["evaluated_count"] == 1
    assert create_kwargs == [{"use_rich": True, "console": console}]
    assert reporter.closed is True
    assert reporter.events[0]["stage"] == "evaluation_started"

    kernel.close()


class _ProgressSession:
    def __init__(self, **_kwargs) -> None:
        pass

    def run(self, kernel: OptimizationKernel, *, objective: str):
        del objective
        _ = kernel.optimization_status()
        return {
            "optimized_dspy_program": "",
            "best_run_id": kernel.state.best_run_id,
            "agent_report": "ok",
            "trajectory": [],
            "final_reasoning": "",
        }


def test_optimizer_uses_single_progress_path():
    optimizer = RLMDocstringOptimizer(
        max_iterations=3,
        root_lm=dspy.LM("openai/mock-root"),
        eval_lm=dspy.LM("openai/mock-eval"),
        session_cls=_ProgressSession,
    )

    optimized = optimizer.compile(
        student=RuleProgram(),
        trainset=build_trainset(3),
        metric=exact_metric,
    )
    assert optimized.best_run_id is not None


def test_rich_progress_reporter_renders_panel_and_closes():
    _ = pytest.importorskip("rich")
    from rich.console import Console

    console = Console(
        file=io.StringIO(),
        force_terminal=False,
        width=100,
    )
    reporter = RichProgressReporter(console=console)

    reporter.handle_event(
        {
            "stage": "evaluation_started",
            "completed": 0,
            "total": 2,
        }
    )

    panel = reporter._build_panel()
    assert panel.title == "[bold]Evaluation[/bold]"
    assert reporter._live is not None
    assert reporter._live.transient is False

    reporter.handle_event(
        {
            "stage": "example_finished",
            "completed": 1,
            "total": 2,
            "score": 1.0,
        }
    )

    reporter.close()
    assert reporter._live is None
