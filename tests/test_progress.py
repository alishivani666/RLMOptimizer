from __future__ import annotations

import dspy
import pytest

from rlmoptimizer import RLMDocstringOptimizer
from rlmoptimizer.kernel import OptimizationKernel

from ._helpers import RuleProgram, build_trainset, exact_metric


def test_kernel_progress_dispatches_internal_and_external_callbacks(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    class _RecorderReporter:
        def __init__(self) -> None:
            self.events: list[dict[str, object]] = []
            self.closed = False

        def handle_event(self, event: dict[str, object]) -> None:
            self.events.append(dict(event))

        def close(self) -> None:
            self.closed = True

    reporter = _RecorderReporter()
    created_modes: list[str] = []

    def _fake_create():
        created_modes.append("created")
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
    assert created_modes == ["created"]
    assert reporter.closed is True
    assert reporter.events[0]["stage"] == "evaluation_started"
    assert sum(1 for event in reporter.events if event.get("stage") == "example_finished") == 2
    assert len(external_events) == len(reporter.events)

    kernel.close()


class _ProgressSession:
    def __init__(self, **_kwargs) -> None:
        pass

    def run(self, kernel: OptimizationKernel, *, objective: str):
        del objective
        _ = kernel.optimization_status()
        return {
            "final_report": "ok",
            "suggested_best_run_id": kernel.state.best_run_id,
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
