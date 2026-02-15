from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import dspy
import pytest

from rlmoptimizer import RLMDocstringOptimizer
from rlmoptimizer.benchmarks.unified_harness import (
    LoadedBenchmark,
    ModelBundle,
    _run_single_benchmark,
    ensure_live_local_package,
    run_harness,
)

from ._helpers import RuleProgram, build_trainset, exact_metric


class _NoopSession:
    def __init__(self, **_kwargs) -> None:
        pass

    def run(self, kernel, *, objective: str):
        del objective
        _ = kernel.optimization_status()
        return {
            "optimized_dspy_program": "",
            "best_run_id": kernel.state.best_run_id,
            "agent_report": "noop",
            "trajectory": [],
            "final_reasoning": "",
        }


def _fake_bundle() -> ModelBundle:
    return ModelBundle(
        root_lm=dspy.LM("openai/mock-root"),
        sub_lm=dspy.LM("openai/mock-sub"),
        eval_lm=dspy.LM("openai/mock-eval"),
    )


def _fake_loader(key: str) -> LoadedBenchmark:
    return LoadedBenchmark(
        key=key,
        trainset=build_trainset(4),
        valset=build_trainset(2),
        testset=build_trainset(3),
        program=RuleProgram(),
        metric=exact_metric,
    )


def _bad_loader() -> LoadedBenchmark:
    raise RuntimeError("synthetic loader failure")


def test_harness_reuses_optimizer_baseline_for_pre_train_and_pre_val(tmp_path: Path):
    output_dir = tmp_path / "runs"
    run_summary = run_harness(
        run_label="baseline-reuse",
        benchmarks=["fake"],
        output_dir=output_dir,
        use_processes=False,
        loaders={"fake": lambda: _fake_loader("fake")},
        model_builder=_fake_bundle,
        session_cls=_NoopSession,
    )

    bench_summary = run_summary["benchmarks"]["fake"]
    bench_dir = output_dir / "baseline-reuse" / "fake"

    pre_train = json.loads((bench_dir / "pre_eval_train.json").read_text(encoding="utf-8"))
    pre_val = json.loads((bench_dir / "pre_eval_val.json").read_text(encoding="utf-8"))
    assert pre_train["run_id"] == bench_summary["baseline_train_run_id"]
    assert pre_val["run_id"] == bench_summary["baseline_val_run_id"]

    events = [
        json.loads(line)
        for line in (bench_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    phases = [
        str(row["payload"]["phase"])
        for row in events
        if row.get("source") == "harness"
        and row.get("event") == "external_evaluation_started"
    ]
    assert phases == ["pre_test", "post_train", "post_val", "post_test"]


def test_harness_isolates_failures_and_keeps_other_results(tmp_path: Path):
    output_dir = tmp_path / "runs"
    run_summary = run_harness(
        run_label="mixed",
        benchmarks=["good", "bad"],
        output_dir=output_dir,
        use_processes=False,
        loaders={"good": lambda: _fake_loader("good"), "bad": _bad_loader},
        model_builder=_fake_bundle,
        session_cls=_NoopSession,
    )

    assert run_summary["benchmarks"]["good"]["status"] == "ok"
    assert run_summary["benchmarks"]["bad"]["status"] == "error"
    assert "synthetic loader failure" in run_summary["benchmarks"]["bad"]["error"]

    good_dir = output_dir / "mixed" / "good"
    bad_dir = output_dir / "mixed" / "bad"
    assert (good_dir / "benchmark_summary.json").exists()
    assert (bad_dir / "benchmark_summary.json").exists()
    assert (good_dir / "post_eval_test.json").exists()


def test_progress_math_uses_correct_expected_total_and_excludes_baseline(tmp_path: Path):
    progress_messages: list[dict[str, Any]] = []
    run_root = tmp_path / "progress"
    summary = _run_single_benchmark(
        benchmark_key="fake",
        run_root=run_root,
        queue_sink=lambda message: progress_messages.append(dict(message)),
        loaders={"fake": lambda: _fake_loader("fake")},
        model_builder=_fake_bundle,
        optimizer_cls=RLMDocstringOptimizer,
        session_cls=_NoopSession,
    )

    assert summary["status"] == "ok"
    assert summary["expected_progress_total"] == 22  # 3 + (2*4+2) + (4+2+3)
    assert summary["optimizer_budget_units"] == 10

    progress_total = sum(
        int(message.get("delta") or 0)
        for message in progress_messages
        if message.get("type") == "progress"
    )
    # No optimizer budget spent by _NoopSession; only external pre-test + post train/val/test.
    assert progress_total == 12  # 3 + (4+2+3)


def test_live_local_import_guard_rejects_non_repo_root(tmp_path: Path):
    with pytest.raises(RuntimeError, match="live local repo package"):
        ensure_live_local_package(tmp_path)
