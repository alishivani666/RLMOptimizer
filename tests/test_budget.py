from __future__ import annotations

from pathlib import Path

from rlmoptimizer.kernel import OptimizationKernel

from ._helpers import RuleProgram, build_trainset, exact_metric


def test_budget_charged_by_actual_evaluated_count(tmp_path: Path):
    kernel = OptimizationKernel(
        program=RuleProgram(),
        trainset=build_trainset(5),
        valset=build_trainset(3),
        metric=exact_metric,
        eval_lm=None,
        num_threads=1,
        max_iterations=3,
        max_output_chars=20_000,
        run_storage_dir=tmp_path / "runs",
    )

    baseline = kernel.run_baseline()
    assert baseline["split"] == "val"
    assert baseline["evaluated_count"] == 3
    assert kernel.state.remaining_budget == 18

    targeted = kernel.evaluate_program(ids="1,3,5")
    assert targeted["evaluated_count"] == 3
    assert kernel.state.remaining_budget == 15

    val_eval = kernel.evaluate_program(split="val")
    assert val_eval["evaluated_count"] == 3
    assert val_eval["examples"] == []
    assert kernel.state.remaining_budget == 12

    kernel.close()
