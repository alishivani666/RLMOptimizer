from __future__ import annotations

from pathlib import Path

from rlmoptimizer.kernel import OptimizationKernel

from ._helpers import RuleProgram, build_trainset, exact_metric


def test_best_tracking_prefers_full_val_runs_when_valset_exists(tmp_path: Path):
    kernel = OptimizationKernel(
        program=RuleProgram(),
        trainset=build_trainset(4),
        valset=build_trainset(2),
        metric=exact_metric,
        eval_lm=None,
        num_threads=1,
        max_iterations=5,
        max_output_chars=20_000,
        run_storage_dir=tmp_path / "runs",
    )

    baseline = kernel.run_baseline()
    assert baseline["split"] == "train"
    assert kernel.state.best_run_id is None
    assert kernel.state.best_score == float("-inf")

    kernel.update_instruction("step", "Copy question exactly.")

    full_train = kernel.evaluate_program(split="train")
    assert full_train["score"] == 100.0
    assert kernel.state.best_run_id is None

    val_ids_all = kernel.evaluate_program(split="val", ids="1,2")
    assert val_ids_all["score"] == 100.0
    assert val_ids_all["evaluated_count"] == 2
    assert kernel.state.best_run_id is None

    full_val = kernel.evaluate_program(split="val", limit=2)
    assert full_val["score"] == 100.0
    assert full_val["evaluated_count"] == 2
    assert kernel.state.best_run_id == full_val["run_id"]
    assert kernel.state.best_score == 100.0

    kernel.close()


def test_best_tracking_uses_full_train_when_no_valset(tmp_path: Path):
    kernel = OptimizationKernel(
        program=RuleProgram(),
        trainset=build_trainset(4),
        valset=None,
        metric=exact_metric,
        eval_lm=None,
        num_threads=1,
        max_iterations=5,
        max_output_chars=20_000,
        run_storage_dir=tmp_path / "runs",
    )

    baseline = kernel.run_baseline()
    assert baseline["split"] == "train"
    assert kernel.state.best_run_id == baseline["run_id"]
    assert kernel.state.best_score == 0.0

    kernel.update_instruction("step", "Copy question exactly.")

    train_subset = kernel.evaluate_program(split="train", limit=1)
    assert train_subset["score"] == 100.0
    assert kernel.state.best_run_id == baseline["run_id"]
    assert kernel.state.best_score == 0.0

    full_train = kernel.evaluate_program(split="train", limit=4)
    assert full_train["score"] == 100.0
    assert kernel.state.best_run_id == full_train["run_id"]
    assert kernel.state.best_score == 100.0

    kernel.close()
