from __future__ import annotations

from pathlib import Path

from rlmoptimizer.kernel import OptimizationKernel

from ._helpers import RuleProgram, build_trainset, exact_metric


def test_run_baseline_keeps_latest_run_on_train_when_valset_exists(tmp_path: Path):
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
    assert baseline["split"] == "val"
    assert kernel.baseline_train_run_id is not None
    assert kernel.baseline_val_run_id == baseline["run_id"]
    assert kernel.state.latest_run_id == kernel.baseline_train_run_id
    assert baseline["examples"] == []

    kernel.close()


def test_apply_submitted_prompt_map_ignores_unknown_steps(tmp_path: Path):
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

    before = dict(kernel.state.current_prompt_map)
    updated = kernel.apply_submitted_prompt_map({"unknown": "x"})
    assert updated == before
    assert kernel.state.current_prompt_map == before

    kernel.close()


def test_apply_submitted_prompt_map_rejects_non_dict(tmp_path: Path):
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

    try:
        kernel.apply_submitted_prompt_map("bad")  # type: ignore[arg-type]
        assert False, "expected TypeError for non-dict submission"
    except TypeError:
        pass

    kernel.close()


def test_apply_submitted_prompt_map_updates_current_prompt_map(tmp_path: Path):
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

    updated = kernel.apply_submitted_prompt_map({"step": "Copy question exactly."})
    assert updated == {"step": "Copy question exactly."}
    assert kernel.state.current_prompt_map == {"step": "Copy question exactly."}

    kernel.close()
