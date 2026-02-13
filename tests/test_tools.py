from __future__ import annotations

from pathlib import Path

import pytest

from rlmoptimizer.kernel import OptimizationKernel
from rlmoptimizer.tools import OptimizationTools
from rlmoptimizer.types import BudgetExceededError

from ._helpers import RuleProgram, build_trainset, exact_metric


def _contains_none(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, dict):
        return any(_contains_none(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_none(item) for item in value)
    return False


def test_evaluate_and_run_data_round_trip(tmp_path: Path):
    kernel = OptimizationKernel(
        program=RuleProgram(),
        trainset=build_trainset(4),
        valset=None,
        metric=exact_metric,
        eval_lm=None,
        num_threads=1,
        max_iterations=3,
        max_output_chars=20_000,
        run_storage_dir=tmp_path / "runs",
    )

    baseline = kernel.run_baseline()
    run_id = baseline["run_id"]

    loaded = kernel.run_data(run_id)
    assert loaded["run_id"] == run_id
    assert loaded["evaluated_count"] == 4
    assert "examples" in loaded
    assert "remaining_budget" not in loaded
    assert "Budget remaining:" not in loaded["summary_line"]
    assert "Run:" not in loaded["summary_line"]

    status = kernel.optimization_status()
    assert status["baseline_run_id"] == run_id
    assert status["remaining_budget"] == 8

    kernel.close()


def test_tool_payload_returns_structured_dict(tmp_path: Path):
    kernel = OptimizationKernel(
        program=RuleProgram(),
        trainset=build_trainset(6),
        valset=None,
        metric=exact_metric,
        eval_lm=None,
        num_threads=1,
        max_iterations=3,
        max_output_chars=1_200,
        run_storage_dir=tmp_path / "runs",
    )

    payload = kernel.evaluate_program(split="train")
    assert isinstance(payload, dict)
    assert payload["evaluated_count"] == 6
    assert isinstance(payload["examples"], list)

    kernel.close()


def test_tools_normalize_inputs_and_return_errors(tmp_path: Path):
    kernel = OptimizationKernel(
        program=RuleProgram(),
        trainset=build_trainset(6),
        valset=None,
        metric=exact_metric,
        eval_lm=None,
        num_threads=1,
        max_iterations=4,
        max_output_chars=10_000,
        run_storage_dir=tmp_path / "runs",
    )
    tools = OptimizationTools(kernel)

    payload = tools.evaluate_program(
        split="train",
        limit="2",
        ids=[1, 2, "3"],
        sample=None,
        sample_seed="0",
    )
    assert isinstance(payload, dict)
    assert payload["evaluated_count"] == 2

    run_id = payload["run_id"]
    loaded = tools.run_data(run_id)
    assert isinstance(loaded, dict)
    assert loaded["run_id"] == run_id
    assert "remaining_budget" not in loaded
    assert "Budget remaining:" not in loaded["summary_line"]
    assert "Run:" not in loaded["summary_line"]

    bad_limit = tools.evaluate_program(limit="not-an-int")
    assert isinstance(bad_limit, dict)
    assert "error" in bad_limit

    bad_ids = tools.evaluate_program(ids={"bad": "shape"})  # type: ignore[arg-type]
    assert isinstance(bad_ids, dict)
    assert "error" in bad_ids

    bad_step = tools.update_prompt("missing", "Copy question exactly.")
    assert "error" in bad_step

    status = tools.optimization_status()
    assert "steps" in status
    assert "current_prompts" in status
    assert "best_prompts" in status

    bad_run = tools.run_data("does-not-exist")
    assert isinstance(bad_run, dict)
    assert "error" in bad_run

    bad_int_run = tools.run_data(5)
    assert isinstance(bad_int_run, dict)
    assert "error" in bad_int_run

    kernel.close()


def test_tools_budget_exhaustion_still_raises(tmp_path: Path):
    kernel = OptimizationKernel(
        program=RuleProgram(),
        trainset=build_trainset(3),
        valset=None,
        metric=exact_metric,
        eval_lm=None,
        num_threads=1,
        max_iterations=1,
        max_output_chars=10_000,
        run_storage_dir=tmp_path / "runs",
    )
    tools = OptimizationTools(kernel)

    _ = tools.evaluate_program(limit=3)
    with pytest.raises(BudgetExceededError):
        tools.evaluate_program(limit=1)

    kernel.close()


def test_update_prompt_prints_readable_success_message(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    kernel = OptimizationKernel(
        program=RuleProgram(),
        trainset=build_trainset(3),
        valset=None,
        metric=exact_metric,
        eval_lm=None,
        num_threads=1,
        max_iterations=3,
        max_output_chars=10_000,
        run_storage_dir=tmp_path / "runs",
    )
    tools = OptimizationTools(kernel)

    response = tools.update_prompt("step", "Copy question exactly.")
    print(response)
    printed = capsys.readouterr().out.strip()

    assert printed == 'Prompt for "step" was successfully updated.'
    assert response["status"] == "ok"
    assert response["step_name"] == "step"
    assert "prompt_hash" in response

    kernel.close()


def test_tools_payloads_are_sanitized_for_none_values(tmp_path: Path):
    kernel = OptimizationKernel(
        program=RuleProgram(),
        trainset=build_trainset(3),
        valset=None,
        metric=exact_metric,
        eval_lm=None,
        num_threads=1,
        max_iterations=3,
        max_output_chars=10_000,
        run_storage_dir=tmp_path / "runs",
    )
    tools = OptimizationTools(kernel)

    payload = tools.evaluate_program(limit=2)
    loaded = tools.run_data(payload["run_id"])
    status = tools.optimization_status()

    assert _contains_none(payload) is False
    assert _contains_none(loaded) is False
    assert _contains_none(status) is False

    kernel.close()


def test_tools_payloads_are_sanitized_for_non_finite_floats(tmp_path: Path):
    kernel = OptimizationKernel(
        program=RuleProgram(),
        trainset=build_trainset(3),
        valset=None,
        metric=exact_metric,
        eval_lm=None,
        num_threads=1,
        max_iterations=3,
        max_output_chars=10_000,
        run_storage_dir=tmp_path / "runs",
    )
    tools = OptimizationTools(kernel)

    kernel.state.best_score = float("-inf")
    assert tools.optimization_status()["best_score"] == "-inf"

    kernel.state.best_score = float("inf")
    assert tools.optimization_status()["best_score"] == "inf"

    kernel.state.best_score = float("nan")
    assert tools.optimization_status()["best_score"] == "nan"

    kernel.close()


def test_tools_allow_round_trip_with_sanitized_optional_config_values(tmp_path: Path):
    kernel = OptimizationKernel(
        program=RuleProgram(),
        trainset=build_trainset(5),
        valset=None,
        metric=exact_metric,
        eval_lm=None,
        num_threads=1,
        max_iterations=4,
        max_output_chars=10_000,
        run_storage_dir=tmp_path / "runs",
    )
    tools = OptimizationTools(kernel)

    payload = tools.evaluate_program(limit=2)
    config = payload["config"]

    replay = tools.evaluate_program(
        split=config["split"],
        limit=config["limit"],
        ids=config["ids"],
        sample=config["sample"],
        sample_seed=config["sample_seed"],
        failed_from_run=config["failed_from_run"],
    )

    assert isinstance(replay, dict)
    assert "error" not in replay
    assert replay["evaluated_count"] == 2

    kernel.close()
