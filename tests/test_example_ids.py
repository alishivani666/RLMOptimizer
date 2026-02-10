from __future__ import annotations

from pathlib import Path

import dspy

from rlmoptimizer.kernel import OptimizationKernel


_NATIVE_ID_1 = "5a7e14725542995f4f4023d4"
_NATIVE_ID_2 = "5a7e14725542995f4f4023d5"


class _EchoProgram(dspy.Module):
    def forward(self, question: str) -> dspy.Prediction:
        return dspy.Prediction(answer=question)


def _exact_metric(example: dspy.Example, pred: dspy.Prediction, _trace=None) -> float:
    return float(example.answer == pred.answer)


def _build_trainset_with_native_ids() -> list[dspy.Example]:
    return [
        dspy.Example(
            id=_NATIVE_ID_1,
            question="q1",
            answer="q1",
        ).with_inputs("question"),
        dspy.Example(
            id=_NATIVE_ID_2,
            question="q2",
            answer="wrong",
        ).with_inputs("question"),
        dspy.Example(
            question="q3",
            answer="q3",
        ).with_inputs("question"),
        dspy.Example(
            example_id="dataset-4",
            question="q4",
            answer="wrong",
        ).with_inputs("question"),
    ]


def _build_kernel(tmp_path: Path) -> OptimizationKernel:
    return OptimizationKernel(
        program=_EchoProgram(),
        trainset=_build_trainset_with_native_ids(),
        valset=None,
        metric=_exact_metric,
        eval_lm=None,
        num_threads=1,
        max_iterations=5,
        max_output_chars=50_000,
        run_storage_dir=tmp_path / "runs",
    )


def test_run_outputs_use_numeric_example_ids_and_preserve_source_ids(tmp_path: Path):
    kernel = _build_kernel(tmp_path)

    baseline = kernel.run_baseline()
    assert [item["example_id"] for item in baseline["examples"]] == ["1", "2", "3", "4"]
    assert [item["source_example_id"] for item in baseline["examples"]] == [
        _NATIVE_ID_1,
        _NATIVE_ID_2,
        None,
        "dataset-4",
    ]

    loaded = kernel.run_data(baseline["run_id"])
    assert [item["example_id"] for item in loaded["examples"]] == ["1", "2", "3", "4"]
    assert [item["source_example_id"] for item in loaded["examples"]] == [
        _NATIVE_ID_1,
        _NATIVE_ID_2,
        None,
        "dataset-4",
    ]

    kernel.close()


def test_id_filters_and_ranges_apply_to_numeric_example_ids(tmp_path: Path):
    kernel = _build_kernel(tmp_path)
    kernel.run_baseline()

    selected = kernel.evaluate_program(ids="1,3")
    assert [item["example_id"] for item in selected["examples"]] == ["1", "3"]
    assert [item["source_example_id"] for item in selected["examples"]] == [
        _NATIVE_ID_1,
        None,
    ]

    ranged = kernel.evaluate_program(ids="2-4")
    assert [item["example_id"] for item in ranged["examples"]] == ["2", "3", "4"]
    assert [item["source_example_id"] for item in ranged["examples"]] == [
        _NATIVE_ID_2,
        None,
        "dataset-4",
    ]

    kernel.close()


def test_failed_from_run_uses_numeric_example_ids(tmp_path: Path):
    kernel = _build_kernel(tmp_path)

    baseline = kernel.run_baseline()
    failed_only = kernel.evaluate_program(failed_from_run=baseline["run_id"])

    assert [item["example_id"] for item in failed_only["examples"]] == ["2", "4"]
    assert [item["source_example_id"] for item in failed_only["examples"]] == [
        _NATIVE_ID_2,
        "dataset-4",
    ]

    kernel.close()
