from __future__ import annotations

from pathlib import Path

import dspy
import pytest

from rlmoptimizer.kernel import OptimizationKernel
from rlmoptimizer.types import InstructionUpdateError

from ._helpers import RuleProgram, build_trainset, exact_metric


def _build_kernel(tmp_path: Path) -> OptimizationKernel:
    return OptimizationKernel(
        program=RuleProgram(),
        trainset=build_trainset(3),
        valset=None,
        metric=exact_metric,
        eval_lm=None,
        num_threads=1,
        max_iterations=3,
        max_output_chars=20_000,
        run_storage_dir=tmp_path / "runs",
    )


def test_update_instruction_rejects_unknown_predictor(tmp_path: Path):
    kernel = _build_kernel(tmp_path)
    kernel.run_baseline()

    with pytest.raises(InstructionUpdateError):
        kernel.update_instruction("missing", "New instruction")

    kernel.close()


def test_update_instruction_allows_noop(tmp_path: Path):
    kernel = _build_kernel(tmp_path)
    kernel.run_baseline()

    current = kernel.state.current_instruction_map["step"]
    result = kernel.update_instruction("step", current)

    assert result["status"] == "ok"
    assert kernel.state.current_instruction_map["step"] == current
    kernel.close()


def test_update_instruction_rejects_structure_changes(tmp_path: Path, monkeypatch):
    kernel = _build_kernel(tmp_path)
    kernel.run_baseline()

    predictor = dict(kernel.program.named_predictors())["step"]
    original_signature = predictor.signature
    mutated_signature = dspy.Signature("question -> answer, extra", "bad")

    def bad_with_instructions(_text: str):
        return mutated_signature

    monkeypatch.setattr(predictor.signature, "with_instructions", bad_with_instructions)

    with pytest.raises(InstructionUpdateError):
        kernel.update_instruction("step", "Copy question exactly")

    assert predictor.signature is original_signature
    kernel.close()
