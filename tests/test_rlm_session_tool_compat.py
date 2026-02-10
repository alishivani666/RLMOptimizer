from __future__ import annotations

from pathlib import Path
from typing import Any

import dspy

from rlmoptimizer.kernel import OptimizationKernel
from rlmoptimizer.rlm_session import RLMSession
from rlmoptimizer.tools import OptimizationTools

from ._helpers import RuleProgram, build_trainset, exact_metric


def _build_kernel(tmp_path: Path) -> OptimizationKernel:
    return OptimizationKernel(
        program=RuleProgram(),
        trainset=build_trainset(3),
        valset=None,
        metric=exact_metric,
        eval_lm=None,
        num_threads=1,
        max_iterations=2,
        max_output_chars=20_000,
        run_storage_dir=tmp_path / "runs",
    )


def test_rlm_session_passes_list_tools(tmp_path: Path):
    captured: dict[str, Any] = {}

    def list_factory(
        *,
        signature: Any,
        max_iterations: int,
        max_llm_calls: int,
        max_output_chars: int,
        verbose: bool,
        tools: list[dspy.Tool] | None = None,
        sub_lm: Any = None,
    ) -> object:
        del signature, max_iterations, max_llm_calls, max_output_chars, verbose, sub_lm
        captured["tools"] = tools
        return object()

    kernel = _build_kernel(tmp_path)
    session = RLMSession(
        root_lm=object(),
        sub_lm=None,
        max_iterations=10,
        max_llm_calls=10,
        max_output_chars=10_000,
        verbose=False,
        rlm_factory=list_factory,
    )
    _ = session._build_rlm(OptimizationTools(kernel), sub_lm=None)

    tools = captured["tools"]
    assert isinstance(tools, list)
    assert [tool.name for tool in tools] == [
        "evaluate_program",
        "run_data",
        "update_instruction",
        "optimization_status",
    ]
    kernel.close()
