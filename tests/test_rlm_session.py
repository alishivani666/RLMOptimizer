from __future__ import annotations

from pathlib import Path
from typing import Any

import dspy

from rlmoptimizer.interpreter import ReRegisteringPythonInterpreter
from rlmoptimizer.rlm_session import RLMSession
from rlmoptimizer.kernel import OptimizationKernel
from rlmoptimizer.tools import OptimizationTools

from ._helpers import RuleProgram, build_trainset, exact_metric


class _FakeTools:
    def as_dspy_tools(self) -> list[object]:
        return []


def test_build_rlm_disables_dspy_verbose_logs_when_debug_display_active():
    captured: dict[str, object] = {}

    def _factory(**kwargs):
        captured.update(kwargs)
        return object()

    session = RLMSession(
        root_lm=object(),
        sub_lm=None,
        max_iterations=3,
        max_llm_calls=10,
        max_output_chars=5_000,
        verbose=True,
        rlm_factory=_factory,
        debug_display=object(),
    )

    _ = session._build_rlm(_FakeTools(), sub_lm=None)

    assert captured["verbose"] is False


def test_build_rlm_keeps_dspy_verbose_logs_without_debug_display():
    captured: dict[str, object] = {}

    def _factory(**kwargs):
        captured.update(kwargs)
        return object()

    session = RLMSession(
        root_lm=object(),
        sub_lm=None,
        max_iterations=3,
        max_llm_calls=10,
        max_output_chars=5_000,
        verbose=True,
        rlm_factory=_factory,
        debug_display=None,
    )

    _ = session._build_rlm(_FakeTools(), sub_lm=None)

    assert captured["verbose"] is True


def test_build_rlm_passes_expected_tool_names(tmp_path: Path):
    captured: dict[str, Any] = {}

    def _factory(**kwargs):
        captured.update(kwargs)
        return object()

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
    )

    session = RLMSession(
        root_lm=object(),
        sub_lm=None,
        max_iterations=10,
        max_llm_calls=10,
        max_output_chars=10_000,
        verbose=False,
        rlm_factory=_factory,
    )
    _ = session._build_rlm(OptimizationTools(kernel), sub_lm=None)

    tools = captured["tools"]
    assert isinstance(tools, list)
    assert [tool.name for tool in tools] == [
        "evaluate_program",
        "run_data",
        "update_prompt",
        "optimization_status",
    ]

    kernel.close()


def test_build_rlm_uses_local_resilient_interpreter():
    session = RLMSession(
        root_lm=object(),
        sub_lm=None,
        max_iterations=3,
        max_llm_calls=10,
        max_output_chars=5_000,
        verbose=False,
    )

    rlm = session._build_rlm(_FakeTools(), sub_lm=None)

    assert isinstance(rlm, dspy.RLM)
    assert isinstance(rlm._interpreter, ReRegisteringPythonInterpreter)
