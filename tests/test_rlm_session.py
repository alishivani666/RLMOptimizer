from __future__ import annotations

from pathlib import Path
from typing import Any

import dspy
from dspy.primitives.repl_types import REPLHistory

from rlmoptimizer.interpreter import ReRegisteringPythonInterpreter
from rlmoptimizer.rlm_session import RLMSession, _RLMMultiTurnHistoryAdapter
from rlmoptimizer.kernel import OptimizationKernel
from rlmoptimizer.tools import OptimizationTools

from ._helpers import RuleProgram, build_trainset, exact_metric


class _FakeTools:
    def as_dspy_tools(self) -> list[object]:
        return []


def _action_signature_for_tests() -> type[dspy.Signature]:
    action_sig, _ = dspy.RLM("question -> answer", max_iterations=4)._build_signatures()
    return action_sig


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


def test_multiturn_adapter_formats_history_as_alternating_chat_turns():
    adapter = _RLMMultiTurnHistoryAdapter()
    signature = _action_signature_for_tests()
    history = (
        REPLHistory()
        .append(reasoning="inspect variables", code="print('a')", output="A")
        .append(reasoning="check budget", code="print('b')", output="B")
    )

    messages = adapter.format(
        signature=signature,
        demos=[],
        inputs={
            "variables_info": "vars",
            "repl_history": history,
            "iteration": "3/4",
        },
    )

    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert "[[ ## repl_history ## ]]" not in str(messages[-1]["content"])
    assert "Execution result from your previous code" not in str(messages[1]["content"])
    assert "Execution result from your previous code" in str(messages[3]["content"])
    assert "A" in str(messages[3]["content"])
    assert "B" in str(messages[-1]["content"])
    assert "[[ ## reasoning ## ]]" in str(messages[2]["content"])
    assert "[[ ## code ## ]]" in str(messages[2]["content"])
    assert "```python\nprint('a')\n```" in str(messages[2]["content"])


def test_multiturn_adapter_keeps_extract_signature_format_unchanged():
    adapter = _RLMMultiTurnHistoryAdapter()
    _, extract_sig = dspy.RLM("question -> answer", max_iterations=4)._build_signatures()
    history = REPLHistory().append(
        reasoning="inspect variables",
        code="print('a')",
        output="A",
    )

    messages = adapter.format(
        signature=extract_sig,
        demos=[],
        inputs={
            "variables_info": "vars",
            "repl_history": history,
        },
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "[[ ## repl_history ## ]]" in str(messages[1]["content"])


def test_run_uses_multiturn_adapter_when_enabled(tmp_path: Path):
    captured: dict[str, Any] = {}

    def _factory(**_kwargs):
        class Agent:
            def __call__(self, **_inputs):
                captured["adapter"] = dspy.settings.get("adapter")
                return dspy.Prediction(
                    optimized_dspy_program={"step": "Copy question exactly."},
                    trajectory=[],
                    final_reasoning="ok",
                )

        return Agent()

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
    kernel.run_baseline()

    session = RLMSession(
        root_lm=object(),
        sub_lm=None,
        max_iterations=3,
        max_llm_calls=10,
        max_output_chars=5_000,
        verbose=False,
        rlm_factory=_factory,
        rlm_multiturn_history=True,
    )
    _ = session.run(kernel)

    assert isinstance(captured.get("adapter"), _RLMMultiTurnHistoryAdapter)
    kernel.close()
