from __future__ import annotations

from pathlib import Path
from typing import Any

import dspy
from dspy.primitives.repl_types import REPLHistory
from litellm.exceptions import ContextWindowExceededError
import pytest

from rlmoptimizer.interpreter import ReRegisteringPythonInterpreter
from rlmoptimizer.rlm_session import (
    RLMSession,
    _RLMMultiTurnHistoryAdapter,
    _RLMMultiTurnJSONFallbackAdapter,
)
from rlmoptimizer.kernel import OptimizationKernel
from rlmoptimizer.tools import OptimizationTools

from ._helpers import RuleProgram, build_trainset, exact_metric


class _FakeTools:
    def as_dspy_tools(self) -> list[object]:
        return []


class _RootLMStub:
    def __init__(self, model: str) -> None:
        self.model = model


def _action_signature_for_tests() -> type[dspy.Signature]:
    action_sig, _ = dspy.RLM("question -> answer", max_iterations=4)._build_signatures()
    return action_sig


def test_multiturn_adapter_keeps_json_adapter_fallback_enabled():
    adapter = _RLMMultiTurnHistoryAdapter()
    assert adapter.use_json_adapter_fallback is True


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
    assert "[[ ## variables_info ## ]]" in str(messages[1]["content"])
    assert "[[ ## variables_info ## ]]" not in str(messages[3]["content"])
    assert "[[ ## variables_info ## ]]" not in str(messages[-1]["content"])
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


def test_multiturn_adapter_keeps_variables_info_on_first_turn_only():
    adapter = _RLMMultiTurnHistoryAdapter()
    signature = _action_signature_for_tests()

    first_turn_messages = adapter.format(
        signature=signature,
        demos=[],
        inputs={
            "variables_info": "vars",
            "repl_history": REPLHistory(),
            "iteration": "1/4",
        },
    )
    assert "[[ ## variables_info ## ]]" in str(first_turn_messages[-1]["content"])

    second_turn_messages = adapter.format(
        signature=signature,
        demos=[],
        inputs={
            "variables_info": "vars",
            "repl_history": REPLHistory().append(
                reasoning="inspect variables",
                code="print('a')",
                output="A",
            ),
            "iteration": "2/4",
        },
    )
    assert "[[ ## variables_info ## ]]" in str(second_turn_messages[1]["content"])
    assert "[[ ## variables_info ## ]]" not in str(second_turn_messages[-1]["content"])


def test_multiturn_adapter_json_fallback_preserves_multiturn_input(monkeypatch):
    class _ProbeLM:
        def __init__(self) -> None:
            self.model = "openai/mock-model"
            self.previous_response_id = "resp_10"
            self.calls: list[dict[str, Any]] = []

        def __call__(self, *, messages: list[dict[str, Any]], **kwargs: Any):
            call_kwargs = dict(kwargs)
            if (
                "previous_response_id" not in call_kwargs
                and self.previous_response_id is not None
            ):
                call_kwargs["previous_response_id"] = self.previous_response_id
            self.calls.append({"messages": list(messages), "kwargs": call_kwargs})
            if len(self.calls) == 1:
                self.previous_response_id = "resp_11_bad"
                return ["[[ ## reasoning ## ]]\nonly reasoning\n\n[[ ## completed ## ]]"]
            self.previous_response_id = "resp_11_json_ok"
            return ['{"reasoning": "fallback reasoning", "code": "```python\\npass\\n```"}']

    def _call_without_litellm_capability_checks(
        self: _RLMMultiTurnJSONFallbackAdapter,
        lm: Any,
        lm_kwargs: dict[str, Any],
        signature: type[dspy.Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
        call_fn: Any,
    ):
        return call_fn(lm, lm_kwargs, signature, demos, inputs)

    monkeypatch.setattr(
        _RLMMultiTurnJSONFallbackAdapter,
        "_json_adapter_call_common",
        _call_without_litellm_capability_checks,
    )

    adapter = _RLMMultiTurnHistoryAdapter()
    signature = _action_signature_for_tests()
    history = (
        REPLHistory()
        .append(reasoning="inspect variables", code="print('a')", output="A")
        .append(reasoning="check budget", code="print('b')", output="B")
    )
    lm = _ProbeLM()

    result = adapter(
        lm=lm,  # type: ignore[arg-type]
        lm_kwargs={},
        signature=signature,
        demos=[],
        inputs={
            "variables_info": "vars",
            "repl_history": history,
            "iteration": "3/4",
        },
    )

    assert result[0]["reasoning"] == "fallback reasoning"
    assert len(lm.calls) == 2
    assert lm.calls[0]["kwargs"]["previous_response_id"] == "resp_10"
    assert lm.calls[1]["kwargs"]["previous_response_id"] == "resp_10"
    assert lm.previous_response_id == "resp_11_json_ok"

    fallback_messages = lm.calls[1]["messages"]
    assert [message["role"] for message in fallback_messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert all(
        "[[ ## repl_history ## ]]" not in str(message["content"])
        for message in fallback_messages
        if message["role"] == "user"
    )
    assert "[[ ## variables_info ## ]]" in str(fallback_messages[1]["content"])
    assert "[[ ## variables_info ## ]]" not in str(fallback_messages[3]["content"])
    assert "[[ ## variables_info ## ]]" not in str(fallback_messages[-1]["content"])


def test_multiturn_adapter_does_not_json_fallback_on_context_window_error(monkeypatch):
    adapter = _RLMMultiTurnHistoryAdapter()
    signature = _action_signature_for_tests()
    context_error = ContextWindowExceededError(
        message="context window exceeded",
        model="openai/mock-model",
        llm_provider="openai",
    )
    fallback_called = {"value": False}

    def _raise_primary(
        lm: dspy.LM,
        lm_kwargs: dict[str, Any],
        signature: type[dspy.Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        del lm, lm_kwargs, signature, demos, inputs
        raise context_error

    def _unexpected_fallback(
        lm: dspy.LM,
        lm_kwargs: dict[str, Any],
        signature: type[dspy.Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        del lm, lm_kwargs, signature, demos, inputs
        fallback_called["value"] = True
        return []

    monkeypatch.setattr(adapter, "_call_primary", _raise_primary)
    monkeypatch.setattr(adapter, "_call_json_fallback", _unexpected_fallback)

    with pytest.raises(ContextWindowExceededError):
        _ = adapter(
            lm=object(),  # type: ignore[arg-type]
            lm_kwargs={},
            signature=signature,
            demos=[],
            inputs={
                "variables_info": "vars",
                "repl_history": REPLHistory(),
                "iteration": "1/4",
            },
        )

    assert fallback_called["value"] is False


def _run_and_capture_adapter(
    *,
    tmp_path: Path,
    root_lm: Any,
) -> Any:
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
        root_lm=root_lm,
        sub_lm=None,
        max_iterations=3,
        max_llm_calls=10,
        max_output_chars=5_000,
        verbose=False,
        rlm_factory=_factory,
        rlm_multiturn_history=True,
    )
    _ = session.run(kernel)
    kernel.close()
    return captured.get("adapter")


def test_run_uses_multiturn_chat_adapter_for_non_gpt5_root_model(tmp_path: Path):
    adapter = _run_and_capture_adapter(
        tmp_path=tmp_path,
        root_lm=_RootLMStub("openai/gpt-4.1-mini"),
    )
    assert isinstance(adapter, _RLMMultiTurnHistoryAdapter)


def test_run_uses_multiturn_json_adapter_for_gpt5_root_model(tmp_path: Path):
    adapter = _run_and_capture_adapter(
        tmp_path=tmp_path,
        root_lm=_RootLMStub("openai/gpt-5.2-codex"),
    )
    assert isinstance(adapter, _RLMMultiTurnJSONFallbackAdapter)
