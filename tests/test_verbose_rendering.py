from __future__ import annotations

from types import SimpleNamespace

import dspy

from rlmoptimizer.rlm_session import _InstrumentedRLM


class _FakeHistoryResult:
    def __init__(self, output: str) -> None:
        self.entries = [SimpleNamespace(output=output)]


class _FakeRepl:
    def __init__(self, events: list[tuple[str, str]]) -> None:
        self._events = events

    def execute(self, code: str, variables: dict[str, str]):
        del variables
        self._events.append(("execute", code))
        return "ok"


def _build_instrumented_rlm(
    *,
    on_start,
    on_output,
) -> _InstrumentedRLM:
    return _InstrumentedRLM(
        signature="question -> answer",
        max_iterations=3,
        iteration_start_callback=on_start,
        iteration_output_callback=on_output,
    )


def test_iteration_rendering_is_causal_order():
    events: list[tuple[str, str]] = []
    start_codes: list[str] = []
    output_values: list[str] = []

    rlm = _build_instrumented_rlm(
        on_start=lambda _i, _max, _reasoning, code: (
            events.append(("start", code)),
            start_codes.append(code),
        ),
        on_output=lambda output: (
            events.append(("output", output)),
            output_values.append(output),
        ),
    )
    rlm.generate_action = lambda **_kwargs: SimpleNamespace(
        reasoning="evaluate now",
        code="```python\nprint('go')\n```",
    )
    rlm._process_execution_result = (
        lambda _action, _execution_result, _history, _output_field_names: _FakeHistoryResult(
            "iteration-output"
        )
    )

    result = rlm._execute_iteration(
        repl=_FakeRepl(events=events),
        variables=[SimpleNamespace(format=lambda: "question: str")],
        history=object(),
        iteration=0,
        input_args={"question": "q1"},
        output_field_names=["answer"],
    )

    assert isinstance(result, _FakeHistoryResult)
    assert [event[0] for event in events] == ["start", "execute", "output"]
    assert start_codes == ["print('go')"]
    assert output_values == ["iteration-output"]


def test_output_callback_uses_prediction_trajectory_output():
    captured_outputs: list[str] = []

    rlm = _build_instrumented_rlm(
        on_start=lambda *_args: None,
        on_output=lambda output: captured_outputs.append(output),
    )
    rlm.generate_action = lambda **_kwargs: SimpleNamespace(
        reasoning="done",
        code="print('final')",
    )
    rlm._process_execution_result = lambda _action, _execution_result, _history, _output_field_names: dspy.Prediction(
        answer="value",
        trajectory=[{"output": "final-output"}],
    )

    result = rlm._execute_iteration(
        repl=_FakeRepl(events=[]),
        variables=[SimpleNamespace(format=lambda: "question: str")],
        history=object(),
        iteration=1,
        input_args={"question": "q2"},
        output_field_names=["answer"],
    )

    assert isinstance(result, dspy.Prediction)
    assert captured_outputs == ["final-output"]
