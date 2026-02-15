from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import dspy

from rlmoptimizer.budgeting import BudgetMeteredLM
from rlmoptimizer.kernel import OptimizationKernel
from rlmoptimizer.rlm_session import RLMSession
from rlmoptimizer.types import BudgetExceededError

from ._helpers import RuleProgram, build_trainset, exact_metric


class DummyLM:
    def __init__(self, name: str) -> None:
        self.model = name

    def __call__(self, *_args, **_kwargs):
        return ["ok"]

    async def acall(self, *_args, **_kwargs):
        return ["ok"]

    def copy(self, **_kwargs):
        return DummyLM(self.model)


class _NoopBudgetConsumer:
    def charge_llm_requests(self, *, source: str, requests: int = 1) -> None:
        del source, requests


def _response_payload(*, model: str, response_id: str):
    return SimpleNamespace(
        id=response_id,
        model=model,
        usage={},
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(text=response_id)],
            )
        ],
    )


class _RecordedResponsesLM:
    def __init__(
        self,
        name: str,
        *,
        model_type: str = "responses",
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.model = name
        self.model_type = model_type
        self.cache = False
        self.kwargs = dict(kwargs or {})
        self.history: list[dict[str, Any]] = []
        self.call_kwargs: list[dict[str, Any]] = []
        self.forward_kwargs: list[dict[str, Any]] = []
        self._next_response = 1

    def __call__(self, *_args: Any, **kwargs: Any):
        self.call_kwargs.append(dict(kwargs))
        return ["ok"]

    async def acall(self, *_args: Any, **kwargs: Any):
        return self.__call__(*_args, **kwargs)

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ):
        del prompt, messages
        self.forward_kwargs.append(dict(kwargs))
        response = _response_payload(
            model=self.model,
            response_id=f"resp_{self._next_response}",
        )
        self._next_response += 1
        return response

    async def aforward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ):
        return self.forward(prompt=prompt, messages=messages, **kwargs)

    def copy(self, **kwargs: Any):
        copied = _RecordedResponsesLM(
            self.model,
            model_type=self.model_type,
            kwargs=self.kwargs,
        )
        copied._next_response = self._next_response
        if "model" in kwargs:
            copied.model = str(kwargs["model"])
        if "model_type" in kwargs:
            copied.model_type = str(kwargs["model_type"])
        for key, value in kwargs.items():
            if key in {"model", "model_type"}:
                continue
            if value is None:
                copied.kwargs.pop(key, None)
            else:
                copied.kwargs[key] = value
        return copied


def _budgeting_factory(**kwargs):
    sub_lm = kwargs.get("sub_lm")

    class Agent:
        def __call__(self, **_inputs):
            root_lm = dspy.settings.get("lm")
            root_lm("root-1")
            root_lm("root-2")
            if sub_lm is not None:
                sub_lm("sub-1")
            return dspy.Prediction(
                optimized_dspy_program={"step": "Return the answer exactly."},
                trajectory=[],
                final_reasoning="",
            )

    return Agent()


def _stateful_budgeting_factory(**kwargs):
    sub_lm = kwargs.get("sub_lm")

    class Agent:
        def __call__(self, **_inputs):
            root_lm = dspy.settings.get("lm")
            root_lm("root-1")
            root_lm("root-2")
            if sub_lm is not None:
                sub_lm("sub-1")
                sub_lm("sub-2")
            return dspy.Prediction(
                optimized_dspy_program={"step": "Return the answer exactly."},
                trajectory=[],
                final_reasoning="",
            )

    return Agent()


def _sub_overflow_factory(**kwargs):
    sub_lm = kwargs.get("sub_lm")

    class Agent:
        def __call__(self, **_inputs):
            root_lm = dspy.settings.get("lm")
            root_lm("root-1")
            root_lm("root-2")
            if sub_lm is not None:
                sub_lm("sub-1")
                sub_lm("sub-2")
                sub_lm("sub-3")
            return dspy.Prediction(
                optimized_dspy_program={"step": "Return the answer exactly."},
                trajectory=[],
                final_reasoning="",
            )

    return Agent()


def test_kernel_tracks_root_calls_but_charges_only_sub_lm_calls():
    kernel = OptimizationKernel(
        program=RuleProgram(),
        trainset=build_trainset(4),
        valset=None,
        metric=exact_metric,
        eval_lm=None,
        num_threads=1,
        max_iterations=2,
        max_output_chars=20_000,
    )
    kernel.run_baseline()

    session = RLMSession(
        root_lm=DummyLM("root"),
        sub_lm=DummyLM("sub"),
        max_iterations=10,
        max_llm_calls=10,
        max_output_chars=20_000,
        verbose=False,
        rlm_factory=_budgeting_factory,
    )
    result = session.run(kernel)

    assert result["optimized_dspy_program"] == {"step": "Return the answer exactly."}
    assert kernel.state.evaluated_examples == 0
    assert kernel.state.root_lm_calls == 2
    assert kernel.state.sub_lm_calls == 1
    assert kernel.state.remaining_budget == 7
    status = kernel.optimization_status()
    assert status["evaluated_examples"] == 0
    assert status["sub_lm_calls"] == 1
    assert "root_lm_calls" not in status

    kernel.close()


def test_kernel_budget_exceeded_by_sub_lm_calls():
    kernel = OptimizationKernel(
        program=RuleProgram(),
        trainset=build_trainset(2),
        valset=None,
        metric=exact_metric,
        eval_lm=None,
        num_threads=1,
        max_iterations=1,
        max_output_chars=20_000,
    )
    kernel.run_baseline()

    session = RLMSession(
        root_lm=DummyLM("root"),
        sub_lm=DummyLM("sub"),
        max_iterations=10,
        max_llm_calls=10,
        max_output_chars=20_000,
        verbose=False,
        rlm_factory=_sub_overflow_factory,
    )

    try:
        session.run(kernel)
        assert False, "expected BudgetExceededError"
    except BudgetExceededError:
        pass

    kernel.close()


def test_budget_metered_lm_is_dspy_baselm_compatible():
    wrapped = BudgetMeteredLM(
        lm=dspy.LM("openai/mock-root"),
        budget_consumer=_NoopBudgetConsumer(),
        source="root",
    )
    predictor = dspy.Predict("question -> answer")
    lm, *_ = predictor._forward_preprocess(question="hello", lm=wrapped)

    assert isinstance(wrapped, dspy.BaseLM)
    assert lm is wrapped


def test_rlm_session_baseline_summary_lists_baseline_inventory():
    captured_inputs: dict[str, Any] = {}

    def capture_factory(**_kwargs):
        class Agent:
            def __call__(self, **inputs):
                captured_inputs.update(inputs)
                return dspy.Prediction(
                    optimized_dspy_program={"step": "Return the answer exactly."},
                    trajectory=[],
                    final_reasoning="",
                )

        return Agent()

    kernel = OptimizationKernel(
        program=RuleProgram(),
        trainset=build_trainset(4),
        valset=None,
        metric=exact_metric,
        eval_lm=None,
        num_threads=1,
        max_iterations=2,
        max_output_chars=20_000,
    )
    baseline = kernel.run_baseline()
    baseline_run_id = str(baseline["run_id"])

    session = RLMSession(
        root_lm=DummyLM("root"),
        sub_lm=None,
        max_iterations=10,
        max_llm_calls=10,
        max_output_chars=20_000,
        verbose=False,
        rlm_factory=capture_factory,
    )
    _ = session.run(kernel)

    summary = str(captured_inputs["unoptimized_baseline_summary"])
    assert "You already have two baseline evaluation runs saved:" in summary
    assert "1) Baseline train run" in summary
    assert f"- run_id: {baseline_run_id}" in summary
    assert "- diagnostics: per-example data available via run_data(run_id)" in summary
    assert "2) Baseline validation run" in summary
    assert "- diagnostics: aggregate-only (examples are hidden)" in summary
    assert f"Current latest_run_id: {baseline_run_id}" in summary
    assert "Budget remaining" not in captured_inputs["unoptimized_baseline_summary"]
    assert captured_inputs["total_budget_remaining"] == kernel.state.remaining_budget
    assert "[step]" in captured_inputs["unoptimized_dspy_program"]
    assert "Inputs:" in captured_inputs["unoptimized_dspy_program"]
    assert "Outputs:" in captured_inputs["unoptimized_dspy_program"]

    kernel.close()


def test_rlm_session_baseline_summary_includes_train_and_val_runs():
    captured_inputs: dict[str, Any] = {}

    def capture_factory(**_kwargs):
        class Agent:
            def __call__(self, **inputs):
                captured_inputs.update(inputs)
                return dspy.Prediction(
                    optimized_dspy_program={"step": "Return the answer exactly."},
                    trajectory=[],
                    final_reasoning="",
                )

        return Agent()

    kernel = OptimizationKernel(
        program=RuleProgram(),
        trainset=build_trainset(4),
        valset=build_trainset(2),
        metric=exact_metric,
        eval_lm=None,
        num_threads=1,
        max_iterations=3,
        max_output_chars=20_000,
    )
    baseline = kernel.run_baseline()
    assert baseline["split"] == "val"
    val_run_id = str(baseline["run_id"])

    train_run_id = next(
        run_id
        for run_id, meta in kernel.state.runs.items()
        if meta.split == "train"
    )

    session = RLMSession(
        root_lm=DummyLM("root"),
        sub_lm=None,
        max_iterations=10,
        max_llm_calls=10,
        max_output_chars=20_000,
        verbose=False,
        rlm_factory=capture_factory,
    )
    _ = session.run(kernel)

    summary = str(captured_inputs["unoptimized_baseline_summary"])
    assert "You already have two baseline evaluation runs saved:" in summary
    assert f"- run_id: {train_run_id}" in summary
    assert f"- run_id: {val_run_id}" in summary
    assert f"Current latest_run_id: {train_run_id}" in summary
    assert "Budget remaining" not in summary

    kernel.close()


def test_rlm_session_threads_and_resets_root_stateful_session():
    kernel = OptimizationKernel(
        program=RuleProgram(),
        trainset=build_trainset(6),
        valset=None,
        metric=exact_metric,
        eval_lm=None,
        num_threads=1,
        max_iterations=2,
        max_output_chars=20_000,
    )
    kernel.run_baseline()

    root_lm = _RecordedResponsesLM("openai/fake-root", kwargs={"store": True})
    session = RLMSession(
        root_lm=root_lm,
        sub_lm=None,
        max_iterations=10,
        max_llm_calls=10,
        max_output_chars=20_000,
        verbose=False,
        rlm_factory=_stateful_budgeting_factory,
        root_stateful_session=True,
    )
    _ = session.run(kernel)
    _ = session.run(kernel)

    assert len(root_lm.forward_kwargs) == 4
    assert root_lm.forward_kwargs[0].get("previous_response_id") is None
    assert root_lm.forward_kwargs[1].get("previous_response_id") == "resp_1"
    assert root_lm.forward_kwargs[2].get("previous_response_id") is None
    assert root_lm.forward_kwargs[3].get("previous_response_id") == "resp_3"
    assert kernel.state.root_lm_calls == 4
    assert kernel.state.sub_lm_calls == 0

    kernel.close()


def test_rlm_session_stateful_root_can_be_disabled():
    kernel = OptimizationKernel(
        program=RuleProgram(),
        trainset=build_trainset(5),
        valset=None,
        metric=exact_metric,
        eval_lm=None,
        num_threads=1,
        max_iterations=2,
        max_output_chars=20_000,
    )
    kernel.run_baseline()

    root_lm = _RecordedResponsesLM("openai/fake-root", kwargs={"store": True})
    session = RLMSession(
        root_lm=root_lm,
        sub_lm=None,
        max_iterations=10,
        max_llm_calls=10,
        max_output_chars=20_000,
        verbose=False,
        rlm_factory=_stateful_budgeting_factory,
        root_stateful_session=False,
    )
    _ = session.run(kernel)

    assert len(root_lm.call_kwargs) == 2
    assert root_lm.call_kwargs[0].get("previous_response_id") is None
    assert root_lm.call_kwargs[1].get("previous_response_id") is None
    assert root_lm.forward_kwargs == []
    assert kernel.state.root_lm_calls == 2

    kernel.close()


def test_rlm_session_incompatible_root_lm_falls_back_to_stateless():
    kernel = OptimizationKernel(
        program=RuleProgram(),
        trainset=build_trainset(5),
        valset=None,
        metric=exact_metric,
        eval_lm=None,
        num_threads=1,
        max_iterations=2,
        max_output_chars=20_000,
    )
    kernel.run_baseline()

    root_lm = _RecordedResponsesLM("openai/fake-root", model_type="chat")
    session = RLMSession(
        root_lm=root_lm,
        sub_lm=None,
        max_iterations=10,
        max_llm_calls=10,
        max_output_chars=20_000,
        verbose=False,
        rlm_factory=_stateful_budgeting_factory,
        root_stateful_session=True,
    )
    _ = session.run(kernel)

    assert len(root_lm.call_kwargs) == 2
    assert root_lm.call_kwargs[0].get("previous_response_id") is None
    assert root_lm.call_kwargs[1].get("previous_response_id") is None
    assert root_lm.forward_kwargs == []
    assert kernel.state.root_lm_calls == 2

    kernel.close()


def test_rlm_session_stateful_root_does_not_change_sub_lm_behavior():
    kernel = OptimizationKernel(
        program=RuleProgram(),
        trainset=build_trainset(6),
        valset=None,
        metric=exact_metric,
        eval_lm=None,
        num_threads=1,
        max_iterations=2,
        max_output_chars=20_000,
    )
    kernel.run_baseline()

    root_lm = _RecordedResponsesLM("openai/fake-root", kwargs={"store": True})
    sub_lm = _RecordedResponsesLM("openai/fake-sub", kwargs={"store": True})
    session = RLMSession(
        root_lm=root_lm,
        sub_lm=sub_lm,
        max_iterations=10,
        max_llm_calls=10,
        max_output_chars=20_000,
        verbose=False,
        rlm_factory=_stateful_budgeting_factory,
        root_stateful_session=True,
    )
    _ = session.run(kernel)

    assert len(root_lm.forward_kwargs) == 2
    assert root_lm.forward_kwargs[1].get("previous_response_id") == "resp_1"

    assert len(sub_lm.call_kwargs) == 2
    assert sub_lm.call_kwargs[0].get("previous_response_id") is None
    assert sub_lm.call_kwargs[1].get("previous_response_id") is None
    assert sub_lm.forward_kwargs == []

    assert kernel.state.root_lm_calls == 2
    assert kernel.state.sub_lm_calls == 2

    kernel.close()
