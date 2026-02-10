from __future__ import annotations

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
                final_report="done",
                suggested_best_run_id="",
                trajectory=[],
                final_reasoning="",
            )

    return Agent()


def test_kernel_charges_root_and_sub_lm_calls():
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
    result = session.run(kernel, objective="budget test")

    assert result["final_report"] == "done"
    assert kernel.state.evaluated_examples == 4
    assert kernel.state.root_lm_calls == 2
    assert kernel.state.sub_lm_calls == 1
    assert kernel.state.remaining_budget == 1
    status = kernel.optimization_status()
    assert status["evaluated_examples"] == 4
    assert status["root_lm_calls"] == 2
    assert status["sub_lm_calls"] == 1

    kernel.close()


def test_kernel_budget_exceeded_by_lm_calls():
    kernel = OptimizationKernel(
        program=RuleProgram(),
        trainset=build_trainset(3),
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
        rlm_factory=_budgeting_factory,
    )

    try:
        session.run(kernel, objective="budget overflow")
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
