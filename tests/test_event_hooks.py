from __future__ import annotations

import threading
from typing import Any

import dspy

from rlmoptimizer.budgeting import BudgetMeteredLM
from rlmoptimizer.kernel import OptimizationKernel
from rlmoptimizer.rlm_session import RLMSession
from rlmoptimizer.tools import OptimizationTools

from ._helpers import RuleProgram, build_trainset, exact_metric


class _DummyLM:
    def __call__(self, *_args: Any, **_kwargs: Any):
        return ["ok"]

    async def acall(self, *_args: Any, **_kwargs: Any):
        return ["ok"]

    def copy(self, **_kwargs: Any):
        return _DummyLM()


def _session_factory(**kwargs):
    sub_lm = kwargs.get("sub_lm")

    class Agent:
        def __call__(self, **_inputs):
            root_lm = dspy.settings.get("lm")
            root_lm("root")
            if sub_lm is not None:
                sub_lm("sub")
            return dspy.Prediction(
                optimized_dspy_program="",
                best_run_id="",
                trajectory=[],
                final_reasoning="done",
            )

    return Agent()


def test_kernel_event_callback_emits_baseline_eval_budget_and_prompt_events():
    events: list[dict[str, Any]] = []
    kernel = OptimizationKernel(
        program=RuleProgram(),
        trainset=build_trainset(4),
        valset=build_trainset(2),
        metric=exact_metric,
        eval_lm=None,
        num_threads=1,
        max_iterations=2,
        max_output_chars=20_000,
        event_callback=lambda event: events.append(dict(event)),
    )

    _ = kernel.run_baseline()
    _ = kernel.evaluate_program(split="train", limit=1)
    _ = kernel.update_prompt("step", "Copy question exactly.")

    names = {(event.get("source"), event.get("event")) for event in events}
    assert ("kernel", "baseline_started") in names
    assert ("kernel", "baseline_completed") in names
    assert ("kernel", "evaluation_started") in names
    assert ("kernel", "evaluation_completed") in names
    assert ("kernel", "budget_charged") in names
    assert ("kernel", "prompt_updated") in names
    kernel.close()


def test_tools_event_callback_emits_tool_call_start_and_end():
    events: list[dict[str, Any]] = []
    kernel = OptimizationKernel(
        program=RuleProgram(),
        trainset=build_trainset(3),
        valset=None,
        metric=exact_metric,
        eval_lm=None,
        num_threads=1,
        max_iterations=2,
        max_output_chars=20_000,
    )
    tools = OptimizationTools(
        kernel,
        event_callback=lambda event: events.append(dict(event)),
    )
    payload = tools.evaluate_program(limit=1)

    assert payload["evaluated_count"] == 1
    assert any(
        event.get("source") == "tools" and event.get("event") == "tool_call_started"
        for event in events
    )
    assert any(
        event.get("source") == "tools" and event.get("event") == "tool_call_completed"
        for event in events
    )
    kernel.close()


def test_budget_metered_lm_event_callback_emits_request_lifecycle():
    events: list[dict[str, Any]] = []
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
    wrapped = BudgetMeteredLM(
        lm=_DummyLM(),
        budget_consumer=kernel,
        source="root",
        event_callback=lambda event: events.append(dict(event)),
    )

    result = wrapped("ping")

    assert result == ["ok"]
    assert any(
        event.get("source") == "budget_lm"
        and event.get("event") == "lm_request_started"
        for event in events
    )
    assert any(
        event.get("source") == "budget_lm"
        and event.get("event") == "lm_request_completed"
        for event in events
    )
    kernel.close()


def test_rlm_session_event_callback_emits_session_and_lm_events():
    events: list[dict[str, Any]] = []
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
        root_lm=_DummyLM(),
        sub_lm=_DummyLM(),
        max_iterations=10,
        max_llm_calls=10,
        max_output_chars=20_000,
        verbose=False,
        rlm_factory=_session_factory,
        event_callback=lambda event: events.append(dict(event)),
    )
    _ = session.run(kernel, objective="event callback test")

    assert any(
        event.get("source") == "session" and event.get("event") == "session_started"
        for event in events
    )
    assert any(
        event.get("source") == "session" and event.get("event") == "session_completed"
        for event in events
    )
    assert any(
        event.get("source") == "budget_lm"
        and event.get("event") == "lm_request_completed"
        for event in events
    )
    kernel.close()


def test_kernel_budget_callback_reentry_does_not_deadlock():
    callback_state = {"reentered": False}
    holder: dict[str, OptimizationKernel] = {}

    def _callback(event: dict[str, Any]) -> None:
        if event.get("event") != "budget_charged":
            return
        if callback_state["reentered"]:
            return
        callback_state["reentered"] = True
        holder["kernel"].charge_llm_requests(source="root", requests=1)

    kernel = OptimizationKernel(
        program=RuleProgram(),
        trainset=build_trainset(3),
        valset=None,
        metric=exact_metric,
        eval_lm=None,
        num_threads=1,
        max_iterations=2,
        max_output_chars=20_000,
        event_callback=_callback,
    )
    holder["kernel"] = kernel

    done = {"value": False}

    def _charge_once() -> None:
        kernel.charge_llm_requests(source="root", requests=1)
        done["value"] = True

    worker = threading.Thread(target=_charge_once, daemon=True)
    worker.start()
    worker.join(timeout=1.0)

    assert worker.is_alive() is False
    assert done["value"] is True
    assert callback_state["reentered"] is True
    assert kernel.state.remaining_budget == kernel.max_budget - 2
    assert kernel.state.root_lm_calls == 2
    kernel.close()
