from __future__ import annotations

import inspect

import dspy
import pytest
from dspy.teleprompt.teleprompt import Teleprompter

from rlmoptimizer import RLMDocstringOptimizer

from ._helpers import RuleProgram, build_trainset, exact_metric


class FakeSession:
    def __init__(self, **_kwargs) -> None:
        pass

    def run(self, kernel, *, objective: str):
        del objective
        kernel.optimization_status()
        kernel.evaluate_program(split="train", limit=1)
        kernel.update_prompt("step", "Copy question exactly.")
        kernel.evaluate_program(split="train")
        return {
            "optimized_dspy_program": "optimized",
            "best_run_id": kernel.state.best_run_id,
            "agent_report": "completed",
            "trajectory": [],
            "final_reasoning": "reasoning",
        }


class ThreadCheckSession:
    def __init__(self, **_kwargs) -> None:
        pass

    def run(self, kernel, *, objective: str):
        del objective
        status = kernel.optimization_status()
        assert status["num_threads"] == 3
        return {
            "optimized_dspy_program": "",
            "best_run_id": kernel.state.best_run_id,
            "agent_report": "thread-check",
            "trajectory": [],
            "final_reasoning": "",
        }


class FinalReasoningOnlySession:
    def __init__(self, **_kwargs) -> None:
        pass

    def run(self, kernel, *, objective: str):
        del objective
        kernel.optimization_status()
        return {
            "optimized_dspy_program": "",
            "best_run_id": kernel.state.best_run_id,
            "trajectory": [],
            "final_reasoning": "reasoning-only summary",
        }


class StatefulFlagSession:
    def __init__(self, *, root_stateful_session: bool, **_kwargs) -> None:
        self._root_stateful_session = root_stateful_session

    def run(self, kernel, *, objective: str):
        del objective
        assert self._root_stateful_session is False
        kernel.optimization_status()
        return {
            "optimized_dspy_program": "",
            "best_run_id": kernel.state.best_run_id,
            "agent_report": "stateful-flag-check",
            "trajectory": [],
            "final_reasoning": "",
        }


def test_optimizer_is_teleprompter_compatible():
    optimizer = RLMDocstringOptimizer(
        max_iterations=3,
        root_lm=dspy.LM("openai/mock-root"),
        eval_lm=dspy.LM("openai/mock-eval"),
        session_cls=FakeSession,
    )
    assert isinstance(optimizer, Teleprompter)

    compile_sig = inspect.signature(optimizer.compile)
    assert "teacher" in compile_sig.parameters
    assert "valset" in compile_sig.parameters


def test_compile_returns_best_checkpoint_program():
    optimizer = RLMDocstringOptimizer(
        max_iterations=3,
        root_lm=dspy.LM("openai/mock-root"),
        eval_lm=dspy.LM("openai/mock-eval"),
        session_cls=FakeSession,
        rlm_max_iterations=64,
    )

    optimized = optimizer.compile(
        student=RuleProgram(),
        trainset=build_trainset(4),
        metric=exact_metric,
    )

    assert isinstance(optimized, dspy.Module)
    assert optimized.best_score == 100.0
    assert optimized.baseline_run_id is not None
    assert optimized.best_run_id is not None
    assert len(optimized.trial_logs) == 3
    assert optimized.agent_final_reasoning == "reasoning"
    assert optimized(question="hello").answer == "hello"


def test_compile_sets_agent_report_from_final_reasoning_when_report_missing():
    optimizer = RLMDocstringOptimizer(
        max_iterations=3,
        root_lm=dspy.LM("openai/mock-root"),
        eval_lm=dspy.LM("openai/mock-eval"),
        session_cls=FinalReasoningOnlySession,
    )

    optimized = optimizer.compile(
        student=RuleProgram(),
        trainset=build_trainset(3),
        metric=exact_metric,
    )

    assert optimized.agent_report == "reasoning-only summary"


def test_rejects_model_name_strings_for_root_lm():
    with pytest.raises(TypeError, match="root_lm must be a dspy.LM instance"):
        RLMDocstringOptimizer(
            max_iterations=3,
            root_lm="openai/mock-root",
            eval_lm=dspy.LM("openai/mock-eval"),
            session_cls=FakeSession,
        )


def test_rejects_model_name_strings_for_sub_and_eval_lm():
    with pytest.raises(TypeError, match="sub_lm must be a dspy.LM instance or None"):
        RLMDocstringOptimizer(
            max_iterations=3,
            root_lm=dspy.LM("openai/mock-root"),
            sub_lm="openai/mock-sub",
            eval_lm=dspy.LM("openai/mock-eval"),
            session_cls=FakeSession,
        )

    with pytest.raises(TypeError, match="eval_lm must be a dspy.LM instance or None"):
        RLMDocstringOptimizer(
            max_iterations=3,
            root_lm=dspy.LM("openai/mock-root"),
            eval_lm="openai/mock-eval",
            session_cls=FakeSession,
        )


def test_compile_forwards_num_threads_to_kernel():
    optimizer = RLMDocstringOptimizer(
        max_iterations=3,
        root_lm=dspy.LM("openai/mock-root"),
        eval_lm=dspy.LM("openai/mock-eval"),
        num_threads=3,
        session_cls=ThreadCheckSession,
    )
    _ = optimizer.compile(
        student=RuleProgram(),
        trainset=build_trainset(3),
        metric=exact_metric,
    )


def test_compile_forwards_root_stateful_session_flag_to_session():
    optimizer = RLMDocstringOptimizer(
        max_iterations=3,
        root_lm=dspy.LM("openai/mock-root"),
        eval_lm=dspy.LM("openai/mock-eval"),
        root_stateful_session=False,
        session_cls=StatefulFlagSession,
    )
    _ = optimizer.compile(
        student=RuleProgram(),
        trainset=build_trainset(3),
        metric=exact_metric,
    )


def test_rejects_non_bool_root_stateful_session():
    with pytest.raises(TypeError, match="root_stateful_session must be a bool"):
        RLMDocstringOptimizer(
            max_iterations=3,
            root_lm=dspy.LM("openai/mock-root"),
            eval_lm=dspy.LM("openai/mock-eval"),
            root_stateful_session="yes",  # type: ignore[arg-type]
            session_cls=FakeSession,
        )


def test_rejects_non_positive_num_threads():
    with pytest.raises(ValueError, match="num_threads"):
        RLMDocstringOptimizer(
            max_iterations=3,
            num_threads=0,
            root_lm=dspy.LM("openai/mock-root"),
            eval_lm=dspy.LM("openai/mock-eval"),
            session_cls=FakeSession,
        )
