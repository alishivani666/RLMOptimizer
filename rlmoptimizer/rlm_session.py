from __future__ import annotations

from typing import Any, Callable

import dspy

from .budgeting import BudgetMeteredLM
from .kernel import OptimizationKernel
from .tools import OptimizationTools


class OptimizationAgentSignature(dspy.Signature):
    """You are optimizing a program that takes inputs and produces outputs through one or more LLM calls.

    Each LLM call point is a predictor with a named instruction — the task description the LLM reads before producing its output. In multi-step programs, predictors are chained in a pipeline where the output of onestep flows as input to the next, so instructions at each step shape the entire downstream chain.

    Your goal is to maximize evaluation score by editing predictor instructions. Each instruction edit is prompt engineering — you are rewriting what the LLM is told to do at that step.

    Every evaluation and every LM call consumes shared budget. When budget is exhausted, optimization ends immediately.
    """

    objective: str = dspy.InputField(desc="Task-specific optimization goal and dataset summary.")
    baseline_run_id: str = dspy.InputField(desc="Run ID of the initial unoptimized evaluation.")
    baseline_summary: str = dspy.InputField(desc="Score and budget summary from the baseline run.")

    final_report: str = dspy.OutputField(desc="Summary of what was tried, what worked, and final recommendation.")
    suggested_best_run_id: str = dspy.OutputField(
        desc="Run ID of the best-performing evaluation."
    )


class RLMSession:
    def __init__(
        self,
        *,
        root_lm: Any,
        sub_lm: Any | None,
        max_iterations: int,
        max_llm_calls: int,
        max_output_chars: int,
        verbose: bool,
        rlm_factory: Callable[..., Any] | None = None,
    ) -> None:
        self._root_lm = root_lm
        self._sub_lm = sub_lm
        self._max_iterations = int(max_iterations)
        self._max_llm_calls = int(max_llm_calls)
        self._max_output_chars = int(max_output_chars)
        self._verbose = bool(verbose)
        self._rlm_factory = rlm_factory

    def _build_rlm(self, tools: OptimizationTools, *, sub_lm: Any | None) -> Any:
        rlm_tools = tools.as_dspy_tools()
        kwargs = {
            "signature": OptimizationAgentSignature,
            "max_iterations": self._max_iterations,
            "max_llm_calls": self._max_llm_calls,
            "max_output_chars": self._max_output_chars,
            "verbose": self._verbose,
            "tools": rlm_tools,
            "sub_lm": sub_lm,
        }

        if self._rlm_factory is not None:
            return self._rlm_factory(**kwargs)

        return dspy.RLM(**kwargs)

    def run(self, kernel: OptimizationKernel, *, objective: str) -> dict[str, Any]:
        tools = OptimizationTools(kernel)
        root_lm = BudgetMeteredLM(lm=self._root_lm, budget_consumer=kernel, source="root")
        sub_lm = (
            BudgetMeteredLM(lm=self._sub_lm, budget_consumer=kernel, source="sub")
            if self._sub_lm is not None
            else None
        )
        rlm = self._build_rlm(tools, sub_lm=sub_lm)

        baseline_run_id = kernel.state.baseline_run_id or ""
        baseline_payload = (
            kernel.run_data_raw(baseline_run_id)
            if baseline_run_id
            else {"summary_line": ""}
        )
        baseline_summary = str(baseline_payload.get("summary_line", ""))

        with dspy.context(lm=root_lm):
            prediction = rlm(
                objective=objective,
                baseline_run_id=baseline_run_id,
                baseline_summary=baseline_summary,
            )

        return {
            "final_report": str(getattr(prediction, "final_report", "")),
            "suggested_best_run_id": str(
                getattr(prediction, "suggested_best_run_id", "")
            ),
            "trajectory": getattr(prediction, "trajectory", []),
            "final_reasoning": getattr(prediction, "final_reasoning", ""),
        }
