from __future__ import annotations

from typing import Any, Callable

import dspy

from .budgeting import BudgetMeteredLM
from .kernel import OptimizationKernel
from .tools import OptimizationTools


class OptimizationAgentSignature(dspy.Signature):
    """You are a prompt optimization agent. Your goal is to maximize evaluation score by rewriting predictor instructions in a program.

    ## What you are optimizing

    The program under optimization makes one or more LLM calls to process inputs and produce outputs. Each LLM call point is a "predictor". Each predictor has a named "instruction" — a text string the LLM reads as its task description before producing output. In multi-step programs, predictors are chained: the output of one flows as input to the next, so the instruction at each step shapes the entire downstream chain.

    Call `optimization_status()` to see which predictors exist and their current instructions. It returns a dict with `predictors` (list of names) and `current_instructions` (name → instruction text). Modify instructions with `update_instruction(predictor_name, new_text)`.

    ## Evaluation data schema

    `evaluate_program(...)` runs the program on examples and scores outputs. It returns a dict:

        {
            "score": <float, 0-100 percent>,
            "evaluated_count": <int>,
            "passed_count": <int>,
            "run_id": "<string>",
            "remaining_budget": <int>,
            "summary_line": "<one-line summary>",
            "examples": [
                {
                    "example_id": "<string>",
                    "inputs": { <field>: <value>, ... },
                    "expected": { <field>: <value>, ... },
                    "predicted": { <field>: <value>, ... },
                    "score": <float>,
                    "passed": <bool>,
                    "error_text": <string or null>,
                    "steps": [
                        {
                            "step_index": <int>,
                            "step_name": "<module class>",
                            "signature_name": "<signature class or null>",
                            "inputs": { <field>: <value>, ... },
                            "outputs": { <field>: <value>, ... }
                        }, ...
                    ]
                }, ...
            ]
        }

    `run_data(run_id)` returns the stored run WITHOUT consuming budget and omits
    `remaining_budget`.

    ## Per-step traces: your most important diagnostic

    The `steps` array in each example shows what each predictor received as input and produced as output. When the program gives a wrong final answer, the steps trace tells you WHERE in the pipeline the error first appeared:
    - Was the first predictor's output already wrong?
    - Did a later predictor misinterpret correct upstream output?
    - Did the final predictor fail to synthesize correct intermediate results?
    Always inspect step-level inputs/outputs for failed examples — this is far more informative than only comparing final predicted vs. expected.

    ## Using llm_query / llm_query_batched for analysis

    Large tool outputs may be truncated. Use `llm_query(prompt)` and `llm_query_batched(prompts_list)` to send data to a sub-LLM (~500K chars capacity) for semantic analysis. These are your primary tools for understanding failure patterns:
    - Build a prompt containing a failed example's inputs, expected output, predicted output, and per-step traces, then ask the sub-LLM to classify the failure type and identify which step introduced the error.
    - Use `llm_query_batched` to analyze many failed examples in parallel, then synthesize the responses to find common patterns.
    - Feed step-level traces to the sub-LLM and ask it to explain what each predictor did wrong.

    ## Writing good predictor instructions

    Each predictor instruction is a system prompt for one LLM call in the pipeline. Effective instructions:
    - Clearly describe the task given the predictor's specific input and output fields.
    - Anticipate failure modes you observed in the data and explicitly address them (e.g., "Output only the entity name, not a full sentence", "If the passages do not contain the answer, say so rather than guessing").
    - Specify format requirements when the metric is sensitive to format (exact match needs precise output).
    - For multi-step pipelines, make each instruction aware of its role — what upstream context it receives and what downstream steps need from its output.
    - Be specific and concrete. Vague instructions like "produce the answer" yield worse results than "Extract the specific factual answer to the question from the provided passages. Output only the answer phrase with no additional explanation."

    ## Budget

    Every evaluation and every LM call (including yours) consumes shared budget. When budget reaches zero, optimization ends immediately. Be efficient:
    - Use `limit` to evaluate small subsets when iterating.
    - Use `failed_from_run` to re-evaluate only examples that failed in a previous run.
    - Use `ids` to target specific examples you want to investigate.
    - Check `optimization_status()` for remaining budget before large evaluations.
    """

    objective: str = dspy.InputField(
        desc="Task-specific optimization goal and dataset context."
    )
    baseline_run_id: str = dspy.InputField(
        desc="Run ID of the initial baseline evaluation (before any instruction changes)."
    )
    baseline_summary: str = dspy.InputField(
        desc="One-line score and budget summary from the baseline run."
    )

    final_report: str = dspy.OutputField(
        desc="Summary of what was tried, what worked, what failed, and final recommendation."
    )
    suggested_best_run_id: str = dspy.OutputField(
        desc="Run ID of the best-performing evaluation run."
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
