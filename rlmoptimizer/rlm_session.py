from __future__ import annotations

import json
import logging
from typing import Any, Callable

import dspy
from dspy.predict.rlm import _strip_code_fences
from dspy.primitives.code_interpreter import CodeInterpreterError
from dspy.primitives.prediction import Prediction

from .budgeting import BudgetMeteredLM
from .kernel import OptimizationKernel
from .root_state import maybe_wrap_stateful_root_lm
from .tools import OptimizationTools

logger = logging.getLogger(__name__)


class _InstrumentedRLM(dspy.RLM):
    """RLM subclass that emits start/output callbacks for each iteration."""

    def __init__(
        self,
        *args: Any,
        iteration_start_callback: Callable[[int, int, str, str], None] | None = None,
        iteration_output_callback: Callable[[str], None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._iteration_start_callback = iteration_start_callback
        self._iteration_output_callback = iteration_output_callback

    def _execute_iteration(self, repl, variables, history, iteration, input_args, output_field_names):
        variables_info = [variable.format() for variable in variables]
        action = self.generate_action(
            variables_info=variables_info,
            repl_history=history,
            iteration=f"{iteration + 1}/{self.max_iterations}",
        )
        if self.verbose:
            logger.info(
                f"RLM iteration {iteration + 1}/{self.max_iterations}\n"
                f"Reasoning: {action.reasoning}\nCode:\n{action.code}"
            )

        code = self._normalize_code(action.code)
        self._emit_iteration_start(
            iteration=iteration,
            reasoning=action.reasoning,
            code=code,
        )

        try:
            execution_result = repl.execute(code, variables=dict(input_args))
        except (CodeInterpreterError, SyntaxError) as exc:
            execution_result = f"[Error] {exc}"

        result = self._process_execution_result(
            action, execution_result, history, output_field_names
        )
        self._emit_iteration_output(self._extract_output_text(result))

        return result

    def _emit_iteration_start(
        self, *, iteration: int, reasoning: str, code: str
    ) -> None:
        self._safe_display_call(
            self._iteration_start_callback,
            iteration,
            self.max_iterations,
            str(reasoning),
            str(code),
        )

    def _emit_iteration_output(self, output: str) -> None:
        self._safe_display_call(self._iteration_output_callback, output)

    def _safe_display_call(self, callback: Callable[..., None] | None, *args: Any) -> None:
        if callback is None:
            return
        try:
            callback(*args)
        except Exception:
            pass

    def _normalize_code(self, code: Any) -> str:
        return _strip_code_fences(str(code))

    def _extract_output_text(self, result: Prediction | Any) -> str:
        if isinstance(result, Prediction):
            trajectory = result.trajectory
            if not trajectory:
                return ""
            return str(trajectory[-1].get("output", ""))

        entries = getattr(result, "entries", None)
        if not entries:
            return ""
        return str(getattr(entries[-1], "output", ""))


def _field_type_name(field: Any) -> str:
    annotation = getattr(field, "annotation", None)
    if annotation is None:
        return "Any"
    if isinstance(annotation, str):
        return annotation
    name = getattr(annotation, "__name__", None)
    if isinstance(name, str) and name:
        return name
    return str(annotation).replace("typing.", "")


def _field_desc(field: Any) -> str:
    extra = getattr(field, "json_schema_extra", None) or {}
    desc = str(extra.get("desc", "")).strip()
    return desc or "No description provided."


def _field_group(field: Any) -> str:
    extra = getattr(field, "json_schema_extra", None) or {}
    return str(extra.get("__dspy_field_type", "")).strip().lower()


def _render_program_text(
    program: dspy.Module,
    *,
    prompt_overrides: dict[str, str] | None = None,
) -> str:
    sections: list[str] = []
    overrides = prompt_overrides or {}

    for step_name, step_module in program.named_predictors():
        signature = getattr(step_module, "signature", None)
        prompt_text = str(
            overrides.get(step_name, getattr(signature, "instructions", "") or "")
        )

        inputs: list[str] = []
        outputs: list[str] = []
        if signature is not None and hasattr(signature, "fields"):
            for field_name, field in signature.fields.items():
                spec = (
                    f"- {field_name} ({_field_type_name(field)}): {_field_desc(field)}"
                )
                group = _field_group(field)
                if group == "input":
                    inputs.append(spec)
                elif group == "output":
                    outputs.append(spec)

        lines = [
            f"[{step_name}]",
            f"Prompt: {json.dumps(prompt_text, ensure_ascii=True)}",
            "Inputs:",
            *(inputs or ["- None"]),
            "Outputs:",
            *(outputs or ["- None"]),
        ]
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


class PromptOptimizationSignature(dspy.Signature):
    """You are optimizing prompts in an LLM program. You are optimizing prompts in an LLM program. The program processes instances through one or more steps. Each step calls an LLM with a prompt that tells it what to do.

    ## Goal
    - Achieve 100% `best_score` or get as close as possible. Keep optimizing until you run out of budget.
    - `best_score` only updates from evaluation on the validation split. Run validation whenever you see a meaningful improvement on the train split.
    - Maximize score on both splits. A large gap between them suggests overfitting.
    - Use `optimization_status()` to check your current `best_score`, prompts, and remaining budget.

    ## Running experiments
    - Use `evaluate_program()` to test your changes. Key arguments for efficient experimentation:
        - `limit`: Evaluate a subset (e.g., `limit=15`) for quick iteration instead of the full dataset.
        - `sample`: With `limit`, use `sample='random'` to select a representative random subset instead of always the first N. Use `sample_seed` for reproducibility.
        - `failed_from_run`: Re-evaluate only instances that failed in a previous run (pass the run_id). The most budget-efficient way to check if a fix worked.
        - `ids`: Target specific instances by ID (e.g., `ids='3,7,12'` or `ids='10-20'`).
        - `split`: Use `'train'` for experimentation. Reserve `'val'` for final validation to confirm generalization.
    - Use `run_data(run_id)` to re-read previous evaluations at no budget cost.

    ## Diagnosing failures
    - Evaluations return per-step traces showing what each step received as input and what it produced as output. This is your primary diagnostic tool. When the final output is wrong, trace back through the steps to find where the error first appeared.
    - In multi-step programs, errors cascade: a bad prompt in step 1 produces flawed output that causes step 2 to fail, which causes step 3 to fail. Fix the root cause, not the downstream symptoms.

    ## What you can change
    - You can only modify prompt text. Use `update_prompt()` to rewrite one step's prompt.
    - The program structure-which steps exist, their inputs and outputs-is fixed.

    ## Budget
    - Each instance evaluated costs one budget unit. Budget is shown in `total_budget_remaining`.
    - When budget reaches zero, optimization ends immediately.

    ## Coding Environment
    - Output exactly one code block per response, formatted as: ```python <code> ```
    - Do not add extra ``` markers or text after the closing fence—they because will cause a SyntaxError.
    - Your environment persists across iterations:
        - Imports, variables, and functions stay available in later iterations.
        - Define helpers early and reuse them. Do not rewrite the same code each turn.
    - Printed output that exceeds the per-iteration character limit is truncated. However, only the display is cut short. The actual data itself is not lost."""

    unoptimized_dspy_program: str = dspy.InputField(
        desc=(
            "An LLM program with multiple steps. Each step has a prompt that instructs "
            "the LLM, plus defined inputs and outputs showing what the LLM receives and "
            "what it must produce."
        )
    )
    unoptimized_baseline_summary: str = dspy.InputField(
        desc="Current score and pass rate of the unoptimized program."
    )
    total_budget_remaining: int = dspy.InputField(
        desc="How many evaluations you can run. Each dataset instance evaluated costs one unit."
    )

    optimized_dspy_program: str = dspy.OutputField(
        desc="The rewritten prompts for each step that achieved the best score."
    )
    best_run_id: str = dspy.OutputField(
        desc="The run ID where the optimized prompts were validated."
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
        debug_display: Any | None = None,
        root_stateful_session: bool = True,
    ) -> None:
        self._root_lm = root_lm
        self._sub_lm = sub_lm
        self._max_iterations = int(max_iterations)
        self._max_llm_calls = int(max_llm_calls)
        self._max_output_chars = int(max_output_chars)
        self._verbose = bool(verbose)
        self._rlm_factory = rlm_factory
        self._debug_display = debug_display
        self._root_stateful_session = bool(root_stateful_session)

    def _rlm_verbose_enabled(self) -> bool:
        """Enable DSPy text logs only when no rich debugger is active."""
        return self._verbose and self._debug_display is None

    def _show_stateful_status(self) -> bool:
        return self._verbose or self._debug_display is not None

    def _build_root_lm_for_run(self) -> Any:
        if not self._root_stateful_session:
            return self._root_lm

        wrapped, issue = maybe_wrap_stateful_root_lm(self._root_lm)
        if issue is not None and self._show_stateful_status():
            logger.warning(
                "Stateful root session disabled; falling back to stateless root LM: %s",
                issue,
            )

        reset_session = getattr(wrapped, "reset_session", None)
        if callable(reset_session):
            reset_session()

        return wrapped

    def _build_rlm(self, tools: OptimizationTools, *, sub_lm: Any | None) -> Any:
        rlm_tools = tools.as_dspy_tools()
        kwargs = {
            "signature": PromptOptimizationSignature,
            "max_iterations": self._max_iterations,
            "max_llm_calls": self._max_llm_calls,
            "max_output_chars": self._max_output_chars,
            "verbose": self._rlm_verbose_enabled(),
            "tools": rlm_tools,
            "sub_lm": sub_lm,
        }

        if self._rlm_factory is not None:
            return self._rlm_factory(**kwargs)

        if self._debug_display is not None:
            return _InstrumentedRLM(
                iteration_start_callback=self._debug_display.show_iteration_start,
                iteration_output_callback=self._debug_display.show_iteration_output,
                **kwargs,
            )

        return dspy.RLM(**kwargs)

    def run(
        self,
        kernel: OptimizationKernel,
        *,
        objective: str | None = None,
    ) -> dict[str, Any]:
        del objective
        tools = OptimizationTools(kernel)
        root_lm = BudgetMeteredLM(
            lm=self._build_root_lm_for_run(),
            budget_consumer=kernel,
            source="root",
        )
        sub_lm = (
            BudgetMeteredLM(lm=self._sub_lm, budget_consumer=kernel, source="sub")
            if self._sub_lm is not None
            else None
        )
        rlm = self._build_rlm(tools, sub_lm=sub_lm)

        baseline_run_id = kernel.state.baseline_run_id or ""
        baseline_payload = kernel.run_data(baseline_run_id) if baseline_run_id else {"summary_line": ""}
        unoptimized_baseline_summary = str(baseline_payload.get("summary_line", ""))
        unoptimized_dspy_program = _render_program_text(kernel.program)

        with dspy.context(lm=root_lm):
            prediction = rlm(
                unoptimized_dspy_program=unoptimized_dspy_program,
                unoptimized_baseline_summary=unoptimized_baseline_summary,
                total_budget_remaining=int(kernel.state.remaining_budget),
            )

        best_run_id = str(getattr(prediction, "best_run_id", "") or kernel.state.best_run_id or "")
        optimized_dspy_program = str(getattr(prediction, "optimized_dspy_program", "")).strip()
        if not optimized_dspy_program:
            optimized_dspy_program = _render_program_text(
                kernel.program,
                prompt_overrides=kernel.state.best_prompt_map,
            )
        agent_report = str(
            getattr(prediction, "agent_report", "")
            or getattr(prediction, "final_reasoning", "")
        ).strip()
        if not agent_report:
            agent_report = (
                f"Optimization completed. Best run id: {best_run_id}."
                if best_run_id
                else "Optimization completed."
            )

        return {
            "optimized_dspy_program": optimized_dspy_program,
            "best_run_id": best_run_id,
            "agent_report": agent_report,
            "trajectory": getattr(prediction, "trajectory", []),
            "final_reasoning": getattr(prediction, "final_reasoning", ""),
        }
