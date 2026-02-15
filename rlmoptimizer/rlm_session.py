from __future__ import annotations

import json
import logging
from typing import Any, Callable

import dspy
from dspy.predict.rlm import _strip_code_fences
from dspy.primitives.code_interpreter import CodeInterpreterError
from dspy.primitives.prediction import Prediction

from .budgeting import BudgetMeteredLM
from .interpreter import ReRegisteringPythonInterpreter
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
        event_callback: Callable[[dict[str, Any]], None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._iteration_start_callback = iteration_start_callback
        self._iteration_output_callback = iteration_output_callback
        self._event_callback = event_callback

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
            self._emit_event(
                "iteration_error",
                {
                    "iteration": iteration,
                    "max_iterations": self.max_iterations,
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                    "code": code,
                },
            )
            execution_result = f"[Error] {exc}"

        result = self._process_execution_result(
            action, execution_result, history, output_field_names
        )
        self._emit_iteration_output(self._extract_output_text(result))

        return result

    def _emit_iteration_start(
        self, *, iteration: int, reasoning: str, code: str
    ) -> None:
        self._emit_event(
            "iteration_started",
            {
                "iteration": iteration,
                "max_iterations": self.max_iterations,
                "reasoning": reasoning,
                "code": code,
            },
        )
        self._safe_display_call(
            self._iteration_start_callback,
            iteration,
            self.max_iterations,
            str(reasoning),
            str(code),
        )

    def _emit_iteration_output(self, output: str) -> None:
        self._emit_event(
            "iteration_output",
            {
                "output": output,
            },
        )
        self._safe_display_call(self._iteration_output_callback, output)

    def _emit_event(self, event_type: str, payload: dict[str, Any]) -> None:
        callback = self._event_callback
        if callback is None:
            return
        try:
            callback({"source": "rlm", "event": event_type, **payload})
        except Exception:
            pass

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


def _build_unoptimized_baseline_summary(kernel: OptimizationKernel) -> str:
    train_run_id = str(getattr(kernel, "baseline_train_run_id", "") or "").strip()
    if not train_run_id:
        raise RuntimeError("baseline_train_run_id is unavailable; call run_baseline() first.")

    val_run_id = str(getattr(kernel, "baseline_val_run_id", "") or "").strip()
    latest_run_id = str(kernel.state.latest_run_id or "").strip()

    train_payload = kernel.run_data(train_run_id)
    if not isinstance(train_payload, dict):
        raise RuntimeError("train baseline payload is unavailable.")

    if val_run_id:
        val_payload = kernel.run_data(val_run_id)
        if not isinstance(val_payload, dict):
            raise RuntimeError("validation baseline payload is unavailable.")
        val_score = f"{val_payload['score']}%"
        val_size = str(val_payload["evaluated_count"])
    else:
        val_score = "n/a"
        val_size = "0"

    lines = [
        "You already have two baseline evaluation runs saved:",
        "",
        "1) Baseline train run",
        f"- run_id: {train_run_id}",
        f"- score: {train_payload['score']}%",
        f"- size: {train_payload['evaluated_count']}",
        "- diagnostics: per-example data available via run_data(run_id)",
        "",
        "2) Baseline validation run",
        f"- run_id: {val_run_id or '(not available)'}",
        f"- score: {val_score}",
        f"- size: {val_size}",
        "- diagnostics: aggregate-only (examples are hidden)",
        "",
        f"Current latest_run_id: {latest_run_id or '(not available)'}",
    ]
    return "\n".join(lines)


class PromptOptimizationSignature(dspy.Signature):
    """You are optimizing prompts in an LLM program. You are optimizing prompts in an LLM program. The program processes instances through one or more steps. Each step calls an LLM with a prompt that tells it what to do.

    ## Goal
    - Achieve the highest score you can on train and validation, while ensuring these improvements generalize to a held-out test set. 
    - Maximize score on both splits. A large gap between them suggests overfitting.
    - Use validation checkpoints throughout optimization when train-side changes look meaningfully better.

    ## Running experiments
    - Use `evaluate_program()` to test your changes.
    - Train split (`split='train'`) for fast iteration and diagnostics:
        - `limit`: Evaluate a subset (e.g., `limit=15`).
        - `sample`: Use `sample='random'` for random sampling; use `sample_seed` for reproducibility.
        - `failed_from_run`: Re-evaluate only instances that failed in a previous run (pass the run_id).
        - `ids`: Target specific instances by ID (e.g., `ids='3,7,12'` or `ids='10-20'`).
    - Validation split (`split='val'`) for checkpointing generalization:
        - Use `evaluate_program(split='val')` when you want a validation checkpoint.
        - It always evaluates the full validation set (no subset validation).
        - Any selector args passed with `split='val'` are ignored.
    - Use `run_data(run_id)` to re-read previous evaluations at no budget cost.
    
    ## Existing baseline runs
    - `unoptimized_baseline_summary` includes baseline train and validation run IDs, scores, and sizes.
    - Before spending budget on new evaluations, inspect the baseline train run first with `run_data(<baseline_train_run_id>)`.

    ## Diagnosing failures
    - Evaluations return per-step traces showing what each step received as input and what it produced as output. This is your primary diagnostic tool. When the final output is wrong, trace back through the steps to find where the error first appeared.
    - In multi-step programs, errors cascade: a bad prompt in step 1 produces flawed output that causes step 2 to fail, which causes step 3 to fail. Fix the root cause, not the downstream symptoms.

    ## Budget
    - Each instance evaluated costs one budget unit.
    - Every `evaluate_program()` result includes `remaining_budget`; treat that as your primary budget signal and keep track of it as you iterate.
    - A requested evaluation fails if its required instances exceed remaining budget.
    - `split='val'` always runs the full validation set, so its cost is the full val-set size.
    - `split='train'` cost equals the number of selected train instances.
    - When budget is getting low, do an explicit affordability check before any evaluation call (`train` or `val`).
    - When evaluation budget reaches zero, no more evaluations can run, but you may still continue analysis and decide which set of prompts you would like to submit.

    ## Coding Environment
    - Output exactly one code block per response, formatted as: ```python <code> ```
    - Do not add extra ``` markers or text after the closing fence—they because will cause a SyntaxError.
    - Your environment persists across iterations:
        - Imports, variables, and functions stay available in later iterations.
        - Define helpers early and reuse them. Do not rewrite the same code each turn.
    - Printed output that exceeds the per-iteration character limit is truncated. However, only the display is cut short. The actual data itself is not lost.
    
    ## Submission
    - Submit `optimized_dspy_program` as a dict[str, str] containing exactly one prompt for every step.
    - Use the same step names returned by `optimization_status()['steps']` and trace `steps[*].step_name`."""

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
        desc="How many evaluations you can run."
    )

    optimized_dspy_program: dict[str, str] = dspy.OutputField(
        desc='Final prompt map as {"step_name": "prompt text", ...} for all steps.'
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
        event_callback: Callable[[dict[str, Any]], None] | None = None,
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
        self._event_callback = event_callback

    def _emit_event(self, event_type: str, payload: dict[str, Any]) -> None:
        callback = self._event_callback
        if callback is None:
            return
        try:
            callback({"source": "session", "event": event_type, **payload})
        except Exception:
            pass

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

        kwargs["interpreter"] = ReRegisteringPythonInterpreter()

        if self._debug_display is not None or self._event_callback is not None:
            start_callback = (
                self._debug_display.show_iteration_start
                if self._debug_display is not None
                else None
            )
            output_callback = (
                self._debug_display.show_iteration_output
                if self._debug_display is not None
                else None
            )
            return _InstrumentedRLM(
                iteration_start_callback=start_callback,
                iteration_output_callback=output_callback,
                event_callback=self._event_callback,
                **kwargs,
            )

        return dspy.RLM(**kwargs)

    def run(
        self,
        kernel: OptimizationKernel,
    ) -> dict[str, Any]:
        self._emit_event(
            "session_started",
            {
                "max_iterations": self._max_iterations,
                "max_llm_calls": self._max_llm_calls,
            },
        )
        tools = OptimizationTools(kernel, event_callback=self._event_callback)
        root_lm = BudgetMeteredLM(
            lm=self._build_root_lm_for_run(),
            budget_consumer=kernel,
            source="root",
            event_callback=self._event_callback,
        )
        sub_lm = (
            BudgetMeteredLM(
                lm=self._sub_lm,
                budget_consumer=kernel,
                source="sub",
                event_callback=self._event_callback,
            )
            if self._sub_lm is not None
            else None
        )
        rlm = self._build_rlm(tools, sub_lm=sub_lm)

        unoptimized_baseline_summary = _build_unoptimized_baseline_summary(kernel)
        unoptimized_dspy_program = _render_program_text(kernel.program)

        try:
            with dspy.context(lm=root_lm, adapter=dspy.JSONAdapter()):
                prediction = rlm(
                    unoptimized_dspy_program=unoptimized_dspy_program,
                    unoptimized_baseline_summary=unoptimized_baseline_summary,
                    total_budget_remaining=int(kernel.state.remaining_budget),
                )

            submitted_prompt_map = getattr(prediction, "optimized_dspy_program", None)
            if not isinstance(submitted_prompt_map, dict):
                raise TypeError(
                    "optimized_dspy_program must be returned as dict[str, str]."
                )

            final_reasoning = str(getattr(prediction, "final_reasoning", "")).strip()
            result = {
                "optimized_dspy_program": dict(submitted_prompt_map),
                "trajectory": getattr(prediction, "trajectory", []),
                "final_reasoning": final_reasoning,
            }
            self._emit_event(
                "session_completed",
                {
                    "final_reasoning": final_reasoning,
                    "submitted_steps": sorted(str(key) for key in submitted_prompt_map.keys()),
                },
            )
            return result
        except Exception as exc:
            self._emit_event(
                "session_failed",
                {
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                },
            )
            raise
