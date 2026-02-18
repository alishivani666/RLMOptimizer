from __future__ import annotations

import json
import logging
from typing import Any, Callable

import dspy
from dspy.adapters.base import Adapter
from dspy.adapters.types.base_type import split_message_content_for_custom_types
from dspy.predict.rlm import _strip_code_fences
from dspy.primitives.code_interpreter import CodeInterpreterError
from dspy.primitives.prediction import Prediction
from dspy.primitives.repl_types import REPLHistory
from litellm.exceptions import ContextWindowExceededError

from .budgeting import BudgetMeteredLM
from .interpreter import ReRegisteringPythonInterpreter
from .kernel import OptimizationKernel
from .root_state import maybe_wrap_stateful_root_lm
from .tools import OptimizationTools

logger = logging.getLogger(__name__)
_MISSING_PARENT_RESPONSE_ID = object()


class _RLMMultiTurnFormatMixin:
    """Multi-turn input formatting shared by chat and JSON fallback adapters."""

    _REPL_HISTORY_FIELD = "repl_history"
    _ITERATION_FIELD = "iteration"
    _VARIABLES_INFO_FIELD = "variables_info"
    _REASONING_FIELD = "reasoning"
    _CODE_FIELD = "code"

    def format(
        self,
        signature: type[dspy.Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if not self._supports_multiturn_history(signature):
            return super().format(signature=signature, demos=demos, inputs=inputs)

        inputs_copy = dict(inputs)
        history = inputs_copy.pop(self._REPL_HISTORY_FIELD, None)
        if not isinstance(history, REPLHistory):
            return super().format(signature=signature, demos=demos, inputs=inputs)

        signature_without_history = signature.delete(self._REPL_HISTORY_FIELD)
        current_iteration = str(inputs_copy.get(self._ITERATION_FIELD, "")).strip()

        messages: list[dict[str, Any]] = []
        messages.append(
            {
                "role": "system",
                "content": self.format_system_message(signature_without_history),
            }
        )
        messages.extend(self.format_demos(signature_without_history, demos))
        messages.extend(
            self._format_historical_turns(
                signature=signature_without_history,
                base_inputs=inputs_copy,
                history=history,
                current_iteration=current_iteration,
            )
        )

        current_inputs = dict(inputs_copy)
        if not current_inputs.get(self._ITERATION_FIELD):
            total = self._max_iterations_from_iteration_label(current_iteration, len(history.entries) + 1)
            current_inputs[self._ITERATION_FIELD] = f"{len(history.entries) + 1}/{total}"
        if history.entries:
            current_inputs.pop(self._VARIABLES_INFO_FIELD, None)

        current_prefix = ""
        if history.entries:
            current_prefix = self._execution_feedback_prefix(history.entries[-1].output)
        current_content = self.format_user_message_content(
            signature_without_history,
            current_inputs,
            prefix=current_prefix,
            main_request=True,
        )
        messages.append({"role": "user", "content": current_content})

        return split_message_content_for_custom_types(messages)

    def _supports_multiturn_history(self, signature: type[dspy.Signature]) -> bool:
        repl_history = signature.input_fields.get(self._REPL_HISTORY_FIELD)
        if repl_history is None:
            return False
        if getattr(repl_history, "annotation", None) is not REPLHistory:
            return False
        if self._ITERATION_FIELD not in signature.input_fields:
            return False
        if self._REASONING_FIELD not in signature.output_fields:
            return False
        if self._CODE_FIELD not in signature.output_fields:
            return False
        return True

    def _format_historical_turns(
        self,
        *,
        signature: type[dspy.Signature],
        base_inputs: dict[str, Any],
        history: REPLHistory,
        current_iteration: str,
    ) -> list[dict[str, Any]]:
        if not history.entries:
            return []

        max_iterations = self._max_iterations_from_iteration_label(
            current_iteration, len(history.entries) + 1
        )
        turns: list[dict[str, Any]] = []
        previous_output: str | None = None

        for index, entry in enumerate(history.entries, start=1):
            user_inputs = dict(base_inputs)
            user_inputs[self._ITERATION_FIELD] = f"{index}/{max_iterations}"
            if index > 1:
                user_inputs.pop(self._VARIABLES_INFO_FIELD, None)
            user_prefix = (
                self._execution_feedback_prefix(previous_output)
                if previous_output is not None
                else ""
            )
            turns.append(
                {
                    "role": "user",
                    "content": self.format_user_message_content(
                        signature,
                        user_inputs,
                        prefix=user_prefix,
                        main_request=False,
                    ),
                }
            )
            turns.append(
                {
                    "role": "assistant",
                    "content": self.format_assistant_message_content(
                        signature,
                        {
                            self._REASONING_FIELD: str(entry.reasoning),
                            self._CODE_FIELD: self._fenced_code(str(entry.code)),
                        },
                    ),
                }
            )
            previous_output = str(entry.output)

        return turns

    def _max_iterations_from_iteration_label(
        self,
        iteration_label: str,
        fallback: int,
    ) -> int:
        try:
            _current, max_text = iteration_label.split("/", maxsplit=1)
            parsed_max = int(max_text.strip())
            if parsed_max > 0:
                return parsed_max
        except (ValueError, AttributeError):
            pass
        return max(int(fallback), 1)

    def _execution_feedback_prefix(self, output: str | None) -> str:
        text = str(output or "")
        return (
            "Execution result from your previous code:\n"
            "```text\n"
            f"{text}\n"
            "```"
        )

    def _fenced_code(self, code: str) -> str:
        return f"```python\n{code}\n```"


class _RLMMultiTurnJSONFallbackAdapter(_RLMMultiTurnFormatMixin, dspy.JSONAdapter):
    """JSON output adapter that keeps the RLM multi-turn input formatting."""


class _RLMMultiTurnHistoryAdapter(_RLMMultiTurnFormatMixin, dspy.ChatAdapter):
    """Render RLM repl history as alternating user/assistant chat turns."""

    def __init__(self) -> None:
        super().__init__(use_json_adapter_fallback=True)

    def _make_json_fallback_adapter(self) -> _RLMMultiTurnJSONFallbackAdapter:
        return _RLMMultiTurnJSONFallbackAdapter(
            callbacks=list(self.callbacks),
            use_native_function_calling=self.use_native_function_calling,
        )

    def _call_primary(
        self,
        lm: dspy.LM,
        lm_kwargs: dict[str, Any],
        signature: type[dspy.Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        return Adapter.__call__(self, lm, lm_kwargs, signature, demos, inputs)

    async def _acall_primary(
        self,
        lm: dspy.LM,
        lm_kwargs: dict[str, Any],
        signature: type[dspy.Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        return await Adapter.acall(self, lm, lm_kwargs, signature, demos, inputs)

    def _should_use_json_fallback(self, exc: Exception) -> bool:
        return self.use_json_adapter_fallback and not isinstance(
            exc, ContextWindowExceededError
        )

    def _parent_response_id_for_fallback(
        self,
        lm: dspy.LM,
        lm_kwargs: dict[str, Any],
    ) -> Any:
        if "previous_response_id" in lm_kwargs:
            return lm_kwargs.get("previous_response_id")
        return getattr(lm, "previous_response_id", _MISSING_PARENT_RESPONSE_ID)

    def _same_parent_fallback_kwargs(
        self,
        lm_kwargs: dict[str, Any],
        parent_response_id: Any,
    ) -> dict[str, Any]:
        fallback_kwargs = dict(lm_kwargs)
        if parent_response_id is not _MISSING_PARENT_RESPONSE_ID:
            fallback_kwargs["previous_response_id"] = parent_response_id
        return fallback_kwargs

    def _call_json_fallback(
        self,
        lm: dspy.LM,
        lm_kwargs: dict[str, Any],
        signature: type[dspy.Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        fallback = self._make_json_fallback_adapter()
        return fallback(lm, lm_kwargs, signature, demos, inputs)

    async def _acall_json_fallback(
        self,
        lm: dspy.LM,
        lm_kwargs: dict[str, Any],
        signature: type[dspy.Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        fallback = self._make_json_fallback_adapter()
        return await fallback.acall(lm, lm_kwargs, signature, demos, inputs)

    def _call_with_json_fallback(
        self,
        lm: dspy.LM,
        lm_kwargs: dict[str, Any],
        signature: type[dspy.Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        parent_response_id = self._parent_response_id_for_fallback(lm, lm_kwargs)
        try:
            return self._call_primary(lm, lm_kwargs, signature, demos, inputs)
        except Exception as exc:
            if not self._should_use_json_fallback(exc):
                raise
            fallback_kwargs = self._same_parent_fallback_kwargs(
                lm_kwargs, parent_response_id
            )
            return self._call_json_fallback(
                lm, fallback_kwargs, signature, demos, inputs
            )

    async def _acall_with_json_fallback(
        self,
        lm: dspy.LM,
        lm_kwargs: dict[str, Any],
        signature: type[dspy.Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        parent_response_id = self._parent_response_id_for_fallback(lm, lm_kwargs)
        try:
            return await self._acall_primary(
                lm, lm_kwargs, signature, demos, inputs
            )
        except Exception as exc:
            if not self._should_use_json_fallback(exc):
                raise
            fallback_kwargs = self._same_parent_fallback_kwargs(
                lm_kwargs, parent_response_id
            )
            return await self._acall_json_fallback(
                lm, fallback_kwargs, signature, demos, inputs
            )

    def __call__(
        self,
        lm: dspy.LM,
        lm_kwargs: dict[str, Any],
        signature: type[dspy.Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        return self._call_with_json_fallback(
            lm, lm_kwargs, signature, demos, inputs
        )

    async def acall(
        self,
        lm: dspy.LM,
        lm_kwargs: dict[str, Any],
        signature: type[dspy.Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        return await self._acall_with_json_fallback(
            lm, lm_kwargs, signature, demos, inputs
        )


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


def _root_model_name_for_routing(lm: Any) -> str:
    raw_model_name = str(getattr(lm, "model", "") or "").strip().lower()
    if not raw_model_name:
        return ""
    if "/" in raw_model_name:
        return raw_model_name.rsplit("/", maxsplit=1)[-1]
    return raw_model_name


def _is_gpt5_family_root_model(lm: Any) -> bool:
    return _root_model_name_for_routing(lm).startswith("gpt-5")


class PromptOptimizationSignature(dspy.Signature):
    """You are optimizing prompts in an LLM program. The program processes instances through one or more steps. Each step calls an LLM with a prompt that tells it what to do.

    ## Goal
    - Achieve the highest score you can on train and validation, while ensuring these improvements generalize to a held-out test set. 
    - Maximize score on both splits. A large gap between them suggests overfitting.
    - Use validation checkpoints throughout optimization when train-side changes look meaningfully better.

    ## Environment
    - You are working in a persistent Python REPL for this run: code executed in previous iterations (if any) may have already created imports, variables, and functions that are still available now, and anything you define successfully will remain available in later iterations.
    - Treat the REPL as a workspace. Use it to achieve your goal and build momentum across iterations, not just to “run the next snippet”.
    - Printed output may be truncated for display; keep important artifacts in variables so they remain available.

    ## Running experiments
    - Use `evaluate_program()` to test your changes.
    - Train split (`split='train'`) for fast iteration and diagnostics:
        - `limit`: Evaluate a subset (e.g., `limit=15`).
        - `sample`: Use `sample='random'` for random sampling; use `sample_seed` for reproducibility.
        - `failed_from_run`: Re-evaluate only instances that failed in a previous run (pass the run_id).
        - `ids`: Target specific instances by ID (e.g., `ids='3,7,12'` or `ids='10-20'`).
    - Validation split (`split='val'`) for checkpointing generalization:
        - It always evaluates the full validation set.
        - Any selector args passed with `split='val'` are ignored.
    - Use `run_data(run_id)` to re-read previous evaluations at no budget cost.
    
    ## Existing baseline runs
    - `unoptimized_baseline_summary` includes baseline train and validation run IDs, scores, and sizes.
    - Before spending budget on new evaluations, inspect the baseline train run first with `run_data(<baseline_train_run_id>)`.

    ## Diagnosing failures
    - Evaluations return per-step traces showing what each step received as input and what it produced as output. This is your primary diagnostic tool. When the final output is wrong, trace back through the steps to find where the error first appeared.
    - In multi-step programs, errors cascade: a bad prompt in step 1 produces flawed output that causes step 2 to fail, which causes step 3 to fail. Fix the root cause, not the downstream symptoms.

    ### Using the sub-LM functions (`llm_query`, `llm_query_batched`)
    - You have two helper functions:
        - `llm_query(prompt)`
        - `llm_query_batched([prompt1, prompt2, ...])`
    - These functions let you query another LLM for anything you need to do where code alone isn't sufficient or doing it manually would pollute your context window. Tasks such as reading, analyzing, or summarzing traces, steps, examples and failures.
    - Include all necessary context in each prompt since the sub-LM has no access to your REPL state or conversation history.
    - Calls are composable: you can chain them, feeding one call's output into the next, building analysis in stages.

    ## Budget
    - Each instance evaluated costs one budget unit.
    - Each `llm_query` call also costs one budget unit; each prompt in a `llm_query_batched` batch costs one unit.
    - Every `evaluate_program()` result includes `remaining_budget`; treat that as your primary budget signal and keep track of it as you iterate.
    - A requested evaluation fails if its required instances exceed remaining budget.
    - `split='val'` always runs the full validation set, so its cost is the full val-set size.
    - `split='train'` cost equals the number of selected train instances.
    - When budget is getting low, do an explicit affordability check before any evaluation call (`train` or `val`).
    - When evaluation budget reaches zero, no more evaluations can run, but you may still continue analysis and decide which prompt(s) you would like to submit.

    ## Output Format
    - `code` output field: Python code to execute. Use markdown code block format: ```python\\n<code>\\n```.
    - `reasoning` output field: plain text only (no markdown code fences).
    
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
        rlm_multiturn_history: bool = False,
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
        self._rlm_multiturn_history = bool(rlm_multiturn_history)
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

    def _multiturn_adapter_for_root_lm(self, root_lm_for_run: Any) -> Any:
        if _is_gpt5_family_root_model(root_lm_for_run):
            return _RLMMultiTurnJSONFallbackAdapter()
        return _RLMMultiTurnHistoryAdapter()

    def run(
        self,
        kernel: OptimizationKernel,
    ) -> dict[str, Any]:
        self._emit_event(
            "session_started",
            {
                "max_iterations": self._max_iterations,
                "max_llm_calls": self._max_llm_calls,
                "rlm_multiturn_history": self._rlm_multiturn_history,
            },
        )
        tools = OptimizationTools(kernel, event_callback=self._event_callback)
        root_lm_for_run = self._build_root_lm_for_run()
        root_lm = BudgetMeteredLM(
            lm=root_lm_for_run,
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
            context_kwargs: dict[str, Any] = {"lm": root_lm}
            if self._rlm_multiturn_history:
                context_kwargs["adapter"] = self._multiturn_adapter_for_root_lm(
                    root_lm_for_run
                )

            with dspy.context(**context_kwargs):
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
