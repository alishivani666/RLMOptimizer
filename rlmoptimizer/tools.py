from __future__ import annotations

import functools
import textwrap
from typing import Any, Callable, TypeVar

import dspy

from .kernel import OptimizationKernel
from .types import BudgetExceededError, InstructionUpdateError, UnknownRunError

_F = TypeVar("_F", bound=Callable[..., Any])

_TOOL_ERRORS = (InstructionUpdateError, UnknownRunError, TypeError, ValueError)


class _DisplayPayload(dict[str, Any]):
    """Dict payload with a concise print representation for REPL readability."""

    def __init__(self, *args: Any, display_text: str | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.display_text = str(display_text or "")

    def __str__(self) -> str:
        if self.display_text:
            return self.display_text
        return dict.__repr__(self)


def _sanitize_tool_payload(value: Any) -> Any:
    """Normalize payload values so Pyodide REPL code receives Python-native types.

    Pyodide maps JSON ``null`` values to ``JsNull`` by default. Downstream RLM
    analysis code often slices/indices payload values and can crash with
    ``'JsNull' object is not subscriptable``. We convert ``None`` to empty
    strings at the tool boundary to keep payloads string-safe and avoid runtime
    crashes inside the interpreter.
    """
    if value is None:
        return ""
    if isinstance(value, _DisplayPayload):
        return _DisplayPayload(
            {str(k): _sanitize_tool_payload(v) for k, v in value.items()},
            display_text=value.display_text,
        )
    if isinstance(value, dict):
        return {str(k): _sanitize_tool_payload(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_tool_payload(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_tool_payload(item) for item in value]
    if isinstance(value, set):
        return [_sanitize_tool_payload(item) for item in value]
    return value


def _tool_method(func: _F) -> _F:
    """Wrap a tool method so user-facing errors become ``{"error": msg}`` dicts.

    ``BudgetExceededError`` is intentionally re-raised because the caller
    (the RLM agent loop) must see it as a hard stop.
    """

    @functools.wraps(func)
    def wrapper(self: OptimizationTools, *args: Any, **kwargs: Any) -> Any:
        try:
            result = func(self, *args, **kwargs)
            return _sanitize_tool_payload(result)
        except BudgetExceededError:
            raise
        except _TOOL_ERRORS as exc:
            return {"error": str(exc)}

    return wrapper  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Normalization helpers (pure functions, no instance state)
# ---------------------------------------------------------------------------


def _normalize_optional_int(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"{field_name} must be an int, float, string integer, or null.")
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field_name} could not be converted to int. Use a numeric value like 5."
        ) from exc


def _normalize_ids(ids: str | int | list[str | int] | None) -> str | None:
    if ids is None:
        return None
    if isinstance(ids, bool):
        raise TypeError("ids must be a string, integer, list[str|int], or null.")
    if isinstance(ids, int):
        return str(ids)
    if isinstance(ids, str):
        text = ids.strip()
        if not text:
            return None
        return text
    if isinstance(ids, list):
        parts: list[str] = []
        for item in ids:
            if isinstance(item, bool):
                raise TypeError("ids list entries must be strings or integers.")
            elif isinstance(item, int):
                parts.append(str(item))
            elif isinstance(item, str):
                text = item.strip()
                if not text:
                    raise ValueError("ids list entries cannot be empty strings.")
                parts.append(text)
            else:
                raise TypeError("ids list entries must be strings or integers.")
        return ",".join(parts)
    raise TypeError("ids must be a string, integer, list[str|int], or null.")


def _normalize_sample(sample: str | None) -> str:
    if sample is None:
        return "first"
    if not isinstance(sample, str):
        raise TypeError("sample must be 'first', 'random', or null.")
    normalized = sample.strip().lower()
    if normalized == "":
        return "first"
    if normalized not in {"first", "random"}:
        raise ValueError("sample must be one of: first, random.")
    return normalized


def _normalize_run_id(
    run_id: str | int | None, *, field_name: str, required: bool = True
) -> str | None:
    if run_id is None:
        if required:
            raise ValueError(f"{field_name} must be provided.")
        return None
    if isinstance(run_id, bool):
        raise TypeError(f"{field_name} must be a run ID string or integer.")
    if isinstance(run_id, int):
        return str(run_id)
    if isinstance(run_id, str):
        text = run_id.strip()
        if not text:
            if required:
                raise ValueError(f"{field_name} must not be empty.")
            return None
        return text
    raise TypeError(f"{field_name} must be a run ID string or integer.")


def _tool_description(text: str) -> str:
    return textwrap.dedent(text).strip()


# ---------------------------------------------------------------------------
# Tool surface
# ---------------------------------------------------------------------------


class OptimizationTools:
    """Host-side tool surface exposed to the RLM interpreter.

    These tools are intentionally narrow:
    - evaluate targeted subsets and inspect detailed traces,
    - mutate only step prompts,
    - query current optimization state.

    Use :meth:`as_dspy_tools` when wiring these methods into ``dspy.RLM`` so
    argument-level descriptions are included in the generated tool schema.
    """

    def __init__(self, kernel: OptimizationKernel) -> None:
        self._kernel = kernel

    @_tool_method
    def evaluate_program(
        self,
        split: str = "train",
        limit: int | float | str | None = None,
        ids: str | int | list[str | int] | None = None,
        sample: str | None = "first",
        sample_seed: int | float | str | None = None,
        failed_from_run: str | int | None = None,
    ) -> dict[str, Any]:
        """Run a budgeted evaluation and return structured run diagnostics.

        Use this after one or more instruction edits when you need fresh score
        evidence. Do not use it to inspect existing runs; call ``run_data`` for
        that because ``run_data`` does not consume evaluation budget.

        This tool:
        - charges budget by the exact number of evaluated examples,
        - prints a one-line score/budget summary immediately,
        - returns per-example outputs and step traces,
        - persists the full payload so it can be retrieved later by run ID.

        Args:
            split: Dataset split to evaluate. Allowed values: ``train`` or ``val``.
            limit: Maximum number of selected examples to evaluate.
            ids: Optional ID selector string such as ``"1,2,8"`` or ``"10-20"``.
            sample: Selection strategy for ``limit``. ``first`` keeps order,
                ``random`` samples with replacement disabled.
            sample_seed: Optional random seed used only when ``sample=random``.
            failed_from_run: Optional previous run ID. When set, evaluate only
                examples that failed in that run.

        Returns:
            A dict with run metadata, score summary, selected examples, and traces.
        """
        return self._kernel.evaluate_program(
            split=split,
            limit=_normalize_optional_int(limit, field_name="limit"),
            ids=_normalize_ids(ids),
            sample=_normalize_sample(sample),
            sample_seed=_normalize_optional_int(sample_seed, field_name="sample_seed"),
            failed_from_run=_normalize_run_id(
                failed_from_run, field_name="failed_from_run", required=False
            ),
        )

    @_tool_method
    def run_data(self, run_id: str | int) -> dict[str, Any]:
        """Fetch a previously recorded run by ID.

        Use this to re-open historical runs for comparison or deeper analysis.
        This does not trigger re-evaluation and does not consume budget.
        This response omits ``remaining_budget``.

        Args:
            run_id: Run identifier returned by ``evaluate_program``.

        Returns:
            A dict with run-level metrics and per-example details.
        """
        return self._kernel.run_data(_normalize_run_id(run_id, field_name="run_id"))

    @_tool_method
    def update_prompt(self, step_name: str, new_text: str) -> dict[str, Any]:
        """Update exactly one step prompt.

        This is the only mutation tool. It enforces prompt-only edits via a
        structural fingerprint check and rejects any change outside
        ``predictor.signature.instructions``. Provide the full replacement text
        for one step; partial patch syntax is not supported.
        """
        if not isinstance(step_name, str):
            raise TypeError("step_name must be a string from optimization_status()['steps'].")
        if not isinstance(new_text, str):
            raise TypeError("new_text must be a non-empty string.")
        result = self._kernel.update_instruction(
            predictor_name=step_name,
            new_text=new_text,
        )
        if "predictor_name" in result:
            result["step_name"] = str(result.pop("predictor_name"))
        display_name = str(result.get("step_name", step_name))
        return _DisplayPayload(
            result,
            display_text=f'Prompt for "{display_name}" was successfully updated.',
        )

    @_tool_method
    def optimization_status(self) -> dict[str, Any]:
        """Return current optimization state.

        Includes budget remaining, current/best run IDs, best score so far, and
        both current and best prompt maps. Use this before/after edits to decide
        whether to keep exploring, revert, or finish.
        """
        status = self._kernel.optimization_status()
        return {
            "remaining_budget": status["remaining_budget"],
            "evaluated_examples": status["evaluated_examples"],
            "root_lm_calls": status["root_lm_calls"],
            "sub_lm_calls": status["sub_lm_calls"],
            "num_threads": status["num_threads"],
            "best_score": status["best_score"],
            "best_run_id": status["best_run_id"],
            "latest_run_id": status["latest_run_id"],
            "baseline_run_id": status["baseline_run_id"],
            "current_prompts": dict(status["current_instructions"]),
            "best_prompts": dict(status["best_instructions"]),
            "steps": sorted(status["predictors"]),
        }

    def as_dspy_tools(self) -> list[dspy.Tool]:
        return [
            dspy.Tool(
                self.evaluate_program,
                desc=_tool_description(
                    """
                    Run the program on dataset instances and score outputs.

                    Costs one budget unit per instance evaluated.

                    Returns a dict:
                    {
                        "score": "float. Percentage of instances that passed, 0-100 scale.",
                        "evaluated_count": "int. How many instances were evaluated.",
                        "passed_count": "int. How many instances passed (where score=1.0).",
                        "run_id": "str. Unique identifier for this run. Pass to run_data() to re-read results later.",
                        "remaining_budget": "int. Budget units remaining after this evaluation.",
                        "summary_line": "str. Human-readable one-line summary of results.",
                        "split": "str. Which dataset was evaluated: 'train' or 'val'.",
                        "config": "dict. The arguments you passed to this call.",
                        "examples": "list[dict]. Per-instance results. One dict per evaluated instance:"
                        [
                            {
                                "example_id": "str. Identifier for this instance.",
                                "source_example_id": "str or null. Original ID from your dataset, if provided.",
                                "inputs": "dict. Input fields that were passed to the program.",
                                "expected": "dict. Expected output fields (the ground truth).",
                                "predicted": "dict. Output fields the program actually produced.",
                                "score": "float. This instance's score on 0.0-1.0 scale. NOTE: This is NOT 0-100 like the top-level score!",
                                "passed": "bool. True if this instance passed (score=1.0).",
                                "error_text": "str or null. If the program crashed on this instance, the error message. Otherwise null.",
                                "steps": "list[dict]. Execution trace showing what each step received and produced:"
                                [
                                    {
                                        "step_index": "int. Position in execution order, starting from 0.",
                                        "step_name": "str. Identifier for this step. THIS IS WHAT YOU PASS TO update_prompt().",
                                        "signature_name": "str or null. The signature class name, if available.",
                                        "inputs": "dict. What this step received as input.",
                                        "outputs": "dict. What this step produced as output."
                                    }
                                ]
                            }
                        ]
                    }

                    The steps list is your primary diagnostic tool. When an instance fails, trace through the steps to find where the error originated. Check: Did step 0 receive correct inputs but produce wrong outputs? Did step 1 receive wrong inputs from step 0? Find the first step that went wrong.
                    """
                ),
                arg_desc={
                    "split": "str. 'train' or 'val'. Use 'train' for experimentation. Use 'val' only for final validation to confirm your improvements generalize.",
                    "limit": "int or null. Maximum instances to evaluate. Use small values like 10-20 for quick experiments. Null evaluates all instances.",
                    "ids": "str or list or null. Evaluate only specific instances. String like '1,2,8' or '10-20', or list like [1, 2, 8]. Null means no filtering.",
                    "sample": "str. 'first' takes the first N instances in order. 'random' takes a random sample, which better represents the full dataset.",
                    "sample_seed": "int or null. Seed for reproducible random sampling. Null gives a different random sample each time. Only used when sample='random'.",
                    "failed_from_run": "str or null. Pass a run_id to evaluate ONLY instances that failed in that run. Most budget-efficient way to test if your changes fixed the failures.",
                },
            ),
            dspy.Tool(
                self.run_data,
                desc=_tool_description(
                    """
                    Retrieve the full results of a previous evaluation. Costs NO budget.

                    Returns a dict with the same structure as evaluate_program:
                    {
                        "score": "float. Percentage 0-100.",
                        "evaluated_count": "int.",
                        "passed_count": "int.",
                        "run_id": "str.",
                        "summary_line": "str.",
                        "split": "str.",
                        "config": "dict.",
                        "examples": "list[dict]. Same structure as evaluate_program:"
                        [
                            {
                                "example_id": "str.",
                                "source_example_id": "str or null.",
                                "inputs": "dict.",
                                "expected": "dict.",
                                "predicted": "dict.",
                                "score": "float. 0.0-1.0 scale.",
                                "passed": "bool.",
                                "error_text": "str or null.",
                                "steps": "list[dict]:"
                                [
                                    {
                                        "step_index": "int.",
                                        "step_name": "str.",
                                        "signature_name": "str or null.",
                                        "inputs": "dict.",
                                        "outputs": "dict."
                                    }
                                ]
                            }
                        ]
                    }

                    Note: remaining_budget is NOT included because it was a snapshot from when that run happened and would be misleading now.

                    Use this freely to re-examine old runs, compare results, or analyze failures without spending budget.
                    """
                ),
                arg_desc={
                    "run_id": "str. The run_id from a previous evaluate_program call.",
                },
            ),
            dspy.Tool(
                self.update_prompt,
                desc=_tool_description(
                    """
                    Rewrite the prompt for one step in the program.

                    You must provide the complete new prompt text - this is a full replacement, not a patch.

                    On success, returns a dict:
                    {
                        "status": "str. Always 'ok' on success.",
                        "step_name": "str. The step that was updated.",
                        "prompt_hash": "str. SHA256 hash of the new prompt text.",
                        "prompt_preview": "str. First 200 characters of the new prompt."
                    }

                    On failure, returns a dict:
                    {
                        "error": "str. Description of what went wrong."
                    }
                    """
                ),
                arg_desc={
                    "step_name": "str. Which step to update. Get valid names from evaluation results at examples[i]['steps'][j]['step_name'], or from optimization_status()['steps'].",
                    "new_text": "str. The complete new prompt text. This entirely replaces the current prompt for this step.",
                },
            ),
            dspy.Tool(
                self.optimization_status,
                desc=_tool_description(
                    """
                    Get the current state of optimization. Costs NO budget.

                    Returns a dict:
                    {
                        "remaining_budget": "int. How many budget units you have left.",
                        "evaluated_examples": "int. Total instances evaluated across all runs so far.",
                        "root_lm_calls": "int. How many LLM calls the optimizer (you) has made.",
                        "sub_lm_calls": "int. How many LLM calls made via llm_query/llm_query_batched.",
                        "num_threads": "int. Number of parallel threads used for evaluation.",
                        "best_score": "float. Highest score achieved so far, 0-100 scale.",
                        "best_run_id": "str or null. run_id of the evaluation that achieved best_score. Null if no runs yet.",
                        "latest_run_id": "str or null. run_id of the most recent evaluation. Null if no runs yet.",
                        "baseline_run_id": "str or null. run_id of the initial baseline evaluation before any changes.",
                        "current_prompts": "dict[str, str]. Maps step_name to current prompt text for each step.",
                        "best_prompts": "dict[str, str]. Maps step_name to prompt text from the best-scoring run.",
                        "steps": "list[str]. Step names you can pass to update_prompt()."
                    }

                    Use current_prompts to see what prompts are currently active.
                    Use best_prompts to see what prompts achieved the best score.
                    Use steps to know what values you can pass to update_prompt().
                    """
                ),
            ),
        ]
