from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar

import dspy

from .kernel import OptimizationKernel
from .types import BudgetExceededError, InstructionUpdateError, UnknownRunError

_F = TypeVar("_F", bound=Callable[..., Any])

_TOOL_ERRORS = (InstructionUpdateError, UnknownRunError, TypeError, ValueError)


def _tool_method(func: _F) -> _F:
    """Wrap a tool method so user-facing errors become ``{"error": msg}`` dicts.

    ``BudgetExceededError`` is intentionally re-raised because the caller
    (the RLM agent loop) must see it as a hard stop.
    """

    @functools.wraps(func)
    def wrapper(self: OptimizationTools, *args: Any, **kwargs: Any) -> Any:
        try:
            return func(self, *args, **kwargs)
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
        return ids
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
            raise ValueError(f"{field_name} must not be empty.")
        return text
    raise TypeError(f"{field_name} must be a run ID string or integer.")


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
        return result

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
                desc=(
                    "Run the program on dataset instances and score the outputs. Costs one budget unit "
                    "per instance. Returns: score (0-100%), evaluated_count, passed_count, run_id, "
                    "remaining_budget, and examples (list of per-instance results). Each entry has: "
                    "example_id, inputs, expected, predicted, score, passed, error_text, and steps—"
                    "a trace of what each step received and produced. Results are stored; re-read via "
                    "run_data(run_id) for free."
                ),
                arg_desc={
                    "split": "Which dataset: 'train' for experimentation, 'val' for final validation.",
                    "limit": "Max instances to evaluate. Use small values (10-30) for quick iteration. Null evaluates all.",
                    "ids": "Evaluate specific instances by ID: '1,2,8' or '10-20' or a list. Null for no filtering.",
                    "sample": "How to select when using limit: 'first' (default) takes first N, 'random' samples randomly.",
                    "sample_seed": "Seed for reproducible random sampling. Null for different sample each time.",
                    "failed_from_run": "Re-run only instances that failed in this run_id. Most budget-efficient for testing fixes.",
                },
            ),
            dspy.Tool(
                self.run_data,
                desc=(
                    "Re-read a previous evaluation's full results by run_id. Same structure as "
                    "evaluate_program returns. Costs no budget—use freely to analyze past runs."
                ),
                arg_desc={
                    "run_id": "The run_id from a previous evaluate_program call.",
                },
            ),
            dspy.Tool(
                self.update_prompt,
                desc=(
                    "Rewrite the prompt for one step. Provide the complete new prompt text, not a patch. "
                    "Returns confirmation on success or an error message on failure."
                ),
                arg_desc={
                    "step_name": "Which step to update. Must be a name from optimization_status()['steps'].",
                    "new_text": "The complete new prompt text for this step.",
                },
            ),
            dspy.Tool(
                self.optimization_status,
                desc=(
                    "Get current optimization state: remaining_budget, best_score, best_run_id, "
                    "latest_run_id, current_prompts (step name → prompt text), best_prompts (from "
                    "best-scoring run), and steps (list of step names for update_prompt)."
                ),
            ),
        ]
