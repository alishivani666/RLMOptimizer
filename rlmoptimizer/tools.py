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
    - mutate only predictor instructions,
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
    def update_instruction(self, predictor_name: str, new_text: str) -> dict[str, Any]:
        """Update exactly one predictor's signature instruction text.

        This is the only mutation tool. It enforces instruction-only edits via a
        structural fingerprint check and rejects any change outside
        ``predictor.signature.instructions``. Provide the full replacement text
        for one predictor; partial patch syntax is not supported.

        Args:
            predictor_name: Predictor key from ``optimization_status()['predictors']``.
            new_text: New instruction text for that predictor.

        Returns:
            A dict containing status, predictor name, and instruction hash/preview.
        """
        if not isinstance(predictor_name, str):
            raise TypeError(
                "predictor_name must be a string from optimization_status()['predictors']."
            )
        if not isinstance(new_text, str):
            raise TypeError("new_text must be a non-empty string.")
        return self._kernel.update_instruction(
            predictor_name=predictor_name,
            new_text=new_text,
        )

    @_tool_method
    def optimization_status(self) -> dict[str, Any]:
        """Return current optimization state.

        Includes budget remaining, current/best run IDs, best score so far, and
        both current and best instruction maps. Use this before/after edits to
        decide whether to keep exploring, revert, or finish.
        """
        return self._kernel.optimization_status()

    def as_dspy_tools(self) -> list[dspy.Tool]:
        """Return tools with explicit argument descriptions for agent reliability."""
        return [
            dspy.Tool(
                self.evaluate_program,
                desc=(
                    "Run the program on dataset examples and score outputs. Consumes budget "
                    "proportional to the number of examples evaluated. Returns a dict with: "
                    "score (0-100%), evaluated_count, passed_count, run_id, remaining_budget, "
                    "summary_line, and examples — a list where each entry contains example_id, "
                    "inputs, expected, predicted, score, passed, error_text, and steps (a list "
                    "of per-predictor traces with step_index, step_name, inputs, and outputs "
                    "showing what each predictor in the pipeline received and produced). "
                    "The run is stored and can be re-read later via run_data(run_id) for free; "
                    "run_data omits remaining_budget. "
                    "Large payloads may be truncated; use llm_query/llm_query_batched on the "
                    "data for deeper semantic analysis."
                ),
                arg_desc={
                    "split": "Dataset split: 'train' or 'val'.",
                    "limit": "Max examples to evaluate. Use small values (e.g. 10-30) for quick iteration. Null evaluates all.",
                    "ids": "Evaluate specific examples: string like '1,2,8' or '10-20', int, list of str/int, or null for all.",
                    "sample": "'first' (default) keeps order, 'random' samples randomly. Only matters with limit.",
                    "sample_seed": "Random seed when sample='random'. Null for non-deterministic.",
                    "failed_from_run": "Run ID — re-evaluate only examples that failed in that run. Very budget-efficient for iteration.",
                },
            ),
            dspy.Tool(
                self.run_data,
                desc=(
                    "Fetch a previously stored run by run_id. Returns the same dict structure as "
                    "evaluate_program except that remaining_budget is omitted. "
                    "Does NOT consume any budget — use this freely to re-read and analyze old runs."
                ),
                arg_desc={
                    "run_id": "Run ID returned by evaluate_program, as string or integer.",
                },
            ),
            dspy.Tool(
                self.update_instruction,
                desc=(
                    "Replace one predictor's instruction text. Provide the complete new instruction "
                    "(not a patch). Returns {status, predictor_name, instruction_hash, "
                    "instruction_preview} on success, or {error: message} on failure."
                ),
                arg_desc={
                    "predictor_name": "Predictor to update; must be a name from optimization_status()['predictors'].",
                    "new_text": "Complete replacement instruction text for that predictor.",
                },
            ),
            dspy.Tool(
                self.optimization_status,
                desc=(
                    "Returns current optimization state as a dict with: remaining_budget, "
                    "evaluated_examples, best_score, best_run_id, latest_run_id, baseline_run_id, "
                    "current_instructions (dict mapping predictor name → current instruction text), "
                    "best_instructions (same mapping for best-scoring version), and predictors "
                    "(list of predictor names you can pass to update_instruction)."
                ),
            ),
        ]
