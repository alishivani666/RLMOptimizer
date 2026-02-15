from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import inspect
import random
from dataclasses import dataclass
from typing import Any, Callable, Literal, Sequence

import dspy

SampleMode = Literal["first", "random"]


@dataclass(frozen=True)
class DatasetRow:
    split: str
    example_id: str
    source_example_id: str | None
    example: dspy.Example


def _safe_json(value: Any, _seen: set[int] | None = None) -> Any:
    if _seen is None:
        _seen = set()

    value_id = id(value)
    if value_id in _seen:
        return "<circular_ref>"

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    if isinstance(value, dict):
        _seen.add(value_id)
        try:
            return {str(k): _safe_json(v, _seen) for k, v in value.items()}
        finally:
            _seen.discard(value_id)

    if isinstance(value, (list, tuple, set)):
        _seen.add(value_id)
        try:
            return [_safe_json(item, _seen) for item in value]
        finally:
            _seen.discard(value_id)

    if hasattr(value, "toDict"):
        return _safe_json(value.toDict(), _seen)

    return str(value)


def extract_source_example_id(example: dspy.Example) -> str | None:
    for key in ("example_id", "id", "__example_id", "_example_id"):
        value = example.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def build_dataset_rows(
    trainset: Sequence[dspy.Example],
    valset: Sequence[dspy.Example] | None,
) -> dict[str, list[DatasetRow]]:
    rows: dict[str, list[DatasetRow]] = {"train": [], "val": []}

    for index, example in enumerate(trainset, start=1):
        if not isinstance(example, dspy.Example):
            raise TypeError("trainset entries must be dspy.Example")
        rows["train"].append(
            DatasetRow(
                split="train",
                example_id=str(index),
                source_example_id=extract_source_example_id(example),
                example=example,
            )
        )

    if valset:
        for index, example in enumerate(valset, start=1):
            if not isinstance(example, dspy.Example):
                raise TypeError("valset entries must be dspy.Example")
            rows["val"].append(
                DatasetRow(
                    split="val",
                    example_id=str(index),
                    source_example_id=extract_source_example_id(example),
                    example=example,
                )
            )

    return rows


def parse_ids(spec: str | None) -> list[str] | None:
    if spec is None:
        return None

    selected: list[str] = []
    for token in spec.split(","):
        part = token.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            if start.isdigit() and end.isdigit():
                for value in range(int(start), int(end) + 1):
                    selected.append(str(value))
                continue
        selected.append(part)

    return selected


def select_rows(
    rows: Sequence[DatasetRow],
    *,
    ids: list[str] | None,
    limit: int | None,
    sample: SampleMode,
    sample_seed: int | None,
) -> list[DatasetRow]:
    selected = list(rows)

    if ids:
        id_set = {str(item) for item in ids}
        selected = [row for row in selected if row.example_id in id_set]

    if limit is None:
        return selected

    if limit <= 0:
        raise ValueError("limit must be greater than 0")

    take = min(limit, len(selected))
    if sample == "random":
        rng = random.Random(sample_seed)
        indexes = sorted(rng.sample(range(len(selected)), take))
        return [selected[index] for index in indexes]

    return selected[:take]


def _extract_expected(example: dspy.Example) -> dict[str, Any]:
    labels = example.labels()
    return _safe_json(dict(labels))


def _prediction_to_json(prediction: dspy.Prediction) -> dict[str, Any]:
    return _safe_json(dict(prediction))


def _trace_to_steps(trace: list[tuple[Any, dict[str, Any], Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, event in enumerate(trace):
        module, inputs, outputs = event

        signature_name = None
        signature = getattr(module, "signature", None)
        if signature is not None:
            name = getattr(signature, "__name__", None)
            if isinstance(name, str) and name != "StringSignature":
                signature_name = name

        step_name = signature_name or module.__class__.__name__

        rows.append(
            {
                "step_index": index,
                "step_name": step_name,
                "signature_name": signature_name,
                "inputs": _safe_json(inputs),
                "outputs": _safe_json(dict(outputs) if isinstance(outputs, dspy.Prediction) else outputs),
            }
        )

    return rows


def _metric_accepts_trace(metric: Callable[..., Any]) -> bool:
    signature = inspect.signature(metric)
    params = list(signature.parameters.values())
    if any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in params):
        return True
    positional = [
        param
        for param in params
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    return len(positional) >= 3


def _metric_score(metric: Callable[..., Any], example: dspy.Example, prediction: dspy.Prediction, trace: list[Any]) -> float:
    if _metric_accepts_trace(metric):
        raw = metric(example, prediction, trace)
    else:
        raw = metric(example, prediction)

    if isinstance(raw, bool):
        return 1.0 if raw else 0.0

    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"metric returned non-numeric score: {raw!r}") from exc


def _evaluate_single_row(
    *,
    program: dspy.Module,
    row: DatasetRow,
    metric: Callable[..., Any],
    eval_lm: Any,
) -> tuple[dict[str, Any], float, bool]:
    trace: list[tuple[Any, dict[str, Any], Any]] = []
    inputs = row.example.inputs()
    predicted: dict[str, Any]
    score: float
    passed: bool

    try:
        if eval_lm is None:
            context = dspy.context(trace=trace, adapter=None)
        else:
            context = dspy.context(lm=eval_lm, trace=trace, adapter=None)

        with context:
            prediction = program(**inputs)

        score = _metric_score(metric, row.example, prediction, trace)
        passed = bool(score == 1.0)
        predicted = _prediction_to_json(prediction)
    except KeyboardInterrupt:
        raise
    except Exception:  # pragma: no cover - provider/network/runtime failures.
        # Treat transient runtime/provider failures as failed examples so a run
        # can continue and still return actionable diagnostics.
        score = 0.0
        passed = False
        predicted = {}

    record = {
        "example_id": row.example_id,
        "source_example_id": row.source_example_id,
        "inputs": _safe_json(dict(inputs)),
        "expected": _extract_expected(row.example),
        "predicted": predicted,
        "score": float(score),
        "passed": passed,
        "steps": _trace_to_steps(trace),
    }
    return record, float(score), passed


def evaluate_rows(
    *,
    program: dspy.Module,
    rows: Sequence[DatasetRow],
    metric: Callable[..., Any],
    eval_lm: Any,
    split: str,
    config: dict[str, Any],
    num_threads: int = 1,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("no examples selected for evaluation")
    if num_threads <= 0:
        raise ValueError("num_threads must be greater than 0")

    records: list[dict[str, Any]] = []
    total_score = 0.0
    passed_count = 0
    total_rows = len(rows)

    if progress_callback is not None:
        try:
            progress_callback(
                {
                    "stage": "evaluation_started",
                    "completed": 0,
                    "total": total_rows,
                }
            )
        except Exception:  # pragma: no cover - diagnostics callback must not break eval.
            pass

    if num_threads == 1:
        for index, row in enumerate(rows, start=1):
            if progress_callback is not None:
                try:
                    progress_callback(
                        {
                            "stage": "example_started",
                            "completed": len(records),
                            "total": total_rows,
                            "current_index": index,
                            "example_id": row.example_id,
                        }
                    )
                except Exception:  # pragma: no cover - diagnostics callback must not break eval.
                    pass

            record, score, passed = _evaluate_single_row(
                program=program,
                row=row,
                metric=metric,
                eval_lm=eval_lm,
            )
            records.append(record)
            passed_count += int(passed)
            total_score += score

            if progress_callback is not None:
                try:
                    progress_callback(
                        {
                            "stage": "example_finished",
                            "completed": len(records),
                            "total": total_rows,
                            "current_index": index,
                            "example_id": row.example_id,
                            "score": float(score),
                            "passed": passed,
                        }
                    )
                except Exception:  # pragma: no cover - diagnostics callback must not break eval.
                    pass
    else:
        ordered_records: list[dict[str, Any] | None] = [None] * total_rows
        completed = 0
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_meta = {
                executor.submit(
                    _evaluate_single_row,
                    program=program,
                    row=row,
                    metric=metric,
                    eval_lm=eval_lm,
                ): (index, row.example_id)
                for index, row in enumerate(rows, start=1)
            }

            for future in as_completed(future_to_meta):
                index, example_id = future_to_meta[future]
                record, score, passed = future.result()
                ordered_records[index - 1] = record
                passed_count += int(passed)
                total_score += score
                completed += 1

                if progress_callback is not None:
                    try:
                        progress_callback(
                            {
                                "stage": "example_finished",
                                "completed": completed,
                                "total": total_rows,
                                "current_index": index,
                                "example_id": example_id,
                                "score": float(score),
                                "passed": passed,
                            }
                        )
                    except Exception:  # pragma: no cover - diagnostics callback must not break eval.
                        pass

        records = [record for record in ordered_records if record is not None]

    score_percent = round(100 * total_score / len(rows), 2)

    return {
        "split": split,
        "score": score_percent,
        "evaluated_count": len(rows),
        "passed_count": passed_count,
        "examples": records,
        "config": config,
    }
