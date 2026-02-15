from __future__ import annotations

import json
import threading
import tempfile
from pathlib import Path
from typing import Any, Callable, Literal
from uuid import uuid4

import dspy

from .evaluator import DatasetRow, build_dataset_rows, evaluate_rows, parse_ids, select_rows
from .fingerprint import apply_prompt_map, prompt_map, structure_hash
from .progress import ProgressReporter, create_progress_reporter
from .types import (
    BudgetExceededError,
    PromptUpdateError,
    OptimizationKernelState,
    RunMeta,
    UnknownRunError,
)

SampleMode = Literal["first", "random"]


def _sha256_text(text: str) -> str:
    import hashlib

    digest = hashlib.sha256()
    digest.update(text.encode("utf-8"))
    return digest.hexdigest()


class OptimizationKernel:
    def __init__(
        self,
        *,
        program: dspy.Module,
        trainset: list[dspy.Example],
        valset: list[dspy.Example] | None,
        metric: Callable[..., Any],
        eval_lm: Any,
        num_threads: int,
        max_iterations: int,
        max_output_chars: int,
        run_storage_dir: Path | None = None,
        debug_display: Any | None = None,
        event_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if max_iterations <= 0:
            raise ValueError("max_iterations must be greater than zero")
        if not trainset:
            raise ValueError("trainset must not be empty")
        if num_threads <= 0:
            raise ValueError("num_threads must be greater than zero")

        self.program = program
        self.metric = metric
        self.eval_lm = eval_lm
        self.num_threads = int(num_threads)
        self.max_output_chars = int(max_output_chars)
        val_size = len(valset) if valset is not None else 0
        self.max_budget = int(max_iterations * len(trainset) + val_size)
        self.dataset_rows = build_dataset_rows(trainset, valset)
        self._budget_lock = threading.Lock()
        self._event_callback = event_callback

        self._storage_tempdir: tempfile.TemporaryDirectory[str] | None = None
        if run_storage_dir is None:
            self._storage_tempdir = tempfile.TemporaryDirectory(prefix="rlmoptimizer-runs-")
            self.run_storage_dir = Path(self._storage_tempdir.name)
        else:
            self.run_storage_dir = run_storage_dir
            self.run_storage_dir.mkdir(parents=True, exist_ok=True)

        initial_prompt_map = prompt_map(self.program)
        self.state = OptimizationKernelState(
            remaining_budget=self.max_budget,
            current_prompt_map=dict(initial_prompt_map),
            best_prompt_map=dict(initial_prompt_map),
        )
        self._baseline_structure_hash = structure_hash(self.program)
        self._debug_display = debug_display

    def _emit_event(self, event_type: str, payload: dict[str, Any]) -> None:
        callback = self._event_callback
        if callback is None:
            return
        try:
            callback({"source": "kernel", "event": event_type, **payload})
        except Exception:
            pass

    def _best_tracking_split(self) -> str:
        """Track best runs on full val when available, else full train."""
        return "val" if self.dataset_rows["val"] else "train"

    def _is_eligible_best_run(
        self,
        *,
        split: str,
        selected_rows: list[DatasetRow],
        selected_ids: list[str] | None,
        failed_from_run: str | None,
    ) -> bool:
        if split != self._best_tracking_split():
            return False
        if failed_from_run is not None:
            return False
        if selected_ids:
            return False
        return len(selected_rows) == len(self.dataset_rows[split])

    def close(self) -> None:
        if self._storage_tempdir is not None:
            self._storage_tempdir.cleanup()
            self._storage_tempdir = None

    def run_baseline(self) -> dict[str, Any]:
        self._emit_event(
            "baseline_started",
            {
                "train_size": len(self.dataset_rows["train"]),
                "val_size": len(self.dataset_rows["val"]),
            },
        )
        train_payload = self._evaluate_program_raw(
            split="train",
            charge_budget=False,
            phase="baseline_train",
        )
        baseline_payload = train_payload
        val_payload: dict[str, Any] | None = None

        # When val exists, initialize best-tracking on the same split used for
        # best-run selection so comparisons stay val-to-val.
        if self._best_tracking_split() == "val":
            val_payload = self._evaluate_program_raw(
                split="val",
                charge_budget=False,
                phase="baseline_val",
            )
            baseline_payload = val_payload

        self.state.baseline_run_id = str(baseline_payload["run_id"])
        self._emit_event(
            "baseline_completed",
            {
                "baseline_run_id": self.state.baseline_run_id,
                "baseline_split": baseline_payload.get("split"),
                "train_run_id": str(train_payload.get("run_id", "")),
                "val_run_id": (
                    str(val_payload.get("run_id", ""))
                    if val_payload is not None
                    else None
                ),
            },
        )
        return self._public_payload_view(baseline_payload)

    def _run_path(self, run_id: str) -> Path:
        return self.run_storage_dir / f"{run_id}.json"

    def _new_run_id(self) -> str:
        return uuid4().hex[:8]

    def _run_meta(self, run_id: str) -> RunMeta:
        meta = self.state.runs.get(run_id)
        if meta is None:
            raise UnknownRunError(f"unknown run id: {run_id}")
        return meta

    def _load_run_payload(self, run_id: str) -> dict[str, Any]:
        meta = self._run_meta(run_id)
        return json.loads(meta.storage_path.read_text(encoding="utf-8"))

    def _store_run_payload(self, run_id: str, payload: dict[str, Any]) -> Path:
        path = self._run_path(run_id)
        path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
        return path

    def _failed_ids_for_run(self, run_id: str) -> list[str]:
        payload = self._load_run_payload(run_id)
        return [
            str(record.get("example_id"))
            for record in payload.get("examples", [])
            if not bool(record.get("passed"))
        ]

    def _validate_failed_from_run_split(
        self,
        *,
        split: str,
        failed_from_run: str | None,
    ) -> None:
        if failed_from_run is None:
            return
        source_meta = self._run_meta(failed_from_run)
        if source_meta.split != split:
            raise ValueError(
                "failed_from_run must reference a run from the same split; "
                f"requested split={split!r}, run split={source_meta.split!r}."
            )

    def _validate_public_evaluation_request(
        self,
        *,
        split: str,
        limit: int | None,
        ids: str | None,
        sample: SampleMode,
        sample_seed: int | None,
        failed_from_run: str | None,
    ) -> None:
        if split == "val":
            if limit is not None:
                raise ValueError(
                    "split='val' only supports full-split evaluation; limit must be null."
                )
            if ids is not None:
                raise ValueError(
                    "split='val' only supports full-split evaluation; ids must be null."
                )
            if failed_from_run is not None:
                raise ValueError(
                    "split='val' only supports full-split evaluation; failed_from_run must be null."
                )
            if sample_seed is not None:
                raise ValueError(
                    "split='val' only supports full-split evaluation; sample_seed must be null."
                )
            if sample != "first":
                raise ValueError(
                    "split='val' only supports full-split evaluation; sample must be 'first'."
                )

        self._validate_failed_from_run_split(
            split=split,
            failed_from_run=failed_from_run,
        )

    def _canonicalize_public_evaluation_request(
        self,
        *,
        split: str,
        limit: int | None,
        ids: str | None,
        sample: SampleMode,
        sample_seed: int | None,
        failed_from_run: str | None,
    ) -> tuple[int | None, str | None, SampleMode, int | None, str | None]:
        """Canonicalize public evaluate_program args while preserving split semantics.

        Validation runs are always full-split. For split='val', train-only
        selectors are ignored and sample is forced to 'first'.
        """
        if split != "val":
            return limit, ids, sample, sample_seed, failed_from_run
        return None, None, "first", None, None

    def _public_payload_view(self, payload: dict[str, Any]) -> dict[str, Any]:
        public_payload = dict(payload)
        if str(public_payload.get("split", "")) == "val":
            public_payload["examples"] = []
        return public_payload

    def _rows_for_split(self, split: str) -> list[DatasetRow]:
        if split not in self.dataset_rows:
            raise ValueError(f"split must be 'train' or 'val'; got: {split!r}")
        rows = self.dataset_rows[split]
        if not rows:
            raise ValueError(f"split has no examples: {split}")
        return rows

    def _charge_budget(self, *, units: int, reason: str) -> int:
        if units <= 0:
            return self.state.remaining_budget

        rejected = False
        with self._budget_lock:
            if self.state.remaining_budget < units:
                rejected = True
                remaining_budget = self.state.remaining_budget
            else:
                self.state.remaining_budget -= units
                remaining_budget = self.state.remaining_budget

        event_payload = {
            "units": units,
            "reason": reason,
            "remaining_budget": remaining_budget,
        }
        if rejected:
            self._emit_event("budget_charge_rejected", event_payload)
            raise BudgetExceededError(
                f"BUDGET_EXCEEDED: requested {units} budget units for {reason} "
                f"with only {remaining_budget} remaining"
            )

        self._emit_event("budget_charged", event_payload)
        return remaining_budget

    def charge_llm_requests(self, *, source: str, requests: int = 1) -> int:
        remaining = self._charge_budget(units=requests, reason=f"{source} LM requests")
        if source == "root":
            self.state.root_lm_calls += requests
        elif source == "sub":
            self.state.sub_lm_calls += requests
        return remaining

    def _evaluate_program_raw(
        self,
        split: str = "train",
        limit: int | None = None,
        ids: str | None = None,
        sample: SampleMode = "first",
        sample_seed: int | None = None,
        failed_from_run: str | None = None,
        _progress_callback: Callable[[dict[str, Any]], None] | None = None,
        charge_budget: bool = True,
        phase: str = "agent",
    ) -> dict[str, Any]:
        if sample not in {"first", "random"}:
            raise ValueError("sample must be one of: first, random")

        rows = self._rows_for_split(split)
        self._validate_failed_from_run_split(
            split=split,
            failed_from_run=failed_from_run,
        )

        selected_ids = parse_ids(ids)
        if failed_from_run is not None:
            selected_ids = self._failed_ids_for_run(failed_from_run)

        selected_rows = select_rows(
            rows,
            ids=selected_ids,
            limit=limit,
            sample=sample,
            sample_seed=sample_seed,
        )
        if not selected_rows:
            raise ValueError("no examples selected for evaluation")

        evaluated_count = len(selected_rows)
        self._emit_event(
            "evaluation_started",
            {
                "split": split,
                "phase": phase,
                "config": {
                    "split": split,
                    "limit": limit,
                    "ids": ids,
                    "sample": sample,
                    "sample_seed": sample_seed,
                    "failed_from_run": failed_from_run,
                },
                "selected_example_ids": (
                    [row.example_id for row in selected_rows]
                    if split == "train"
                    else []
                ),
                "selected_count": evaluated_count,
                "charge_budget": charge_budget,
            },
        )
        if charge_budget:
            self._charge_budget(units=evaluated_count, reason="evaluation examples")
            self.state.evaluated_examples += evaluated_count

        progress_callback, progress_reporter = self._progress_callback_with_reporter(
            external_callback=_progress_callback
        )

        run_id = self._new_run_id()
        try:
            payload = evaluate_rows(
                program=self.program,
                rows=selected_rows,
                metric=self.metric,
                eval_lm=self.eval_lm,
                split=split,
                config={
                    "split": split,
                    "limit": limit,
                    "ids": ids,
                    "sample": sample,
                    "sample_seed": sample_seed,
                    "failed_from_run": failed_from_run,
                },
                num_threads=self.num_threads,
                progress_callback=progress_callback,
            )
        finally:
            if progress_reporter is not None:
                try:
                    progress_reporter.close()
                except Exception:  # pragma: no cover - diagnostics cleanup must not fail eval.
                    pass

        payload["run_id"] = run_id
        payload["remaining_budget"] = self.state.remaining_budget
        summary = (
            f"Score: {payload['score']}% | {payload['passed_count']}/{payload['evaluated_count']} passed "
            f"| Budget remaining: {self.state.remaining_budget}"
        )
        payload["summary_line"] = summary

        path = self._store_run_payload(run_id, self._public_payload_view(payload))
        self.state.runs[run_id] = RunMeta(
            run_id=run_id,
            split=split,
            score=float(payload["score"]),
            evaluated_count=int(payload["evaluated_count"]),
            passed_count=int(payload["passed_count"]),
            remaining_budget=int(self.state.remaining_budget),
            config=dict(payload["config"]),
            storage_path=path,
        )

        self.state.latest_run_id = run_id
        if self._is_eligible_best_run(
            split=split,
            selected_rows=selected_rows,
            selected_ids=selected_ids,
            failed_from_run=failed_from_run,
        ):
            score = float(payload["score"])
            if self.state.best_run_id is None or score > self.state.best_score:
                self.state.best_score = score
                self.state.best_run_id = run_id
                self.state.best_prompt_map = dict(self.state.current_prompt_map)

        if self._debug_display is None:
            print(summary)
        event_payload = self._public_payload_view(payload)
        self._emit_event(
            "evaluation_completed",
            {
                "split": split,
                "phase": phase,
                "run_id": run_id,
                "payload": event_payload,
            },
        )
        return payload

    def _progress_callback_with_reporter(
        self,
        *,
        external_callback: Callable[[dict[str, Any]], None] | None,
    ) -> tuple[Callable[[dict[str, Any]], None] | None, ProgressReporter | None]:
        console = (
            self._debug_display.progress_console()
            if self._debug_display is not None
            else None
        )
        reporter = create_progress_reporter(
            use_rich=self._debug_display is not None,
            console=console,
        )

        def dispatch(event: dict[str, Any]) -> None:
            try:
                reporter.handle_event(event)
            except Exception:  # pragma: no cover - diagnostics callback must not break eval.
                pass
            if external_callback is not None:
                external_callback(event)

        return dispatch, reporter

    def evaluate_program(
        self,
        split: str = "train",
        limit: int | None = None,
        ids: str | None = None,
        sample: SampleMode = "first",
        sample_seed: int | None = None,
        failed_from_run: str | None = None,
        _progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        limit, ids, sample, sample_seed, failed_from_run = (
            self._canonicalize_public_evaluation_request(
                split=split,
                limit=limit,
                ids=ids,
                sample=sample,
                sample_seed=sample_seed,
                failed_from_run=failed_from_run,
            )
        )
        self._validate_public_evaluation_request(
            split=split,
            limit=limit,
            ids=ids,
            sample=sample,
            sample_seed=sample_seed,
            failed_from_run=failed_from_run,
        )
        payload = self._evaluate_program_raw(
            split=split,
            limit=limit,
            ids=ids,
            sample=sample,
            sample_seed=sample_seed,
            failed_from_run=failed_from_run,
            _progress_callback=_progress_callback,
        )
        return self._public_payload_view(payload)

    def run_data_raw(self, run_id: str) -> dict[str, Any]:
        payload = self._load_run_payload(run_id)
        return payload

    def _summary_line_without_budget(self, payload: dict[str, Any]) -> str:
        return (
            f"Score: {payload['score']}% | "
            f"{payload['passed_count']}/{payload['evaluated_count']} passed"
        )

    def run_data(self, run_id: str) -> dict[str, Any]:
        payload = self._public_payload_view(dict(self.run_data_raw(run_id)))
        # Historical per-run budget snapshots are intentionally hidden in run_data()
        # to avoid confusion with current shared budget.
        payload.pop("remaining_budget", None)
        payload["summary_line"] = self._summary_line_without_budget(payload)
        return payload

    def update_prompt(self, step_name: str, new_text: str) -> dict[str, Any]:
        if not step_name:
            self._emit_event(
                "prompt_update_rejected",
                {
                    "step_name": step_name,
                    "reason": "missing_step_name",
                },
            )
            raise PromptUpdateError("step_name must be provided")
        text = str(new_text).strip()
        if not text:
            self._emit_event(
                "prompt_update_rejected",
                {
                    "step_name": step_name,
                    "reason": "empty_prompt_text",
                },
            )
            raise PromptUpdateError("new_text must be non-empty")

        step_modules = {name: module for name, module in self.program.named_predictors()}
        if step_name not in step_modules:
            self._emit_event(
                "prompt_update_rejected",
                {
                    "step_name": step_name,
                    "reason": "unknown_step",
                },
            )
            raise PromptUpdateError(f"unknown step: {step_name}")

        step_module = step_modules[step_name]
        old_signature = step_module.signature
        before_structure = structure_hash(self.program)
        before_prompts = prompt_map(self.program)

        step_module.signature = step_module.signature.with_instructions(text)

        after_structure = structure_hash(self.program)
        after_prompts = prompt_map(self.program)

        changed = {
            key
            for key in before_prompts.keys() | after_prompts.keys()
            if before_prompts.get(key) != after_prompts.get(key)
        }

        if before_structure != after_structure or (
            changed and changed != {step_name}
        ):
            step_module.signature = old_signature
            self._emit_event(
                "prompt_update_rejected",
                {
                    "step_name": step_name,
                    "reason": "non_prompt_structure_change",
                },
            )
            raise PromptUpdateError(
                "prompt update rejected: attempted change outside signature.instructions"
            )

        self.state.current_prompt_map = after_prompts
        result = {
            "status": "ok",
            "step_name": step_name,
            "prompt_hash": _sha256_text(text),
            "prompt_preview": text[:200],
        }
        self._emit_event(
            "prompt_updated",
            {
                "step_name": step_name,
                "new_text": text,
                "result": result,
            },
        )
        return result

    def optimization_status(self) -> dict[str, Any]:
        status = {
            "remaining_budget": self.state.remaining_budget,
            "evaluated_examples": self.state.evaluated_examples,
            "root_lm_calls": self.state.root_lm_calls,
            "sub_lm_calls": self.state.sub_lm_calls,
            "num_threads": self.num_threads,
            "best_score": self.state.best_score,
            "best_run_id": self.state.best_run_id,
            "latest_run_id": self.state.latest_run_id,
            "baseline_run_id": self.state.baseline_run_id,
            "current_prompts": dict(self.state.current_prompt_map),
            "best_prompts": dict(self.state.best_prompt_map),
            "steps": sorted(self.state.current_prompt_map.keys()),
        }
        return status

    def restore_best_prompts(self) -> None:
        apply_prompt_map(self.program, self.state.best_prompt_map)
        self.state.current_prompt_map = prompt_map(self.program)

    def trial_logs(self) -> list[dict[str, Any]]:
        logs: list[dict[str, Any]] = []
        for run_id, meta in sorted(self.state.runs.items(), key=lambda item: item[0]):
            logs.append(
                {
                    "run_id": run_id,
                    "split": meta.split,
                    "score": meta.score,
                    "evaluated_count": meta.evaluated_count,
                    "passed_count": meta.passed_count,
                    "remaining_budget": meta.remaining_budget,
                    "config": dict(meta.config),
                }
            )
        return logs
