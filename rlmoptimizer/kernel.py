from __future__ import annotations

import json
import threading
import tempfile
from pathlib import Path
from typing import Any, Callable, Literal
from uuid import uuid4

import dspy

from .evaluator import DatasetRow, build_dataset_rows, evaluate_rows, parse_ids, select_rows
from .fingerprint import apply_instruction_map, instruction_map, structure_hash
from .types import (
    BudgetExceededError,
    InstructionUpdateError,
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
        self.max_budget = int(max_iterations * len(trainset))
        self.dataset_rows = build_dataset_rows(trainset, valset)
        self._budget_lock = threading.Lock()

        self._storage_tempdir: tempfile.TemporaryDirectory[str] | None = None
        if run_storage_dir is None:
            self._storage_tempdir = tempfile.TemporaryDirectory(prefix="rlmoptimizer-runs-")
            self.run_storage_dir = Path(self._storage_tempdir.name)
        else:
            self.run_storage_dir = run_storage_dir
            self.run_storage_dir.mkdir(parents=True, exist_ok=True)

        initial_instruction_map = instruction_map(self.program)
        self.state = OptimizationKernelState(
            remaining_budget=self.max_budget,
            current_instruction_map=dict(initial_instruction_map),
            best_instruction_map=dict(initial_instruction_map),
        )
        self._baseline_structure_hash = structure_hash(self.program)

    def close(self) -> None:
        if self._storage_tempdir is not None:
            self._storage_tempdir.cleanup()
            self._storage_tempdir = None

    def run_baseline(self) -> dict[str, Any]:
        payload = self._evaluate_program_raw(split="train")
        baseline_run_id = str(payload["run_id"])
        self.state.baseline_run_id = baseline_run_id
        return payload

    def _run_path(self, run_id: str) -> Path:
        return self.run_storage_dir / f"{run_id}.json"

    def _new_run_id(self) -> str:
        return uuid4().hex[:8]

    def _load_run_payload(self, run_id: str) -> dict[str, Any]:
        meta = self.state.runs.get(run_id)
        if meta is None:
            raise UnknownRunError(f"unknown run id: {run_id}")
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

        with self._budget_lock:
            if self.state.remaining_budget < units:
                raise BudgetExceededError(
                    f"BUDGET_EXCEEDED: requested {units} budget units for {reason} "
                    f"with only {self.state.remaining_budget} remaining"
                )
            self.state.remaining_budget -= units
            return self.state.remaining_budget

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
    ) -> dict[str, Any]:
        if sample not in {"first", "random"}:
            raise ValueError("sample must be one of: first, random")

        rows = self._rows_for_split(split)

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
        self._charge_budget(units=evaluated_count, reason="evaluation examples")
        self.state.evaluated_examples += evaluated_count

        run_id = self._new_run_id()
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
            progress_callback=_progress_callback,
        )

        payload["run_id"] = run_id
        payload["remaining_budget"] = self.state.remaining_budget
        summary = (
            f"Score: {payload['score']}% | {payload['passed_count']}/{payload['evaluated_count']} passed "
            f"| Budget remaining: {self.state.remaining_budget} | Run: {run_id}"
        )
        payload["summary_line"] = summary

        path = self._store_run_payload(run_id, payload)
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
        if payload["score"] > self.state.best_score:
            self.state.best_score = float(payload["score"])
            self.state.best_run_id = run_id
            self.state.best_instruction_map = dict(self.state.current_instruction_map)

        print(summary)
        return payload

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
        payload = self._evaluate_program_raw(
            split=split,
            limit=limit,
            ids=ids,
            sample=sample,
            sample_seed=sample_seed,
            failed_from_run=failed_from_run,
            _progress_callback=_progress_callback,
        )
        return payload

    def run_data_raw(self, run_id: str) -> dict[str, Any]:
        payload = self._load_run_payload(run_id)
        return payload

    def run_data(self, run_id: str) -> dict[str, Any]:
        payload = self.run_data_raw(run_id)
        return payload

    def update_instruction(self, predictor_name: str, new_text: str) -> dict[str, Any]:
        if not predictor_name:
            raise InstructionUpdateError("predictor_name must be provided")
        text = str(new_text).strip()
        if not text:
            raise InstructionUpdateError("new_text must be non-empty")

        predictors = {name: predictor for name, predictor in self.program.named_predictors()}
        if predictor_name not in predictors:
            raise InstructionUpdateError(f"unknown predictor: {predictor_name}")

        predictor = predictors[predictor_name]
        old_signature = predictor.signature
        before_structure = structure_hash(self.program)
        before_instructions = instruction_map(self.program)

        predictor.signature = predictor.signature.with_instructions(text)

        after_structure = structure_hash(self.program)
        after_instructions = instruction_map(self.program)

        changed = {
            key
            for key in before_instructions.keys() | after_instructions.keys()
            if before_instructions.get(key) != after_instructions.get(key)
        }

        if before_structure != after_structure or (
            changed and changed != {predictor_name}
        ):
            predictor.signature = old_signature
            raise InstructionUpdateError(
                "instruction update rejected: attempted change outside signature.instructions"
            )

        self.state.current_instruction_map = after_instructions
        return {
            "status": "ok",
            "predictor_name": predictor_name,
            "instruction_hash": _sha256_text(text),
            "instruction_preview": text[:200],
        }

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
            "current_instructions": dict(self.state.current_instruction_map),
            "best_instructions": dict(self.state.best_instruction_map),
            "predictors": sorted(self.state.current_instruction_map.keys()),
        }
        return status

    def restore_best_instructions(self) -> None:
        apply_instruction_map(self.program, self.state.best_instruction_map)
        self.state.current_instruction_map = instruction_map(self.program)

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
