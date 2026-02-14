from __future__ import annotations

import copy
import inspect
from pathlib import Path
from typing import Any, Callable, Sequence

import dspy
from dspy.teleprompt.teleprompt import Teleprompter

from .debugger import create_debug_display
from .fingerprint import prompt_map
from .kernel import OptimizationKernel
from .rlm_session import RLMSession
from .types import BudgetExceededError


def _accepts_param(func: Any, name: str) -> bool:
    """Return True if *func* explicitly accepts *name* or has **kwargs."""
    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        return False
    for param in sig.parameters.values():
        if param.name == name:
            return True
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return False


class RLMDocstringOptimizer(Teleprompter):
    def __init__(
        self,
        *,
        max_iterations: int,
        root_lm: dspy.LM,
        sub_lm: dspy.LM | None = None,
        eval_lm: dspy.LM | None = None,
        num_threads: int = 1,
        rlm_max_iterations: int = 200,
        rlm_max_llm_calls: int = 200,
        rlm_max_output_chars: int = 10000,
        root_stateful_session: bool = True,
        verbose: bool = False,
        run_storage_dir: str | Path | None = None,
        rlm_factory: Callable[..., Any] | None = None,
        session_cls: type[RLMSession] = RLMSession,
    ) -> None:
        super().__init__()
        if max_iterations <= 0:
            raise ValueError("max_iterations must be greater than zero")
        if num_threads <= 0:
            raise ValueError("num_threads must be greater than zero")
        if rlm_max_iterations <= 0:
            raise ValueError("rlm_max_iterations must be greater than zero")
        if rlm_max_llm_calls <= 0:
            raise ValueError("rlm_max_llm_calls must be greater than zero")
        if rlm_max_output_chars <= 0:
            raise ValueError("rlm_max_output_chars must be greater than zero")
        if not isinstance(root_stateful_session, bool):
            raise TypeError("root_stateful_session must be a bool")

        self.max_iterations = int(max_iterations)
        self.num_threads = int(num_threads)
        self.rlm_max_iterations = int(rlm_max_iterations)
        self.rlm_max_llm_calls = int(rlm_max_llm_calls)
        self.rlm_max_output_chars = int(rlm_max_output_chars)
        self.root_stateful_session = root_stateful_session
        self.verbose = bool(verbose)
        self.run_storage_dir = (
            Path(run_storage_dir).resolve() if run_storage_dir is not None else None
        )
        self.rlm_factory = rlm_factory
        self.session_cls = session_cls

        self.root_lm = self._require_lm(root_lm, role="root_lm")
        self.sub_lm = self._optional_lm(sub_lm, role="sub_lm")
        self.eval_lm = self._optional_lm(eval_lm, role="eval_lm")

    def _require_lm(self, value: Any, *, role: str) -> dspy.LM:
        if isinstance(value, dspy.LM):
            return value
        raise TypeError(f"{role} must be a dspy.LM instance")

    def _optional_lm(self, value: Any, *, role: str) -> dspy.LM | None:
        if value is None:
            return None
        if isinstance(value, dspy.LM):
            return value
        raise TypeError(f"{role} must be a dspy.LM instance or None")

    def _resolve_eval_lm(self) -> dspy.LM:
        if self.eval_lm is not None:
            return self.eval_lm
        configured = dspy.settings.get("lm")
        if configured is None:
            raise ValueError("eval_lm is required when dspy.settings.lm is not configured")
        if isinstance(configured, dspy.LM):
            return configured
        raise TypeError("dspy.settings.lm must be a dspy.LM instance")

    def _clone_program(self, student: dspy.Module) -> dspy.Module:
        if hasattr(student, "deepcopy") and callable(getattr(student, "deepcopy")):
            return student.deepcopy()
        return copy.deepcopy(student)

    def _objective_text(self, train_size: int, val_size: int) -> str:
        val_part = f"Val set: {val_size} instances. " if val_size else ""
        return f"Train set: {train_size} instances. {val_part}When finished, call SUBMIT(optimized_dspy_program=..., best_run_id=...)."

    def compile(
        self,
        student: dspy.Module,
        *,
        trainset: Sequence[dspy.Example],
        metric: Callable[..., Any],
        teacher: dspy.Module | None = None,
        valset: Sequence[dspy.Example] | None = None,
        **kwargs: Any,
    ) -> dspy.Module:
        del teacher, kwargs
        if not isinstance(student, dspy.Module):
            raise TypeError("student must be a dspy.Module")
        if not trainset:
            raise ValueError("trainset must not be empty")
        if not callable(metric):
            raise TypeError("metric must be callable")

        program = self._clone_program(student)
        eval_lm = self._resolve_eval_lm()
        debug_display = create_debug_display(verbose=self.verbose)

        kernel = OptimizationKernel(
            program=program,
            trainset=list(trainset),
            valset=list(valset) if valset is not None else None,
            metric=metric,
            eval_lm=eval_lm,
            num_threads=self.num_threads,
            max_iterations=self.max_iterations,
            max_output_chars=self.rlm_max_output_chars,
            run_storage_dir=self.run_storage_dir,
            debug_display=debug_display,
        )

        try:
            if debug_display is not None:
                debug_display.show_header(
                    model=str(getattr(self.root_lm, "model", "unknown")),
                    train_size=len(trainset),
                    val_size=len(valset) if valset is not None else 0,
                )

            baseline_payload = kernel.run_baseline()

            if debug_display is not None:
                steps = sorted(prompt_map(program).keys())
                debug_display.show_baseline(
                    score=float(baseline_payload.get("score", 0)),
                    budget=kernel.state.remaining_budget,
                    steps=steps,
                )

            session_kwargs: dict[str, Any] = {
                "root_lm": self.root_lm,
                "sub_lm": self.sub_lm,
                "max_iterations": self.rlm_max_iterations,
                "max_llm_calls": self.rlm_max_llm_calls,
                "max_output_chars": self.rlm_max_output_chars,
                "verbose": self.verbose,
                "rlm_factory": self.rlm_factory,
            }
            if _accepts_param(self.session_cls.__init__, "debug_display"):
                session_kwargs["debug_display"] = debug_display
            if _accepts_param(self.session_cls.__init__, "root_stateful_session"):
                session_kwargs["root_stateful_session"] = self.root_stateful_session
            session = self.session_cls(**session_kwargs)

            session_result: dict[str, Any] = {}
            try:
                run_kwargs: dict[str, Any] = {}
                if _accepts_param(session.run, "objective"):
                    run_kwargs["objective"] = self._objective_text(
                        train_size=len(trainset),
                        val_size=len(valset) if valset is not None else 0,
                    )
                session_result = session.run(kernel, **run_kwargs)
            except BudgetExceededError:
                # Budget exhaustion is a normal hard-stop path.
                session_result = {
                    "optimized_dspy_program": "",
                    "best_run_id": kernel.state.best_run_id or "",
                    "agent_report": "Stopped due to budget exhaustion.",
                    "trajectory": [],
                    "final_reasoning": "",
                }

            kernel.restore_best_prompts()

            if debug_display is not None:
                debug_display.show_final_summary(
                    baseline_score=float(baseline_payload.get("score", 0)),
                    best_score=kernel.state.best_score,
                    budget_used=kernel.max_budget - kernel.state.remaining_budget,
                    total_budget=kernel.max_budget,
                    iterations=len(kernel.state.runs),
                )

            program.trial_logs = kernel.trial_logs()
            program.best_score = kernel.state.best_score
            program.best_run_id = kernel.state.best_run_id
            program.baseline_run_id = kernel.state.baseline_run_id
            program.agent_optimized_dspy_program = session_result.get(
                "optimized_dspy_program", ""
            )
            agent_best_run_id = str(
                session_result.get("best_run_id")
                or kernel.state.best_run_id
                or ""
            )
            program.agent_best_run_id = agent_best_run_id
            program.agent_report = str(
                session_result.get("agent_report")
                or session_result.get("final_reasoning")
                or ""
            )
            program.agent_trajectory = session_result.get("trajectory", [])
            program.agent_final_reasoning = session_result.get("final_reasoning", "")
            return program
        finally:
            if debug_display is not None:
                debug_display.close()
            kernel.close()
