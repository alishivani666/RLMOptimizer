from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class BudgetExceededError(RuntimeError):
    """Raised when an evaluation would exceed remaining budget."""


class PromptUpdateError(RuntimeError):
    """Raised when a prompt update is invalid or unsafe."""


class UnknownRunError(KeyError):
    """Raised when a run id is not found in storage."""


@dataclass
class RunMeta:
    run_id: str
    split: str
    score: float
    evaluated_count: int
    passed_count: int
    remaining_budget: int
    config: dict[str, Any]
    storage_path: Path


@dataclass
class OptimizationKernelState:
    remaining_budget: int
    evaluated_examples: int = 0
    root_lm_calls: int = 0
    sub_lm_calls: int = 0
    latest_run_id: str | None = None
    current_prompt_map: dict[str, str] = field(default_factory=dict)
    runs: dict[str, RunMeta] = field(default_factory=dict)
