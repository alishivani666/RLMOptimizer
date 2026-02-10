# RLMOptimizer

RLMOptimizer is an agentic prompt optimizer for DSPy programs.
It runs an optimization loop that evaluates your program, analyzes failures, and updates predictor instructions to improve score.

Status of this open-source snapshot:
- Includes: core optimizer library + deterministic local example + core tests.
- Excludes (for now): benchmark replication code and internal docs.

## Quickstart

### 1. Install

```bash
python -m pip install -e ".[dev]"
```

Requirements:
- Python 3.10+
- `dspy>=3.1.3`

### 2. Run the local example (no API keys required)

```bash
python example.py
```

This example uses a scripted optimizer session so you can see the full compile flow without network calls.

### 3. Run with real LMs

```python
import dspy
from rlmoptimizer import RLMDocstringOptimizer

optimizer = RLMDocstringOptimizer(
    max_iterations=5,
    root_lm=dspy.LM("openai/gpt-5"),
    sub_lm=dspy.LM("openai/gpt-5-mini"),
    eval_lm=dspy.LM("openai/gpt-5-mini"),
)

optimized_program = optimizer.compile(
    student=program,
    trainset=trainset,
    metric=metric,
    valset=valset,
)
```

For real model runs, configure provider keys in your environment (for example `OPENAI_API_KEY`).

## How It Works

`compile(...)` performs this loop:
1. Clone the student program.
2. Run baseline evaluation on the train set.
3. Start an RLM-driven loop that can:
   - evaluate program slices,
   - inspect stored run data,
   - update predictor instructions,
   - check optimizer status.
4. Track the best-performing instruction map.
5. Restore best instructions and return the optimized program.

## Budget Model

Total budget units are initialized as:

`max_iterations * len(trainset)`

Budget is charged by:
- examples evaluated (`evaluate_program`),
- root LM calls used by the optimizer agent,
- sub-LM calls used by agent helper queries.

When budget reaches zero, optimization ends with `BudgetExceededError`.

## API Notes

`RLMDocstringOptimizer` is Teleprompter-compatible and supports:
- `max_iterations` (required)
- `root_lm` (required `dspy.LM`)
- `sub_lm` / `eval_lm` (optional `dspy.LM`)
- `num_threads`
- `run_storage_dir`
- `rlm_max_iterations`, `rlm_max_llm_calls`, `rlm_max_output_chars`
- test hooks: `rlm_factory`, `session_cls`

Returned program metadata includes:
- `best_score`
- `best_run_id`
- `baseline_run_id`
- `trial_logs`
- `agent_report`
- `agent_trajectory`
- `agent_final_reasoning`

## Development

Run quality checks:

```bash
ruff check .
pytest -q
```

## Repository Layout

```text
rlmoptimizer/
  __init__.py
  optimizer.py
  kernel.py
  rlm_session.py
  tools.py
  evaluator.py
  budgeting.py
  fingerprint.py
  types.py
example.py
tests/
```
