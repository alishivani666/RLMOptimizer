# RLMOptimizer

An autonomous prompt optimizer for DSPy programs. Instead of applying fixed optimization heuristics, RLMOptimizer uses a RLM that can evaluate your program, inspect per-example failures and per-step traces, diagnose what went wrong, and rewrite step prompts — iterating until it finds prompts that work.

It's built on DSPy's [Recursive Language Model (RLM)](https://arxiv.org/abs/2512.24601) module, which gives the optimizer LLM a persistent code environment to analyze evaluation data programmatically.

## Why

Most DSPy optimizers (GEPA, MIPROv2, etc.) search over prompts using fixed strategies. That works well for many tasks, but they can't look at *why* examples fail and reason about what to change. RLMOptimizer can:

- Run an evaluation, then drill into the specific examples that failed
- Trace through each step of a multi-step pipeline to find where errors were introduced
- Use a sub-LLM to batch-classify failure patterns across many examples
- Write new prompts that specifically address the patterns it found
- Re-evaluate to check if the changes helped, and keep iterating

The optimizer agent does all of this autonomously — you just call `compile()`.

## Install

```bash
pip install -e ".[dev]"
```

Requires Python 3.10+ and `dspy>=3.1.3`.

## Quick Example (no API keys needed)

```bash
python example.py
```

This runs a self-contained demo with a scripted optimizer session so you can see the full flow without any API calls.

## Usage

```python
import dspy
from rlmoptimizer import RLMDocstringOptimizer

optimizer = RLMDocstringOptimizer(
    max_iterations=5,
    root_lm=dspy.LM("openai/gpt-5", model_type="responses"),  # stateful optimizer agent
    sub_lm=dspy.LM("openai/gpt-5-mini"),     # used by the agent for analysis
    eval_lm=dspy.LM("openai/gpt-5-mini"),    # runs your program during evaluation
)

optimized = optimizer.compile(
    student=program,
    trainset=trainset,
    metric=metric,
    valset=valset,   # optional
)

# Use the optimized program
result = optimized(question="What is the capital of France?")

# Inspect what happened during optimization
print(optimized.best_score)
print(optimized.trial_logs)
print(optimized.agent_report)
```

Set `OPENAI_API_KEY` (or the appropriate provider key) in your environment.

### What the three LMs do

| Parameter | Role | Recommendation |
|-----------|------|----------------|
| `root_lm` | The optimizer agent that decides what to evaluate, analyzes results, and writes new prompts | Your strongest model |
| `sub_lm` | Helper LLM the agent can call for semantic analysis (e.g., classifying failure patterns across examples) | A capable but cheaper model |
| `eval_lm` | Runs your actual DSPy program during evaluation | Whatever model your program targets |

## How It Works

When you call `compile()`:

1. Your program is cloned and evaluated on the training set to establish a baseline. If a validation set is provided, it is also evaluated once at baseline and the canonical baseline (the one surfaced as `baseline_run_id` and passed to the RLM) is the val baseline.
2. An RLM agent takes over. It has access to four tools:
   - **`evaluate_program`** — run the program on examples. Train evaluations return per-example predictions and per-step traces. Validation evaluations are blind and return aggregate-only metrics.
   - **`run_data`** — re-read any previous evaluation run (no budget cost). Validation runs stay blind/aggregate-only here as well.
   - **`update_prompt`** — rewrite the prompt text for any step.
   - **`optimization_status`** — check remaining budget, current/best scores, and all step prompts.
3. The agent also has `llm_query` and `llm_query_batched` to send data to a sub-LLM for analysis — useful when evaluation payloads are large or when it needs semantic understanding of failures.
4. The agent iterates: evaluating, analyzing, updating prompts, and re-evaluating until it's satisfied or budget runs out.
5. The best-performing prompts are restored and the optimized program is returned.

### Blind validation

Validation is treated as holdout scoring:

- `split='train'` is for diagnostics and supports targeted evaluation (`limit`, `ids`, `sample='random'`, `sample_seed`, `failed_from_run`).
- `split='val'` is blind and aggregate-only: no per-example labels/predictions/traces are returned.
- `split='val'` must be a full-split evaluation (`sample='first'` and `limit`, `ids`, `sample_seed`, `failed_from_run` must be unset).

### Budget

Optimization has a finite budget measured in units:

```
budget = max_iterations * len(trainset)
```

Every example evaluated and every LM call the agent makes costs budget. This includes baseline evaluations (train, and val baseline when `valset` is provided). When budget hits zero, optimization stops and the best result so far is returned. The agent is told about its budget and can use targeted train evaluations (`limit`, `failed_from_run`, specific `ids`) to spend it wisely.

## Parameters

```python
RLMDocstringOptimizer(
    max_iterations=5,           # controls total budget (iterations * trainset size)
    root_lm=...,                # required — the optimizer agent LM
    sub_lm=...,                 # optional — helper LM for agent analysis
    eval_lm=...,                # optional — LM for running your program (falls back to dspy.settings.lm)
    num_threads=8,              # parallel threads for evaluation
    run_storage_dir=None,       # directory to persist evaluation runs (default: temp dir)
    rlm_max_iterations=200,     # max agent loop iterations
    rlm_max_llm_calls=200,      # max sub-LM calls the agent can make
    rlm_max_output_chars=10000, # max output size per agent iteration
    root_stateful_session=True, # auto-thread previous_response_id for root_lm when compatible
    verbose=False,              # print agent trajectory
)
```

Stateful root sessions require `root_lm` in Responses mode (`model_type="responses"`). If the root model is not compatible (for example chat mode, or `store=False`), optimization still runs but root reasoning state is not threaded across turns.

The returned program has extra attributes:

```python
optimized.best_score              # highest score achieved
optimized.best_run_id             # run ID of the best evaluation
optimized.baseline_run_id         # run ID of the canonical baseline (val if provided, else train)
optimized.trial_logs              # list of all evaluation runs with scores and configs
optimized.agent_report            # the agent's summary of what it tried
optimized.agent_trajectory        # full trajectory of agent actions
optimized.agent_final_reasoning   # the agent's final reasoning
```

## Development

```bash
ruff check .
pytest -q
```

## License

MIT
