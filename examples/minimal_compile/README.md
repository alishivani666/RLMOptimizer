# Minimal Compile Example

This example demonstrates the optimizer loop end-to-end without calling external LMs.

## Why this example exists

- Runs locally with no API keys.
- Shows baseline evaluation, instruction update, and best-checkpoint restore.
- Produces deterministic output for quick sanity checks.

## Run

From repository root:

```bash
python examples/minimal_compile/apt_example.py
```

## Expected behavior

You should see:
- baseline score starts low,
- one instruction update (`"Copy question exactly."`),
- best score reaches `100.0`,
- final prediction echoes the question text.
