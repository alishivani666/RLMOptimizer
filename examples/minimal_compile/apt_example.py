from __future__ import annotations

import dspy

from rlmoptimizer import RLMDocstringOptimizer


class EchoSignature(dspy.Signature):
    """Initial instructions that do not solve the task."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class _FakePredictor:
    def __init__(self) -> None:
        self.signature = EchoSignature
        self.demos: list[dict[str, str]] = []


class DemoProgram(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.step = _FakePredictor()

    def named_predictors(self):
        return [("step", self.step)]

    def predictors(self):
        return [self.step]

    def forward(self, question: str) -> dspy.Prediction:
        instructions = self.step.signature.instructions.lower()
        if "copy question" in instructions:
            return dspy.Prediction(answer=question)
        return dspy.Prediction(answer="wrong")


def exact_metric(example: dspy.Example, pred: dspy.Prediction, _trace=None) -> float:
    return float(example.answer == pred.answer)


def build_trainset(size: int = 4) -> list[dspy.Example]:
    rows: list[dspy.Example] = []
    for index in range(1, size + 1):
        question = f"q{index}"
        rows.append(
            dspy.Example(id=str(index), question=question, answer=question).with_inputs(
                "question"
            )
        )
    return rows


class ScriptedSession:
    """Deterministic stand-in for dspy.RLM so this example runs offline."""

    def __init__(self, **_kwargs) -> None:
        pass

    def run(self, kernel, *, objective: str):
        print("Objective:", objective.splitlines()[0])
        before = kernel.optimization_status()
        print(
            "Before update:",
            f"best_score={before['best_score']}",
            f"remaining_budget={before['remaining_budget']}",
        )

        kernel.update_instruction("step", "Copy question exactly.")
        _ = kernel.evaluate_program(split="train")

        after = kernel.optimization_status()
        print(
            "After update:",
            f"best_score={after['best_score']}",
            f"remaining_budget={after['remaining_budget']}",
        )

        return {
            "final_report": "Updated 'step' instruction to copy question exactly.",
            "suggested_best_run_id": kernel.state.best_run_id,
            "trajectory": [
                {
                    "action": "update_instruction",
                    "predictor": "step",
                    "new_text": "Copy question exactly.",
                }
            ],
            "final_reasoning": "The updated instruction aligns outputs with the exact-match metric.",
        }


def main() -> None:
    trainset = build_trainset(size=4)

    optimizer = RLMDocstringOptimizer(
        max_iterations=3,
        root_lm=dspy.LM("openai/mock-root"),
        eval_lm=dspy.LM("openai/mock-eval"),
        session_cls=ScriptedSession,
    )

    optimized = optimizer.compile(
        student=DemoProgram(),
        trainset=trainset,
        metric=exact_metric,
    )

    print("Baseline run id:", optimized.baseline_run_id)
    print("Best run id:", optimized.best_run_id)
    print("Best score:", optimized.best_score)
    print("Final answer for 'hello':", optimized(question="hello").answer)
    print("Final instruction:", optimized.step.signature.instructions)


if __name__ == "__main__":
    main()
