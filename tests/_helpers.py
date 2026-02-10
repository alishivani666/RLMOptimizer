from __future__ import annotations

import dspy


class DemoSignature(dspy.Signature):
    """Initial instructions that do not solve the task."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class FakePredictor:
    def __init__(self) -> None:
        self.signature = DemoSignature
        self.demos: list[dict[str, str]] = []


class RuleProgram(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.step = FakePredictor()

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
