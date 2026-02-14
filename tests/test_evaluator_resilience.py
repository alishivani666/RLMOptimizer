from __future__ import annotations

import dspy

from rlmoptimizer.evaluator import build_dataset_rows, evaluate_rows


class _AlwaysFailProgram(dspy.Module):
    def forward(self, question: str):
        del question
        raise RuntimeError("simulated provider timeout")


def _metric(gold, pred, trace=None):
    del gold, pred, trace
    return 0.0


class _EchoProgram(dspy.Module):
    def forward(self, question: str):
        return dspy.Prediction(answer=question)


def _exact_metric(gold, pred, trace=None):
    del trace
    return 1.0 if gold.answer == pred.answer else 0.0


def test_evaluate_rows_continues_on_program_exceptions():
    trainset = [
        dspy.Example(id="1", question="q1", answer="a1").with_inputs("question"),
        dspy.Example(id="2", question="q2", answer="a2").with_inputs("question"),
    ]
    rows = build_dataset_rows(trainset, None)["train"]
    payload = evaluate_rows(
        program=_AlwaysFailProgram(),
        rows=rows,
        metric=_metric,
        eval_lm=None,
        split="train",
        config={"split": "train"},
    )

    assert payload["evaluated_count"] == 2
    assert payload["passed_count"] == 0
    assert payload["score"] == 0.0
    assert len(payload["examples"]) == 2
    assert all(example["score"] == 0.0 for example in payload["examples"])
    assert all(example["passed"] is False for example in payload["examples"])
    assert all("error_text" not in example for example in payload["examples"])


def test_evaluate_rows_reports_progress_events():
    trainset = [
        dspy.Example(id="1", question="q1", answer="a1").with_inputs("question"),
        dspy.Example(id="2", question="q2", answer="a2").with_inputs("question"),
    ]
    rows = build_dataset_rows(trainset, None)["train"]
    events: list[dict[str, object]] = []

    payload = evaluate_rows(
        program=_AlwaysFailProgram(),
        rows=rows,
        metric=_metric,
        eval_lm=None,
        split="train",
        config={"split": "train"},
        progress_callback=lambda event: events.append(dict(event)),
    )

    assert payload["evaluated_count"] == 2
    assert len(events) == 5
    assert events[0]["stage"] == "evaluation_started"
    assert events[1]["stage"] == "example_started"
    assert events[2]["stage"] == "example_finished"
    assert events[3]["stage"] == "example_started"
    assert events[4]["stage"] == "example_finished"
    assert events[-1]["completed"] == 2
    assert events[-1]["total"] == 2
    assert all("error_text" not in event for event in events)


def test_evaluate_rows_parallel_path_preserves_record_order():
    trainset = [
        dspy.Example(id="1", question="a1", answer="a1").with_inputs("question"),
        dspy.Example(id="2", question="a2", answer="a2").with_inputs("question"),
        dspy.Example(id="3", question="a3", answer="a3").with_inputs("question"),
        dspy.Example(id="4", question="a4", answer="a4").with_inputs("question"),
    ]
    rows = build_dataset_rows(trainset, None)["train"]
    events: list[dict[str, object]] = []

    payload = evaluate_rows(
        program=_EchoProgram(),
        rows=rows,
        metric=_exact_metric,
        eval_lm=None,
        split="train",
        config={"split": "train"},
        num_threads=4,
        progress_callback=lambda event: events.append(dict(event)),
    )

    assert payload["evaluated_count"] == 4
    assert payload["passed_count"] == 4
    assert payload["score"] == 100.0
    assert [item["example_id"] for item in payload["examples"]] == ["1", "2", "3", "4"]
    assert events[0]["stage"] == "evaluation_started"
    assert sum(1 for event in events if event.get("stage") == "example_finished") == 4


def test_evaluate_rows_rejects_non_positive_num_threads():
    trainset = [dspy.Example(id="1", question="q1", answer="a1").with_inputs("question")]
    rows = build_dataset_rows(trainset, None)["train"]

    try:
        evaluate_rows(
            program=_EchoProgram(),
            rows=rows,
            metric=_exact_metric,
            eval_lm=None,
            split="train",
            config={"split": "train"},
            num_threads=0,
        )
        assert False, "expected ValueError for num_threads=0"
    except ValueError as exc:
        assert "num_threads" in str(exc)
