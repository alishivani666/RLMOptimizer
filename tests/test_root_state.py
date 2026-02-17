from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import dspy

from rlmoptimizer.root_state import StatefulRootLM, maybe_wrap_stateful_root_lm


def _response(response_id: str, text: str):
    return SimpleNamespace(
        id=response_id,
        model="openai/fake-responses",
        usage={},
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(text=text)],
            )
        ],
    )


class _FakeResponsesLM:
    def __init__(
        self,
        *,
        model_type: str = "responses",
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.model = "openai/fake-responses"
        self.model_type = model_type
        self.cache = False
        self.kwargs = dict(kwargs or {})
        self.history: list[dict[str, Any]] = []
        self.calls: list[dict[str, Any]] = []
        self._next_id = 1

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ):
        del prompt, messages
        self.calls.append(dict(kwargs))
        response_id = f"resp_{self._next_id}"
        self._next_id += 1
        return _response(response_id=response_id, text=response_id)

    async def aforward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ):
        return self.forward(prompt=prompt, messages=messages, **kwargs)

    def copy(self, **kwargs: Any):
        copied = _FakeResponsesLM(model_type=self.model_type, kwargs=self.kwargs)
        copied.model = self.model
        copied._next_id = self._next_id
        if "model" in kwargs:
            copied.model = str(kwargs["model"])
        if "model_type" in kwargs:
            copied.model_type = str(kwargs["model_type"])
        for key, value in kwargs.items():
            if key in {"model", "model_type"}:
                continue
            if value is None:
                copied.kwargs.pop(key, None)
            else:
                copied.kwargs[key] = value
        return copied


def test_stateful_root_lm_threads_previous_response_id():
    lm = _FakeResponsesLM()
    wrapped = StatefulRootLM(lm)

    first = wrapped("first turn")
    second = wrapped("second turn")

    assert first[0]["text"] == "resp_1"
    assert second[0]["text"] == "resp_2"
    assert lm.calls[0].get("previous_response_id") is None
    assert lm.calls[1].get("previous_response_id") == "resp_1"
    assert wrapped.previous_response_id == "resp_2"


def test_stateful_root_lm_explicit_previous_response_id_wins():
    lm = _FakeResponsesLM()
    wrapped = StatefulRootLM(lm)

    _ = wrapped("first turn")
    _ = wrapped("second turn", previous_response_id="resp_override")

    assert lm.calls[1].get("previous_response_id") == "resp_override"
    assert wrapped.previous_response_id == "resp_2"


def test_stateful_root_lm_copy_keeps_wrapper_and_session_state():
    lm = _FakeResponsesLM()
    wrapped = StatefulRootLM(lm)
    _ = wrapped("first turn")

    copied = wrapped.copy()
    _ = copied("second turn")

    assert isinstance(copied, StatefulRootLM)
    assert copied._lm.calls[0].get("previous_response_id") == "resp_1"
    assert copied.previous_response_id == "resp_2"


def test_stateful_root_lm_reset_session_clears_previous_response_id():
    lm = _FakeResponsesLM()
    wrapped = StatefulRootLM(lm)

    _ = wrapped("first turn")
    wrapped.reset_session()
    _ = wrapped("second turn")

    assert lm.calls[1].get("previous_response_id") is None


def test_maybe_wrap_stateful_root_lm_compatibility_checks():
    compatible_lm = _FakeResponsesLM(model_type="responses")
    wrapped, issue = maybe_wrap_stateful_root_lm(compatible_lm)
    assert issue is None
    assert isinstance(wrapped, StatefulRootLM)

    chat_lm = _FakeResponsesLM(model_type="chat")
    unchanged, issue = maybe_wrap_stateful_root_lm(chat_lm)
    assert unchanged is chat_lm
    assert issue is not None
    assert "model_type='responses'" in issue

    store_false_lm = _FakeResponsesLM(model_type="responses", kwargs={"store": False})
    unchanged, issue = maybe_wrap_stateful_root_lm(store_false_lm)
    assert unchanged is store_false_lm
    assert issue is not None
    assert "store=False" in issue


def test_stateful_root_lm_preserves_multiturn_input_for_dspy_responses_lm(monkeypatch):
    calls: list[dict[str, Any]] = []

    def _fake_responses(**kwargs: Any):
        calls.append(dict(kwargs))
        response_id = f"resp_{len(calls)}"
        return _response(response_id=response_id, text="ok")

    monkeypatch.setattr("rlmoptimizer.root_state.litellm.responses", _fake_responses)

    lm = dspy.LM(
        "openai/mock-root",
        model_type="responses",
        cache=False,
        store=True,
        reasoning_effort="medium",
    )
    wrapped = StatefulRootLM(lm)

    first_messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
    ]
    second_messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "follow up"},
    ]

    first = wrapped(messages=first_messages)
    second = wrapped(messages=second_messages)

    assert first[0]["text"] == "ok"
    assert second[0]["text"] == "ok"

    assert len(calls) == 2
    assert calls[0].get("reasoning_effort") is None
    assert calls[0].get("reasoning") == {"effort": "medium", "summary": "auto"}

    first_input = calls[0].get("input")
    assert isinstance(first_input, list)
    assert [item.get("role") for item in first_input] == ["system", "user", "assistant", "user"]
    assert first_input[0]["content"] == [{"type": "input_text", "text": "sys"}]
    assert first_input[1]["content"] == [{"type": "input_text", "text": "u1"}]
    assert first_input[2]["content"] == [{"type": "input_text", "text": "a1"}]
    assert first_input[3]["content"] == [{"type": "input_text", "text": "u2"}]

    assert calls[0].get("previous_response_id") is None
    assert calls[1].get("previous_response_id") == "resp_1"
    second_input = calls[1].get("input")
    assert isinstance(second_input, list)
    assert [item.get("role") for item in second_input] == ["user"]
    assert second_input[0]["content"] == [{"type": "input_text", "text": "follow up"}]
