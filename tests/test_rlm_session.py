from __future__ import annotations

from rlmoptimizer.rlm_session import RLMSession


class _FakeTools:
    def as_dspy_tools(self) -> list[object]:
        return []


def test_build_rlm_disables_dspy_verbose_logs_when_debug_display_active():
    captured: dict[str, object] = {}

    def _factory(**kwargs):
        captured.update(kwargs)
        return object()

    session = RLMSession(
        root_lm=object(),
        sub_lm=None,
        max_iterations=3,
        max_llm_calls=10,
        max_output_chars=5_000,
        verbose=True,
        rlm_factory=_factory,
        debug_display=object(),
    )

    _ = session._build_rlm(_FakeTools(), sub_lm=None)

    assert captured["verbose"] is False


def test_build_rlm_keeps_dspy_verbose_logs_without_debug_display():
    captured: dict[str, object] = {}

    def _factory(**kwargs):
        captured.update(kwargs)
        return object()

    session = RLMSession(
        root_lm=object(),
        sub_lm=None,
        max_iterations=3,
        max_llm_calls=10,
        max_output_chars=5_000,
        verbose=True,
        rlm_factory=_factory,
        debug_display=None,
    )

    _ = session._build_rlm(_FakeTools(), sub_lm=None)

    assert captured["verbose"] is True
