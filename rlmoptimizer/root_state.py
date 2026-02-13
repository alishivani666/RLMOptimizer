from __future__ import annotations

from typing import Any, Mapping

import dspy


def _extract_response_id(response: Any) -> str | None:
    response_id = getattr(response, "id", None)
    if response_id is None and isinstance(response, dict):
        response_id = response.get("id")
    if response_id is None and hasattr(response, "model_dump"):
        try:
            dumped = response.model_dump()
            if isinstance(dumped, dict):
                response_id = dumped.get("id")
        except Exception:
            response_id = None

    if response_id is None:
        return None
    text = str(response_id).strip()
    return text or None


class StatefulRootLM(dspy.BaseLM):
    """Wrapper that threads OpenAI Responses state across root-lm turns."""

    def __init__(
        self,
        lm: Any,
        *,
        previous_response_id: str | None = None,
    ) -> None:
        self._lm = lm
        self._previous_response_id = previous_response_id

        # Mirror key BaseLM fields so callback/history plumbing keeps working.
        self.model = str(getattr(lm, "model", ""))
        self.model_type = str(getattr(lm, "model_type", "chat"))
        self.cache = bool(getattr(lm, "cache", True))

        kwargs = getattr(lm, "kwargs", {})
        self.kwargs = kwargs if isinstance(kwargs, dict) else {}

        history = getattr(lm, "history", [])
        self.history = history if isinstance(history, list) else []

        callbacks = getattr(lm, "callbacks", [])
        self.callbacks = callbacks if isinstance(callbacks, list) else []

    @property
    def previous_response_id(self) -> str | None:
        return self._previous_response_id

    def reset_session(self) -> None:
        self._previous_response_id = None

    def _with_session_kwargs(self, kwargs: Mapping[str, Any]) -> dict[str, Any]:
        call_kwargs = dict(kwargs)
        if (
            "previous_response_id" not in call_kwargs
            and self._previous_response_id is not None
        ):
            call_kwargs["previous_response_id"] = self._previous_response_id
        return call_kwargs

    def _remember_response(self, response: Any) -> None:
        response_id = _extract_response_id(response)
        if response_id is not None:
            self._previous_response_id = response_id

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        call_kwargs = self._with_session_kwargs(kwargs)
        response = self._lm.forward(prompt=prompt, messages=messages, **call_kwargs)
        self._remember_response(response)
        return response

    async def aforward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        call_kwargs = self._with_session_kwargs(kwargs)
        aforward = getattr(self._lm, "aforward", None)
        if callable(aforward):
            response = await aforward(prompt=prompt, messages=messages, **call_kwargs)
        else:
            response = self._lm.forward(prompt=prompt, messages=messages, **call_kwargs)
        self._remember_response(response)
        return response

    def copy(self, **kwargs: Any) -> StatefulRootLM:
        copied_lm = self._lm.copy(**kwargs)
        return StatefulRootLM(
            copied_lm,
            previous_response_id=self._previous_response_id,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._lm, name)


def stateful_root_compatibility_issue(lm: Any) -> str | None:
    if isinstance(lm, StatefulRootLM):
        return None

    model_type = getattr(lm, "model_type", None)
    if model_type != "responses":
        return (
            "root_lm stateful sessions require Responses mode "
            f"(model_type='responses', got {model_type!r})."
        )

    kwargs = getattr(lm, "kwargs", None)
    if isinstance(kwargs, dict) and kwargs.get("store") is False:
        return (
            "root_lm stateful sessions are disabled when store=False because "
            "Responses previous_response_id chaining is unavailable in this mode."
        )

    if not callable(getattr(lm, "forward", None)):
        return "root_lm must implement forward() to support stateful sessions."

    return None


def maybe_wrap_stateful_root_lm(lm: Any) -> tuple[Any, str | None]:
    if isinstance(lm, StatefulRootLM):
        return lm, None

    issue = stateful_root_compatibility_issue(lm)
    if issue is not None:
        return lm, issue

    return StatefulRootLM(lm), None
