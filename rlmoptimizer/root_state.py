from __future__ import annotations

from typing import Any, Mapping

import dspy
from dspy.clients.lm import (
    _add_dspy_identifier_to_headers,
    _convert_chat_request_to_responses_request,
    litellm,
)


def _convert_content_item_to_responses_format(item: dict[str, Any]) -> dict[str, Any]:
    item_type = str(item.get("type", ""))
    if item_type == "image_url":
        image_url = item.get("image_url", {})
        if isinstance(image_url, dict):
            return {
                "type": "input_image",
                "image_url": str(image_url.get("url", "")),
            }
    if item_type == "text":
        return {
            "type": "input_text",
            "text": str(item.get("text", "")),
        }
    if item_type == "file":
        file_item = item.get("file", {})
        if isinstance(file_item, dict):
            return {
                "type": "input_file",
                "file_data": file_item.get("file_data"),
                "filename": file_item.get("filename"),
                "file_id": file_item.get("file_id"),
            }
    return dict(item)


def _messages_to_responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for raw_message in messages:
        role = str(raw_message.get("role", "user"))
        content = raw_message.get("content")
        blocks: list[dict[str, Any]] = []
        if isinstance(content, str):
            blocks.append({"type": "input_text", "text": content})
        elif isinstance(content, list):
            for content_item in content:
                if isinstance(content_item, dict):
                    blocks.append(_convert_content_item_to_responses_format(content_item))
                else:
                    blocks.append({"type": "input_text", "text": str(content_item)})
        elif content is None:
            blocks.append({"type": "input_text", "text": ""})
        else:
            blocks.append({"type": "input_text", "text": str(content)})
        items.append({"role": role, "content": blocks})
    return items


def _delta_messages_for_chained_response(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not messages:
        return []

    last_assistant_idx = -1
    for idx, message in enumerate(messages):
        if str(message.get("role", "")).strip().lower() == "assistant":
            last_assistant_idx = idx

    if last_assistant_idx >= 0:
        delta = messages[last_assistant_idx + 1 :]
        if delta:
            return delta

    return [messages[-1]]


def _convert_preserving_turns_to_responses_request(
    request: dict[str, Any],
) -> dict[str, Any]:
    converted_request = dict(request)
    messages = converted_request.pop("messages", None)
    converted_request = _convert_chat_request_to_responses_request(converted_request)
    if isinstance(messages, list):
        converted_request["input"] = _messages_to_responses_input(messages)
    return converted_request


def _responses_completion_preserving_turns(
    request: dict[str, Any],
    num_retries: int,
    cache: dict[str, Any] | None = None,
):
    cache = cache or {"no-cache": True, "no-store": True}
    request = dict(request)
    request.pop("rollout_id", None)
    headers = request.pop("headers", None)
    request = _convert_preserving_turns_to_responses_request(request)

    return litellm.responses(
        cache=cache,
        num_retries=num_retries,
        retry_strategy="exponential_backoff_retry",
        headers=_add_dspy_identifier_to_headers(headers),
        **request,
    )


async def _aresponses_completion_preserving_turns(
    request: dict[str, Any],
    num_retries: int,
    cache: dict[str, Any] | None = None,
):
    cache = cache or {"no-cache": True, "no-store": True}
    request = dict(request)
    request.pop("rollout_id", None)
    headers = request.pop("headers", None)
    request = _convert_preserving_turns_to_responses_request(request)

    return await litellm.aresponses(
        cache=cache,
        num_retries=num_retries,
        retry_strategy="exponential_backoff_retry",
        headers=_add_dspy_identifier_to_headers(headers),
        **request,
    )


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

    def _should_preserve_responses_turns(
        self,
        messages: list[dict[str, Any]] | None,
    ) -> bool:
        if self.model_type != "responses":
            return False
        if not isinstance(messages, list):
            return False
        # Preserve existing behavior for non-DSPy LMs used in local tests/mocks.
        return isinstance(self._lm, dspy.LM)

    def _prepare_messages(
        self,
        *,
        prompt: str | None,
        messages: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        prepared = messages or [{"role": "user", "content": prompt}]
        if getattr(self._lm, "use_developer_role", False) and self.model_type == "responses":
            prepared = [
                {**message, "role": "developer"} if message.get("role") == "system" else message
                for message in prepared
            ]
        return prepared

    def _merged_call_kwargs(self, kwargs: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
        call_kwargs = dict(kwargs)
        cache = bool(call_kwargs.pop("cache", self.cache))
        base_kwargs = getattr(self._lm, "kwargs", {})
        merged = {**base_kwargs, **call_kwargs} if isinstance(base_kwargs, dict) else call_kwargs
        warn_rollout = getattr(self._lm, "_warn_zero_temp_rollout", None)
        if callable(warn_rollout):
            warn_rollout(merged.get("temperature"), merged.get("rollout_id"))
        if merged.get("rollout_id") is None:
            merged.pop("rollout_id", None)
        return merged, cache

    def _check_truncation_and_track_usage(self, response: Any) -> None:
        check_truncation = getattr(self._lm, "_check_truncation", None)
        if callable(check_truncation):
            check_truncation(response)
        if (
            not getattr(response, "cache_hit", False)
            and dspy.settings.usage_tracker
            and hasattr(response, "usage")
        ):
            dspy.settings.usage_tracker.add_usage(self.model, dict(response.usage))

    def _forward_responses_preserving_turns(
        self,
        *,
        prompt: str | None,
        messages: list[dict[str, Any]],
        kwargs: Mapping[str, Any],
    ) -> Any:
        merged_kwargs, cache = self._merged_call_kwargs(kwargs)
        prepared_messages = self._prepare_messages(prompt=prompt, messages=messages)
        if merged_kwargs.get("previous_response_id"):
            prepared_messages = _delta_messages_for_chained_response(prepared_messages)
        completion = _responses_completion_preserving_turns
        cache_args = {"no-cache": True, "no-store": True}

        get_cached = getattr(self._lm, "_get_cached_completion_fn", None)
        if callable(get_cached):
            completion, cache_args = get_cached(completion, cache)

        request_payload = dict(model=self.model, messages=prepared_messages, **merged_kwargs)
        num_retries = int(getattr(self._lm, "num_retries", 3))
        response = completion(
            request=request_payload,
            num_retries=num_retries,
            cache=cache_args,
        )
        self._check_truncation_and_track_usage(response)

        return response

    async def _aforward_responses_preserving_turns(
        self,
        *,
        prompt: str | None,
        messages: list[dict[str, Any]],
        kwargs: Mapping[str, Any],
    ) -> Any:
        merged_kwargs, cache = self._merged_call_kwargs(kwargs)
        prepared_messages = self._prepare_messages(prompt=prompt, messages=messages)
        if merged_kwargs.get("previous_response_id"):
            prepared_messages = _delta_messages_for_chained_response(prepared_messages)
        completion = _aresponses_completion_preserving_turns
        cache_args = {"no-cache": True, "no-store": True}

        get_cached = getattr(self._lm, "_get_cached_completion_fn", None)
        if callable(get_cached):
            completion, cache_args = get_cached(completion, cache)

        request_payload = dict(model=self.model, messages=prepared_messages, **merged_kwargs)
        num_retries = int(getattr(self._lm, "num_retries", 3))
        response = await completion(
            request=request_payload,
            num_retries=num_retries,
            cache=cache_args,
        )
        self._check_truncation_and_track_usage(response)

        return response

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        call_kwargs = self._with_session_kwargs(kwargs)
        if self._should_preserve_responses_turns(messages):
            response = self._forward_responses_preserving_turns(
                prompt=prompt,
                messages=list(messages or []),
                kwargs=call_kwargs,
            )
        else:
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
        if self._should_preserve_responses_turns(messages):
            response = await self._aforward_responses_preserving_turns(
                prompt=prompt,
                messages=list(messages or []),
                kwargs=call_kwargs,
            )
        else:
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
