from __future__ import annotations

from typing import Any

import dspy


class BudgetMeteredLM(dspy.BaseLM):
    """LM adapter that charges optimizer budget per LM request."""

    def __init__(
        self,
        *,
        lm: Any,
        budget_consumer: Any,
        source: str,
        event_callback: Any | None = None,
    ) -> None:
        self._lm = lm
        self._budget_consumer = budget_consumer
        self._source = source
        self._event_callback = event_callback

    def _emit_event(self, event_type: str, payload: dict[str, Any]) -> None:
        callback = self._event_callback
        if callback is None:
            return
        try:
            callback(
                {
                    "source": "budget_lm",
                    "event": event_type,
                    "lm_source": self._source,
                    **payload,
                }
            )
        except Exception:
            pass

    def __call__(self, *args: Any, **kwargs: Any):
        self._emit_event(
            "lm_request_started",
            {"method": "__call__", "args": args, "kwargs": kwargs},
        )
        self._budget_consumer.charge_llm_requests(source=self._source, requests=1)
        try:
            response = self._lm(*args, **kwargs)
        except Exception as exc:
            self._emit_event(
                "lm_request_failed",
                {
                    "method": "__call__",
                    "args": args,
                    "kwargs": kwargs,
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                },
            )
            raise
        self._emit_event(
            "lm_request_completed",
            {
                "method": "__call__",
                "args": args,
                "kwargs": kwargs,
                "response": response,
            },
        )
        return response

    async def acall(self, *args: Any, **kwargs: Any):
        self._emit_event(
            "lm_request_started",
            {"method": "acall", "args": args, "kwargs": kwargs},
        )
        self._budget_consumer.charge_llm_requests(source=self._source, requests=1)
        try:
            response = await self._lm.acall(*args, **kwargs)
        except Exception as exc:
            self._emit_event(
                "lm_request_failed",
                {
                    "method": "acall",
                    "args": args,
                    "kwargs": kwargs,
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                },
            )
            raise
        self._emit_event(
            "lm_request_completed",
            {
                "method": "acall",
                "args": args,
                "kwargs": kwargs,
                "response": response,
            },
        )
        return response

    def forward(self, *args: Any, **kwargs: Any):
        self._emit_event(
            "lm_request_started",
            {"method": "forward", "args": args, "kwargs": kwargs},
        )
        self._budget_consumer.charge_llm_requests(source=self._source, requests=1)
        forward = getattr(self._lm, "forward", None)
        try:
            if callable(forward):
                response = forward(*args, **kwargs)
            else:
                response = self._lm(*args, **kwargs)
        except Exception as exc:
            self._emit_event(
                "lm_request_failed",
                {
                    "method": "forward",
                    "args": args,
                    "kwargs": kwargs,
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                },
            )
            raise
        self._emit_event(
            "lm_request_completed",
            {
                "method": "forward",
                "args": args,
                "kwargs": kwargs,
                "response": response,
            },
        )
        return response

    async def aforward(self, *args: Any, **kwargs: Any):
        self._emit_event(
            "lm_request_started",
            {"method": "aforward", "args": args, "kwargs": kwargs},
        )
        self._budget_consumer.charge_llm_requests(source=self._source, requests=1)
        aforward = getattr(self._lm, "aforward", None)
        try:
            if callable(aforward):
                response = await aforward(*args, **kwargs)
            else:
                acall = getattr(self._lm, "acall", None)
                if callable(acall):
                    response = await acall(*args, **kwargs)
                else:
                    response = self._lm(*args, **kwargs)
        except Exception as exc:
            self._emit_event(
                "lm_request_failed",
                {
                    "method": "aforward",
                    "args": args,
                    "kwargs": kwargs,
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                },
            )
            raise
        self._emit_event(
            "lm_request_completed",
            {
                "method": "aforward",
                "args": args,
                "kwargs": kwargs,
                "response": response,
            },
        )
        return response

    def copy(self, **kwargs: Any) -> BudgetMeteredLM:
        copied = self._lm.copy(**kwargs)
        return BudgetMeteredLM(
            lm=copied,
            budget_consumer=self._budget_consumer,
            source=self._source,
            event_callback=self._event_callback,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._lm, name)
