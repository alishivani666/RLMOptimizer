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
    ) -> None:
        self._lm = lm
        self._budget_consumer = budget_consumer
        self._source = source

    def __call__(self, *args: Any, **kwargs: Any):
        self._budget_consumer.charge_llm_requests(source=self._source, requests=1)
        return self._lm(*args, **kwargs)

    async def acall(self, *args: Any, **kwargs: Any):
        self._budget_consumer.charge_llm_requests(source=self._source, requests=1)
        return await self._lm.acall(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any):
        self._budget_consumer.charge_llm_requests(source=self._source, requests=1)
        forward = getattr(self._lm, "forward", None)
        if callable(forward):
            return forward(*args, **kwargs)
        return self._lm(*args, **kwargs)

    async def aforward(self, *args: Any, **kwargs: Any):
        self._budget_consumer.charge_llm_requests(source=self._source, requests=1)
        aforward = getattr(self._lm, "aforward", None)
        if callable(aforward):
            return await aforward(*args, **kwargs)
        acall = getattr(self._lm, "acall", None)
        if callable(acall):
            return await acall(*args, **kwargs)
        return self._lm(*args, **kwargs)

    def copy(self, **kwargs: Any) -> BudgetMeteredLM:
        copied = self._lm.copy(**kwargs)
        return BudgetMeteredLM(
            lm=copied,
            budget_consumer=self._budget_consumer,
            source=self._source,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._lm, name)
