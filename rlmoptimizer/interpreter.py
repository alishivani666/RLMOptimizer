from __future__ import annotations

from typing import Any

from dspy.primitives.python_interpreter import PythonInterpreter


class ReRegisteringPythonInterpreter(PythonInterpreter):
    """Local interpreter wrapper that aggressively refreshes tool registration.

    DSPy's sandbox receives tool wrappers and typed output signatures through a
    register RPC. If the underlying Deno process restarts unexpectedly, stale
    host-side flags can incorrectly skip that registration step, leaving the
    REPL without tool functions or with a fallback single-argument SUBMIT.

    This wrapper makes registration resilient by:
    - clearing registration/mount flags whenever the subprocess instance changes
    - optionally forcing tool/output re-registration before every execute call
    """

    def __init__(self, *args: Any, always_reregister: bool = True, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._always_reregister = bool(always_reregister)

    def _ensure_deno_process(self) -> None:
        previous_process = self.deno_process
        super()._ensure_deno_process()

        if self.deno_process is not previous_process:
            # Fresh subprocess: prior tool registration/mount state is invalid.
            self._tools_registered = False
            self._mounted_files = False

    def execute(
        self,
        code: str,
        variables: dict[str, Any] | None = None,
    ) -> Any:
        if self._always_reregister:
            self._tools_registered = False
        return super().execute(code, variables)
