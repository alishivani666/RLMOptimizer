from __future__ import annotations

from typing import Any

from dspy.primitives.python_interpreter import PythonInterpreter

from rlmoptimizer.interpreter import ReRegisteringPythonInterpreter


def test_execute_forces_reregistration_flag(monkeypatch):
    captured: dict[str, Any] = {}

    def _fake_execute(self: PythonInterpreter, code: str, variables=None):
        del code, variables
        captured["tools_registered_before_super"] = self._tools_registered
        return "ok"

    monkeypatch.setattr(PythonInterpreter, "execute", _fake_execute)

    interpreter = ReRegisteringPythonInterpreter(always_reregister=True)
    interpreter._tools_registered = True

    result = interpreter.execute("print('x')")

    assert result == "ok"
    assert captured["tools_registered_before_super"] is False


def test_process_replacement_resets_mount_and_registration_flags(monkeypatch):
    old_process = object()
    new_process = object()

    def _fake_ensure(self: PythonInterpreter) -> None:
        # Simulate the parent ensuring/replacing the subprocess object.
        self.deno_process = new_process

    monkeypatch.setattr(PythonInterpreter, "_ensure_deno_process", _fake_ensure)

    interpreter = ReRegisteringPythonInterpreter(always_reregister=False)
    interpreter.deno_process = old_process
    interpreter._tools_registered = True
    interpreter._mounted_files = True

    interpreter._ensure_deno_process()

    assert interpreter.deno_process is new_process
    assert interpreter._tools_registered is False
    assert interpreter._mounted_files is False
