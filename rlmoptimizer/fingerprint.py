from __future__ import annotations

import hashlib
import json
from typing import Any

import dspy


def _safe_json(value: Any, _seen: set[int] | None = None) -> Any:
    if _seen is None:
        _seen = set()

    value_id = id(value)
    if value_id in _seen:
        return "<circular_ref>"

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    if isinstance(value, dict):
        _seen.add(value_id)
        try:
            return {str(k): _safe_json(v, _seen) for k, v in value.items()}
        finally:
            _seen.discard(value_id)

    if isinstance(value, (list, tuple, set)):
        _seen.add(value_id)
        try:
            return [_safe_json(item, _seen) for item in value]
        finally:
            _seen.discard(value_id)

    if hasattr(value, "toDict"):
        return _safe_json(value.toDict(), _seen)

    return str(value)


def _sha256_text(text: str) -> str:
    digest = hashlib.sha256()
    digest.update(text.encode("utf-8"))
    return digest.hexdigest()


def prompt_map(program: dspy.Module) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for name, step_module in program.named_predictors():
        signature = getattr(step_module, "signature", None)
        prompt_text = getattr(signature, "instructions", "")
        mapping[name] = str(prompt_text or "")
    return mapping


def _field_specs(signature: type[dspy.Signature]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for field_name, field in signature.fields.items():
        extra = field.json_schema_extra or {}
        specs.append(
            {
                "name": field_name,
                "annotation": str(getattr(field, "annotation", None)),
                "prefix": str(extra.get("prefix", "")),
                "desc": str(extra.get("desc", "")),
                "field_type": str(extra.get("__dspy_field_type", "")),
            }
        )
    return specs


def _demo_hash(step_module: Any) -> str:
    demos = getattr(step_module, "demos", [])
    serialized = json.dumps(_safe_json(demos), ensure_ascii=True, sort_keys=True)
    return _sha256_text(serialized)


def structure_fingerprint(program: dspy.Module) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    for name, step_module in sorted(program.named_predictors(), key=lambda item: item[0]):
        signature = getattr(step_module, "signature", None)
        steps.append(
            {
                "name": name,
                "step_class": (
                    f"{step_module.__class__.__module__}.{step_module.__class__.__qualname__}"
                ),
                "fields": _field_specs(signature),
                "demo_hash": _demo_hash(step_module),
            }
        )

    return {"steps": steps}


def structure_hash(program: dspy.Module) -> str:
    serialized = json.dumps(
        structure_fingerprint(program), ensure_ascii=True, sort_keys=True
    )
    return _sha256_text(serialized)


def apply_prompt_map(program: dspy.Module, updates: dict[str, str]) -> None:
    step_modules = {name: module for name, module in program.named_predictors()}
    for name, text in updates.items():
        if name not in step_modules:
            continue
        step_module = step_modules[name]
        step_module.signature = step_module.signature.with_instructions(text)
