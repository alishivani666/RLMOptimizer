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


def instruction_map(program: dspy.Module) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for name, predictor in program.named_predictors():
        signature = getattr(predictor, "signature", None)
        instructions = getattr(signature, "instructions", "")
        mapping[name] = str(instructions or "")
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


def _demo_hash(predictor: Any) -> str:
    demos = getattr(predictor, "demos", [])
    serialized = json.dumps(_safe_json(demos), ensure_ascii=True, sort_keys=True)
    return _sha256_text(serialized)


def structure_fingerprint(program: dspy.Module) -> dict[str, Any]:
    predictors: list[dict[str, Any]] = []
    for name, predictor in sorted(program.named_predictors(), key=lambda item: item[0]):
        signature = getattr(predictor, "signature", None)
        predictors.append(
            {
                "name": name,
                "predictor_class": (
                    f"{predictor.__class__.__module__}.{predictor.__class__.__qualname__}"
                ),
                "fields": _field_specs(signature),
                "demo_hash": _demo_hash(predictor),
            }
        )

    return {"predictors": predictors}


def structure_hash(program: dspy.Module) -> str:
    serialized = json.dumps(
        structure_fingerprint(program), ensure_ascii=True, sort_keys=True
    )
    return _sha256_text(serialized)


def apply_instruction_map(program: dspy.Module, updates: dict[str, str]) -> None:
    predictors = {name: predictor for name, predictor in program.named_predictors()}
    for name, text in updates.items():
        if name not in predictors:
            continue
        predictor = predictors[name]
        predictor.signature = predictor.signature.with_instructions(text)
