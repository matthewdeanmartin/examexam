# examexam/utils/toml_normalize.py
from __future__ import annotations

from typing import Any


def _is_array_of_tables(val: Any) -> bool:
    return isinstance(val, list) and (len(val) == 0 or isinstance(val[0], dict))


def normalize_question_for_toml(q: dict[str, Any]) -> dict[str, Any]:
    """
    Return a new dict with scalar keys first, then any array-of-tables (e.g., 'options') last.
    Keeps stable/pleasant ordering for common fields.
    """
    # Start with a preferred scalar order if present
    preferred = ["id", "question", "user_answers", "user_score", "defective"]
    out: dict[str, Any] = {}

    for k in preferred:
        if k in q and not _is_array_of_tables(q[k]):
            out[k] = q[k]

    # Add remaining scalars (not already added)
    for k, v in q.items():
        if k not in out and not _is_array_of_tables(v):
            out[k] = v

    # Finally append array-of-tables (AoT) keys (e.g., 'options')
    for k, v in q.items():
        if _is_array_of_tables(v):
            out[k] = v

    return out


def normalize_exam_for_toml(obj: dict[str, Any]) -> dict[str, Any]:
    questions = obj.get("questions", [])
    return {"questions": [normalize_question_for_toml(q) for q in questions]}
