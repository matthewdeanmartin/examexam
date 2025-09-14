# tests/test_take_exam.py
from __future__ import annotations

from typing import Any

import pytest

# Import after you fix the syntax issues mentioned in the main reply.
import examexam.take_exam as te


def _minimal_questions() -> dict[str, Any]:
    """Return a minimal valid TOML structure as a Python dict."""
    return {
        "questions": [
            {
                "id": "q-1",
                "question": "Which one is correct? (Select 1)",
                "options": [
                    {"text": "alpha", "explanation": "Because reasons.", "is_correct": True},
                    {"text": "beta", "explanation": "Nope.", "is_correct": False},
                    {"text": "gamma", "explanation": "Nope.", "is_correct": False},
                ],
            },
            {
                "id": "q-2",
                "question": "Pick two values. (Select 2)",
                "options": [
                    {"text": "red", "explanation": "Correct color.", "is_correct": True},
                    {"text": "blue", "explanation": "Correct color.", "is_correct": True},
                    {"text": "green", "explanation": "Not this time.", "is_correct": False},
                ],
            },
        ]
    }


# --------------------------
# Small pure helpers
# --------------------------


@pytest.mark.parametrize(
    "s,expected",
    [
        ("Question (Select 1)", "(Select 1)"),
        ("(Select 5) Choose wisely", "(Select 5)"),
        ("No select here", ""),
    ],
)
def test_find_select_pattern(s: str, expected: str):
    assert te.find_select_pattern(s) == expected


@pytest.mark.parametrize(
    "answer, option_count, answer_count, ok",
    [
        ("1", 4, 1, True),
        ("2,3", 5, 2, True),
        ("", 3, 1, False),  # empty
        ("a", 3, 1, False),  # non-numeric
        ("4", 3, 1, False),  # out of range
        #  ("0", 3, 1, False),         # less than 1
        ("1,2", 3, 1, False),  # wrong number of answers
    ],
)
def test_is_valid(answer: str, option_count: int, answer_count: int, ok: bool):
    assert te.is_valid(answer, option_count, answer_count)[0] is ok

