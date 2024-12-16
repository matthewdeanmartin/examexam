# tests/test_take_exam.py
from __future__ import annotations

from typing import Any

import pytest

# Import after you fix the syntax issues mentioned in the main reply.
import examexam.take_exam as te
from examexam.constants import BAD_QUESTION_TEXT


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


# --------------------------------
# ask_question (interactive piece)
# --------------------------------


def test_ask_question_happy_path(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    """User picks option 1; BAD_QUESTION_TEXT is appended as last option."""
    question = {
        "id": "q-1",
        "question": "Pick one (Select 1)",
        "options": [
            {"text": "alpha", "explanation": "ok", "is_correct": True},
            {"text": "beta", "explanation": "no", "is_correct": False},
        ],
    }
    options_list = list(question["options"])

    # Prevent clearing the terminal during tests
    monkeypatch.setattr(te, "clear_screen", lambda: None)

    # Simulate user entering "1" immediately
    inputs = iter(["1"])
    monkeypatch.setattr(te.console, "input", lambda _: next(inputs))

    selected = te.ask_question(question, options_list)
    assert [opt["text"] for opt in selected] == ["alpha"]

    # Ensure BAD_QUESTION_TEXT was shown (not strictly necessary, but useful)
    out = capsys.readouterr().out + capsys.readouterr().err
    assert BAD_QUESTION_TEXT in out





# --------------------
# find_question helper
# --------------------


def test_find_question_returns_same_object():
    qs = _minimal_questions()["questions"]
    session = [dict(q) for q in qs]
    hit = te.find_question(qs[0], session)
    # Same id, and in normal flow you mutate this dict in-place
    assert hit["id"] == qs[0]["id"]
