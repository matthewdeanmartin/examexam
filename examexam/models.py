"""Core data models for examexam.

These dataclasses replace raw dicts throughout the codebase,
providing type safety and clear contracts between business logic and UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any


@dataclass
class Option:
    """A single answer option for a question."""

    text: str
    explanation: str = ""
    is_correct: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Option:
        return cls(
            text=d.get("text", ""),
            explanation=d.get("explanation", ""),
            is_correct=d.get("is_correct", False),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "explanation": self.explanation,
            "is_correct": self.is_correct,
        }


@dataclass
class Question:
    """A single exam question with its options."""

    question: str
    options: list[Option]
    id: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Question:
        return cls(
            question=d.get("question", ""),
            options=[Option.from_dict(o) for o in d.get("options", [])],
            id=d.get("id", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"question": self.question, "id": self.id}
        result["options"] = [o.to_dict() for o in self.options]
        return result

    @property
    def answer_count(self) -> int:
        return sum(1 for o in self.options if o.is_correct)

    @property
    def correct_answers(self) -> set[str]:
        return {o.text for o in self.options if o.is_correct}


@dataclass
class AnswerFeedback:
    """Feedback for a single answered question."""

    is_correct: bool
    correct_answers: set[str]
    user_answers: set[str]
    explanations: list[tuple[str, bool]]  # (explanation_text, is_correct_option)


@dataclass
class ExamResult:
    """Final or interim exam results."""

    score: int
    total: int
    elapsed: timedelta
    avg_time_per_question: timedelta | None = None
    estimated_time_left: timedelta | None = None
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    exact_ci_lower: float = 0.0
    exact_ci_upper: float = 0.0
    p_value: float = 1.0
    is_final: bool = False

    @property
    def percent(self) -> float:
        return (self.score / self.total * 100) if self.total else 0.0

    @property
    def passed(self) -> bool:
        return self.percent >= 70


@dataclass
class ProgressInfo:
    """Information about an ongoing progress operation."""

    task_id: str
    description: str
    current: int = 0
    total: int = 0


@dataclass
class TestInfo:
    """Information about an available test."""

    name: str
    index: int
