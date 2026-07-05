from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import rtoml as toml

from examexam import take_exam
from examexam.take_exam import take_exam_machine

SAMPLE_TOML = """
[[questions]]
question = "What is the primary purpose of Amazon Athena? (Select n)"
id = "10fc5083-5528-4be1-a3cf-f377ae963dfc"

[[questions.options]]
text = "To perform ad-hoc querying on data stored in S3 using SQL."
explanation = "Amazon Athena allows users to run SQL queries directly on data in S3 without needing to manage any infrastructure. Correct."
is_correct = true

[[questions.options]]
text = "To manage relational databases on EC2."
explanation = "Amazon Athena is a serverless query service, and it does not manage databases on EC2. Incorrect."
is_correct = false
""".strip()


def _minimal_questions() -> dict[str, Any]:
    """Return a minimal valid TOML structure as a Python dict."""
    return {
        "questions": [
            {
                "id": "q-1",
                "question": "Which one is correct? (Select 1)",
                "options": [
                    {
                        "text": "alpha",
                        "explanation": "Because reasons.",
                        "is_correct": True,
                    },
                    {"text": "beta", "explanation": "Nope.", "is_correct": False},
                    {"text": "gamma", "explanation": "Nope.", "is_correct": False},
                ],
            },
            {
                "id": "q-2",
                "question": "Pick two values. (Select 2)",
                "options": [
                    {
                        "text": "red",
                        "explanation": "Correct color.",
                        "is_correct": True,
                    },
                    {
                        "text": "blue",
                        "explanation": "Correct color.",
                        "is_correct": True,
                    },
                    {
                        "text": "green",
                        "explanation": "Not this time.",
                        "is_correct": False,
                    },
                ],
            },
        ]
    }


def _make_mock_ui(**overrides):
    """Create a mock FrontendUI for testing."""
    ui = MagicMock()
    ui.show_message = MagicMock()
    ui.show_error = MagicMock()
    ui.confirm = MagicMock(return_value=True)
    ui.show_session_info = MagicMock()
    ui.show_test_selection = MagicMock(return_value=0)
    ui.show_results = MagicMock()
    ui.show_panel = MagicMock()
    for k, v in overrides.items():
        setattr(ui, k, MagicMock(return_value=v))
    return ui


def write(tmp_path: Path, rel: str, content: str) -> Path:
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


# --------------------------
# Small pure helpers
# --------------------------


@pytest.mark.parametrize(
    "s,expected",
    [
        ("Question (Select 1)", "(Select 1)"),
        ("(Select 5) Choose wisely", "(Select 5)"),
        ("No select here", ""),
        ("Pick two (Select 2) now", "(Select 2)"),
    ],
)
def test_find_select_pattern(s: str, expected: str):
    assert take_exam.find_select_pattern(s) == expected


@pytest.mark.parametrize(
    "answer, option_count, answer_count, ok",
    [
        ("1", 4, 1, True),
        ("2,3", 5, 2, True),
        ("", 3, 1, False),  # empty
        ("a", 3, 1, False),  # non-numeric
        ("4", 3, 1, False),  # out of range
        ("1,2", 3, 1, False),  # wrong number of answers
        ("1", 5, 1, True),
        ("1,2", 5, 2, True),
        ("1,2,3,4", 5, 4, True),
        ("", 5, 0, False),
        ("", 5, 3, False),
        ("10", 5, 3, False),  # too big
        ("11,11,11", 5, 3, False),  # several out of range
        ("a,2", 3, 2, False),
    ],
)
def test_is_valid(answer: str, option_count: int, answer_count: int, ok: bool):
    assert take_exam.is_valid(answer, option_count, answer_count)[0] is ok


def test_is_valid_special_bad_question_slot():
    # Special case: last option reserved for "bad question"
    # If answer_count==1 and user picks exactly the last option, it's valid.
    option_count = 4
    valid, _ = take_exam.is_valid(
        "4", option_count, answer_count=1, last_is_bad_question_flag=True
    )
    assert valid is True


# ---------- load_questions / file helpers ----------


def test_load_questions_reads_toml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    qfile = write(tmp_path, "data/athena.toml", SAMPLE_TOML)
    qs = take_exam.load_questions(str(qfile))
    assert isinstance(qs, list)
    assert qs and qs[0]["id"] == "10fc5083-5528-4be1-a3cf-f377ae963dfc"
    assert len(qs[0]["options"]) == 2


def test_get_session_path_creates_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = take_exam.get_session_path("athena")
    assert p.parent.name == ".session"
    assert p.parent.exists() and p.name == "athena.toml"


def test_get_available_tests_when_present(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write(tmp_path, "data/athena.toml", SAMPLE_TOML)
    write(tmp_path, "data/other.toml", SAMPLE_TOML)
    ui = _make_mock_ui()
    names = take_exam.get_available_tests(ui)
    assert set(names) == {"athena", "other"}


def test_get_available_tests_when_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ui = _make_mock_ui()
    names = take_exam.get_available_tests(ui)
    # Should call show_error and return []
    assert names == []


# ---------- resume session logic ----------


def _session_file_content(questions, start_time: datetime) -> dict:
    return {
        "questions": questions,
        "start_time": start_time.isoformat(),
        "last_updated": start_time.isoformat(),
    }


def test_check_resume_session_user_resumes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    test_name = "athena"
    session_path = take_exam.get_session_path(test_name)
    start_time = datetime.now() - timedelta(minutes=5)

    questions = [
        {
            "id": "q1",
            "user_score": 1,
            "start_time": start_time.isoformat(),
            "completion_time": (start_time + timedelta(seconds=10)).isoformat(),
        },
        {"id": "q2", "user_score": None},
    ]
    session_path.write_text(
        toml.dumps(_session_file_content(questions, start_time)), encoding="utf-8"
    )

    ui = _make_mock_ui(confirm=True)

    resumed, session_data, start_dt = take_exam.check_resume_session(test_name, ui)
    assert resumed is True
    assert isinstance(session_data, list) and len(session_data) == 2
    assert isinstance(start_dt, datetime)


def test_check_resume_session_user_declines_deletes_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    test_name = "athena"
    session_path = take_exam.get_session_path(test_name)
    start_time = datetime.now() - timedelta(minutes=5)
    questions = [{"id": "q1", "user_score": 1}]
    session_path.write_text(
        toml.dumps(_session_file_content(questions, start_time)), encoding="utf-8"
    )

    ui = _make_mock_ui(confirm=False)

    resumed, session_data, start_dt = take_exam.check_resume_session(test_name, ui)
    assert resumed is False
    assert session_data is None
    assert start_dt is None
    assert not session_path.exists()  # file deleted


# ---------- small pure helpers ----------


@pytest.mark.parametrize(
    "td, expected_contains",
    [
        (timedelta(seconds=5), "5 second"),
        (timedelta(minutes=2, seconds=1), "2 minute"),
        (timedelta(hours=1, minutes=1), "1 hour"),
    ],
)
def test_humanize_timedelta(td, expected_contains):
    s = take_exam.humanize_timedelta(td)
    assert expected_contains in s


def test_calculate_confidence_interval_basic():
    lo, hi = take_exam.calculate_confidence_interval(7, 10)
    assert 0.0 <= lo <= hi <= 1.0
    assert pytest.approx(0.7, rel=0.25) == (lo + hi) / 2  # rough center near p
    assert take_exam.calculate_confidence_interval(0, 0) == (0.0, 0.0)


# ---------- timing estimates (with outlier filtering) ----------


def test_calculate_time_estimates_filters_outliers():
    start = datetime.now() - timedelta(minutes=10)
    # Two normal 10s questions + one big outlier 100s (should be filtered: >3x median=10 => 30)
    session = [
        {
            "id": "q1",
            "start_time": (start + timedelta(seconds=0)).isoformat(),
            "completion_time": (start + timedelta(seconds=10)).isoformat(),
            "user_score": 1,
        },
        {
            "id": "q2",
            "start_time": (start + timedelta(seconds=20)).isoformat(),
            "completion_time": (start + timedelta(seconds=30)).isoformat(),
            "user_score": 0,
        },
        {
            "id": "q3",
            "start_time": (start + timedelta(seconds=40)).isoformat(),
            "completion_time": (start + timedelta(seconds=140)).isoformat(),
            "user_score": 1,
        },
        {"id": "q4", "user_score": None},  # remaining
    ]
    avg, eta = take_exam.calculate_time_estimates(session, start)
    # Outlier removed => average ~ (10 + 10)/2 = 10 seconds
    assert timedelta(seconds=5) <= avg <= timedelta(seconds=20)
    # One remaining question -> eta close to avg
    assert isinstance(eta, timedelta)
    assert timedelta(seconds=5) <= eta <= timedelta(seconds=20)


# ---------- save / display / find ----------


def test_save_session_file_writes_toml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    session_file = take_exam.get_session_path("athena")
    state = [{"id": "q1", "user_score": 1}]
    start_time = datetime.now() - timedelta(minutes=1)
    take_exam.save_session_file(session_file, state, start_time)
    assert session_file.exists()
    data = toml.load(session_file)
    assert "questions" in data and isinstance(data["questions"], list)


def test_show_results_does_not_crash():
    """Smoke test: ensure building and showing results doesn't raise."""
    start_time = datetime.now() - timedelta(seconds=30)
    ui = _make_mock_ui()
    result = take_exam._build_exam_result(
        score=1, total=2, start_time=start_time, session=None, is_final=False
    )
    ui.show_results(result)
    ui.show_results.assert_called_once()


# ---------- machine mode (end-to-end over a real TOML file) ----------


@pytest.fixture
def exam_file(tmp_path):
    """Create a temporary TOML file with two exam questions."""
    content = """
    [[questions]]
    id = "11111111-1111-4111-8111-111111111111"
    question = "Which AWS service lets you run SQL directly on data in S3? (Select 1)"


    [[questions.options]]
    text = "Amazon Athena"
    explanation = "Athena is serverless and queries S3 with SQL. Correct."
    is_correct = true


    [[questions.options]]
    text = "Amazon RDS"
    explanation = "RDS is for managed relational databases. Incorrect."
    is_correct = false


    [[questions.options]]
    text = "Amazon Redshift"
    explanation = "Data warehouse service; not direct ad‑hoc on raw S3 objects. Incorrect."
    is_correct = false


    [[questions]]
    id = "22222222-2222-4222-8222-222222222222"
    question = "Choose two highly durable AWS storage classes. (Select 2)"


    [[questions.options]]
    text = "S3 Standard"
    explanation = "Designed for 11 nines of durability. Correct."
    is_correct = true


    [[questions.options]]
    text = "S3 Glacier Deep Archive"
    explanation = "Also 11 nines durability for archival data. Correct."
    is_correct = true


    [[questions.options]]
    text = "Instance Store"
    explanation = "Ephemeral, not durable. Incorrect."
    is_correct = false


    [[questions.options]]
    text = "EFS Infrequent Access"
    explanation = "Durable, but not typically cited at the same durability level as S3 classes; incorrect here."
    is_correct = false
    """
    exam_file = tmp_path / "athena_sample.toml"
    exam_file.write_text(content.strip(), encoding="utf-8")
    return exam_file


def test_exam_machine_oracle(exam_file):
    result = take_exam_machine(str(exam_file), strategy="oracle", seed=123, quiet=True)
    assert result["total"] == 2
    # In oracle mode we should be perfect if TOML marks correct answers
    assert result["score"] == result["total"]
    assert result["percent"] == 100.0
