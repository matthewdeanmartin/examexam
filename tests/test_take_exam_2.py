from datetime import datetime, timedelta
from pathlib import Path

import pytest
import rtoml as toml

# Import the module under test
# Adjust the import if your package layout differs
from examexam import take_exam

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


def write(tmp_path: Path, rel: str, content: str) -> Path:
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


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
    names = take_exam.get_available_tests()
    assert set(names) == {"athena", "other"}


def test_get_available_tests_when_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    names = take_exam.get_available_tests()
    # Should print an error and return []
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
    session_path.write_text(toml.dumps(_session_file_content(questions, start_time)), encoding="utf-8")

    # Minimal mocking: only the user prompt
    monkeypatch.setattr(take_exam.Confirm, "ask", lambda *a, **k: True)

    resumed, session_data, start_dt = take_exam.check_resume_session(test_name)
    assert resumed is True
    assert isinstance(session_data, list) and len(session_data) == 2
    assert isinstance(start_dt, datetime)


def test_check_resume_session_user_declines_deletes_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    test_name = "athena"
    session_path = take_exam.get_session_path(test_name)
    start_time = datetime.now() - timedelta(minutes=5)
    questions = [{"id": "q1", "user_score": 1}]
    session_path.write_text(toml.dumps(_session_file_content(questions, start_time)), encoding="utf-8")

    monkeypatch.setattr(take_exam.Confirm, "ask", lambda *a, **k: False)

    resumed, session_data, start_dt = take_exam.check_resume_session(test_name)
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


def test_find_select_pattern():
    assert take_exam.find_select_pattern("Pick two (Select 2) now") == "(Select 2)"
    assert take_exam.find_select_pattern("No pattern here") == ""


@pytest.mark.parametrize(
    "answer, option_count, answer_count, ok",
    [
        ("1", 3, 1, True),
        ("1,2", 3, 2, True),
        ("", 3, 1, False),
        ("a,2", 3, 2, False),
        ("4", 3, 1, False),  # out of range
    ],
)
def test_is_valid_common_cases(answer, option_count, answer_count, ok):
    valid, _ = take_exam.is_valid(answer, option_count, answer_count)
    assert valid is ok


def test_is_valid_special_bad_question_slot():
    # Special case: last option reserved for "bad question"
    # If answer_count==1 and user picks exactly the last option, it's valid.
    option_count = 4
    valid, _ = take_exam.is_valid("4", option_count, answer_count=1, last_is_bad_question_flag=True)
    assert valid is True


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


def test_display_results_does_not_crash():
    # Smoke test: ensure printing/rendering doesn't raise
    start_time = datetime.now() - timedelta(seconds=30)
    take_exam.display_results(score=1, total=2, start_time=start_time, session=None, withhold_judgement=True)
