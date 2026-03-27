from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest
import rtoml as toml
from fastapi.testclient import TestClient

from examexam.frontends.web_ui import WebUI
from examexam.take_exam import get_session_path, save_session_file

SAMPLE_WEB_TOML = """
[[questions]]
id = "q-1"
question = "What is 2 + 2?"

[[questions.options]]
text = "4"
explanation = "Basic arithmetic."
is_correct = true

[[questions.options]]
text = "5"
explanation = "That is too high."
is_correct = false
""".strip()


def write(tmp_path: Path, rel: str, content: str) -> Path:
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def web_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.chdir(tmp_path)
    write(tmp_path, "data/sample_exam.toml", SAMPLE_WEB_TOML)
    monkeypatch.setattr("examexam.frontends.web_ui.random.shuffle", lambda seq: None)
    ui = WebUI()
    return TestClient(ui.app)


def test_index_lists_exam_with_accessible_navigation(web_client: TestClient) -> None:
    response = web_client.get("/")
    assert response.status_code == 200
    assert "Skip to main content" in response.text
    assert "Choose an exam" in response.text
    assert "sample_exam" in response.text
    assert "Start exam" in response.text


def test_web_exam_flow_records_result_and_renders_feedback(web_client: TestClient, tmp_path: Path) -> None:
    start = web_client.post("/start", data={"test_name": "sample_exam", "resume": "false"}, follow_redirects=False)
    assert start.status_code == 303
    assert start.headers["location"] == "/question/1"

    question_page = web_client.get("/question/1")
    assert question_page.status_code == 200
    assert "Answer the current question" in question_page.text
    assert 'type="radio"' in question_page.text

    answer = web_client.post("/answer/1", data={"answer": "1"}, follow_redirects=False)
    assert answer.status_code == 303
    assert answer.headers["location"] == "/feedback/1"

    feedback = web_client.get("/feedback/1")
    assert feedback.status_code == 200
    assert "Correct." in feedback.text
    assert "You selected this" in feedback.text

    results_redirect = web_client.post("/next/1", follow_redirects=False)
    assert results_redirect.status_code == 303
    assert results_redirect.headers["location"] == "/results"

    results = web_client.get("/results")
    assert results.status_code == 200
    assert "Final results" in results.text
    assert "Score: 1/1" in results.text

    session_data = toml.load(get_session_path("sample_exam"))
    assert session_data["questions"][0]["user_score"] == 1


def test_index_offers_resume_when_saved_session_exists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    write(tmp_path, "data/sample_exam.toml", SAMPLE_WEB_TOML)

    start_time = datetime.now() - timedelta(minutes=5)
    save_session_file(
        get_session_path("sample_exam"),
        [
            {
                "id": "q-1",
                "question": "What is 2 + 2?",
                "options": [
                    {"text": "4", "explanation": "Basic arithmetic.", "is_correct": True},
                    {"text": "5", "explanation": "That is too high.", "is_correct": False},
                ],
                "user_answers": ["4"],
                "user_score": 1,
                "start_time": start_time.isoformat(),
                "completion_time": (start_time + timedelta(seconds=15)).isoformat(),
            }
        ],
        start_time,
    )

    client = TestClient(WebUI().app)
    response = client.get("/")
    assert response.status_code == 200
    assert "Resume session" in response.text
    assert "Resume available" in response.text
    assert "1/1" in response.text
