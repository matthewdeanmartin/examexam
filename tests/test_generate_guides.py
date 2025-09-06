# File: tests/test_generate_guides.py
import textwrap

import pytest

import examexam.generate_study_plan as gsp
import examexam.generate_topic_research as gtr


@pytest.fixture(autouse=True)
def patch_router_call(monkeypatch):
    """
    Patch Router.call to always return predictable markdown text.
    This simulates a successful LLM response.
    """

    def fake_call(self, request, model, essential=False):
        return textwrap.dedent(
            f"""
        # Guide for {request[:20]}...

        ## Core Concepts
        Explanation goes here.

        ## Key Terminology
        Definitions...

        ## Code Examples
        Example code...

        ## Common Pitfalls
        Avoid these.

        ## Further Research
        - google query 1
        - google query 2
        """
        )

    monkeypatch.setattr("examexam.apis.conversation_and_router.Router.call", fake_call)


def test_generate_study_guide_returns_content(monkeypatch):
    content = gtr.generate_study_guide("pytest topic", "fakebot")
    assert content is not None
    assert "Core Concepts" in content
    assert "Further Research" in content


def test_save_and_display_guide_writes_file_and_content(tmp_path):
    guide = "# Fake Guide\n\nContent here."
    topic = "Pytest Topic"
    cwd = tmp_path
    out_dir = cwd / "study_guide"

    # run inside tmp_path
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(cwd)

    gtr.save_and_display_guide(guide, topic)

    files = list(out_dir.glob("pytest_topic.md"))
    assert len(files) == 1
    text = files[0].read_text()
    assert "Fake Guide" in text
    monkeypatch.undo()


def test_generate_topic_research_now_creates_file(monkeypatch, tmp_path):
    cwd = tmp_path
    monkeypatch.chdir(cwd)

    # Patch Confirm.ask to avoid blocking input
    monkeypatch.setattr("examexam.generate_topic_research.Confirm.ask", lambda *a, **kw: True)

    gtr.generate_topic_research_now("pytest topic", model="fakebot")

    out_file = cwd / "study_guide" / "pytest_topic.md"
    assert out_file.exists()
    text = out_file.read_text()
    assert "Core Concepts" in text


def test_generate_study_plan_now_creates_file(tmp_path, monkeypatch):
    cwd = tmp_path
    monkeypatch.chdir(cwd)

    toc_file = cwd / "topics.txt"
    toc_file.write_text("topic one\ntopic two\n")

    gsp.generate_study_plan_now(str(toc_file), model="fakebot")

    out_file = cwd / "study_guide" / "topics_study_plan.md"
    assert out_file.exists()
    text = out_file.read_text()
    assert "# Study Plan for topics" in text
    assert "topic one" in text
    assert "topic two" in text


def test_generate_study_plan_now_handles_empty_file(tmp_path, monkeypatch):
    cwd = tmp_path
    monkeypatch.chdir(cwd)

    toc_file = cwd / "empty.txt"
    toc_file.write_text("\n")

    # Should return early and not create file
    gsp.generate_study_plan_now(str(toc_file), model="fakebot")
    out_dir = cwd / "study_guide"
    assert not out_dir.exists()


def test_generate_study_plan_now_handles_missing_file(tmp_path, monkeypatch):
    cwd = tmp_path
    monkeypatch.chdir(cwd)

    # Should return early and not crash
    gsp.generate_study_plan_now(str(cwd / "doesnotexist.txt"), model="fakebot")
    out_dir = cwd / "study_guide"
    assert not out_dir.exists()
