# test_quiz_converter.py
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from examexam.convert_to_pretty import convert_markdown_to_html, generate_markdown, read_toml_file, run, write_to_file

SAMPLE_TOML = textwrap.dedent(
    """
    [[questions]]
    id = "Q1"
    question = "What is 2+2?"
      [[questions.options]]
      text = "3"
      explanation = "Too low"
      is_correct = false
      [[questions.options]]
      text = "4"
      explanation = "Basic arithmetic"
      is_correct = true

    [[questions]]
    id = "Q2"
    question = "Pick all prime numbers below 5"
      [[questions.options]]
      text = "2"
      explanation = "Prime"
      is_correct = true
      [[questions.options]]
      text = "4"
      explanation = "Composite"
      is_correct = false

    [[questions]]
    id = "Q3"
    question = "No correct marked"
      [[questions.options]]
      text = "Option A"
      explanation = "Nope"
      is_correct = false
    """
).strip()


def write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def test_read_toml_file_reads_questions_list(tmp_path: Path) -> None:
    toml_path = write(tmp_path, "q.toml", SAMPLE_TOML)
    questions = read_toml_file(str(toml_path))

    # basic shape checks
    assert isinstance(questions, list)
    assert len(questions) == 3
    assert questions[0]["id"] == "Q1"
    assert questions[0]["options"][1]["is_correct"] is True


def test_read_toml_file_returns_empty_when_no_questions_key(tmp_path: Path) -> None:
    toml_path = write(tmp_path, "empty.toml", 'title = "No questions here"')
    questions = read_toml_file(str(toml_path))
    assert questions == []


def test_generate_markdown_structure_and_content(tmp_path: Path) -> None:
    toml_path = write(tmp_path, "q.toml", SAMPLE_TOML)
    questions = read_toml_file(str(toml_path))
    md = generate_markdown(questions)

    # Headings per question
    assert "### Question Q1: What is 2+2?" in md
    assert "### Question Q2: Pick all prime numbers below 5" in md
    assert "### Question Q3: No correct marked" in md

    # Options listed
    assert "#### Options:" in md
    assert "- 3" in md
    assert "- 4" in md
    assert "- 2" in md
    assert "- Option A" in md

    # Correct Answers sections
    # Q1 should list "4" as correct
    # Q3 should say no correct answer marked
    assert "\n#### Correct Answers:\n" in md
    assert "- 4" in md  # present for Q1
    assert "- *No correct answer marked in source file.*" in md  # present for Q3

    # Explanation lines show Correct/Incorrect status
    assert "- **3**: Too low *(Incorrect)*" in md
    assert "- **4**: Basic arithmetic *(Correct)*" in md
    assert "- **2**: Prime *(Correct)*" in md
    assert "- **4**: Composite *(Incorrect)*" in md
    assert "- **Option A**: Nope *(Incorrect)*" in md

    # separators
    assert "\n---\n\n" in md


def test_convert_markdown_to_html_contains_expected_bits(tmp_path: Path) -> None:
    toml_path = write(tmp_path, "q.toml", SAMPLE_TOML)
    questions = read_toml_file(str(toml_path))
    md = generate_markdown(questions)
    html = convert_markdown_to_html(md)

    # We don't assert on exact HTML (markdown lib may vary),
    # but we expect heading tags and some recognizable text.
    assert "<h3" in html
    assert "Question Q1: What is 2+2?" in html
    assert "Correct Answers:" in html
    assert "No correct answer marked in source file." in html


def test_write_to_file_round_trip(tmp_path: Path) -> None:
    out_path = tmp_path / "out.txt"
    payload = "hello world 🌍"
    write_to_file(payload, str(out_path))
    assert out_path.read_text(encoding="utf-8") == payload


def test_run_end_to_end_creates_files_and_prints(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    toml_path = write(tmp_path, "q.toml", SAMPLE_TOML)
    md_path = tmp_path / "questions.md"
    html_path = tmp_path / "questions.html"

    run(str(toml_path), str(md_path), str(html_path))

    # Files created with non-empty content
    assert md_path.exists() and md_path.stat().st_size > 0
    assert html_path.exists() and html_path.stat().st_size > 0

    # Check a couple of key substrings in the outputs
    md_text = md_path.read_text(encoding="utf-8")
    html_text = html_path.read_text(encoding="utf-8")
    assert "### Question Q1: What is 2+2?" in md_text
    assert "No correct answer marked in source file." in md_text
    assert "<h3" in html_text

    # Confirm the printed success message
    captured = capsys.readouterr()
    assert "Successfully created" in captured.out
    assert "questions.md" in captured.out
    assert "questions.html" in captured.out
