from pathlib import Path

from examexam.generate_questions import generate_questions_now


def test_generate_questions_now(tmp_path: Path):
    toc_path = tmp_path / "toc.txt"
    with open(toc_path, "w", encoding="utf-8") as file:
        file.write("topic\ntopic\ntopic\n\n")

    result = generate_questions_now(
        questions_per_toc_topic=10,  # Number of questions to generate
        file_name=str(tmp_path / "aws_developer_questions.toml"),
        toc_file=str(toc_path),
        model="fakebot",
        system_prompt="You are a test maker for AWS Tests.",
    )
    assert result == 0
