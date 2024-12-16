from __future__ import annotations

from typing import Any

import rtoml as toml
from markdown import markdown


def read_toml_file(file_path: str) -> list[dict[str, Any]]:
    """Reads a TOML file and returns the list of questions."""
    with open(file_path, encoding="utf-8") as file:
        data = toml.load(file)
    return data.get("questions", [])


def generate_markdown(questions: list[dict[str, Any]]) -> str:
    """
    Generates a Markdown string from a list of questions using Schema A.

    Schema A structure:
    [[questions]]
    id = "..."
    question = "..."
    [[questions.options]]
    text = "Some answer"
    explanation = "Why it is right/wrong"
    is_correct = true
    """
    markdown_content = ""
    for question in questions:
        markdown_content += f"### Question {question['id']}: {question['question']}\n\n"

        # Display all options
        markdown_content += "#### Options:\n"
        for option in question.get("options", []):
            markdown_content += f"- {option.get('text', 'N/A')}\n"

        # Find and display the correct answers by checking the 'is_correct' flag
        markdown_content += "\n#### Correct Answers:\n"
        correct_answers = [opt.get("text") for opt in question.get("options", []) if opt.get("is_correct")]
        if not correct_answers:
            markdown_content += "- *No correct answer marked in source file.*\n"
        else:
            for answer in correct_answers:
                markdown_content += f"- {answer}\n"

        # Display the explanation for each option
        markdown_content += "\n#### Explanation:\n"
        for option in question.get("options", []):
            status = "Correct" if option.get("is_correct") else "Incorrect"
            explanation = option.get("explanation", "No explanation provided.")
            option_text = option.get("text", "N/A")
            markdown_content += f"- **{option_text}**: {explanation} *({status})*\n"

        markdown_content += "\n---\n\n"
    return markdown_content


def convert_markdown_to_html(markdown_content: str) -> str:
    """Converts a Markdown string to an HTML string."""
    html_content = markdown(markdown_content)
    return html_content


def write_to_file(content: str, file_path: str) -> None:
    """Writes content to a specified file."""
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)


def run(toml_file_path: str, markdown_file_path: str, html_file_path: str) -> None:
    """
    Main function to run the conversion process.
    Reads a TOML file and writes Markdown and HTML outputs.
    """
    questions = read_toml_file(toml_file_path)

    markdown_content = generate_markdown(questions)
    write_to_file(markdown_content, markdown_file_path)

    html_content = convert_markdown_to_html(markdown_content)
    write_to_file(html_content, html_file_path)

    print(f"Successfully created '{markdown_file_path}' and '{html_file_path}'.")
