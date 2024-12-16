from __future__ import annotations

import os
import random
import re
from typing import Any, cast

import rtoml as toml
from rich.align import Align
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from examexam.constants import BAD_QUESTION_TEXT


# Load questions from a file
def load_questions(file_path: str) -> list[dict[str, Any]]:
    with open(file_path, encoding="utf-8") as file:
        return cast(list[dict[str, Any]], toml.load(file)["questions"])


console = Console()


# Function to clear the terminal
def clear_screen() -> None:
    # maybe use print("\x1B[2J")?
    os.system("cls" if os.name == "nt" else "clear")  # nosec


# Function to play sound effects
def play_sound(_file: str) -> None:
    # playsound(_file)
    pass


def find_select_pattern(input_string: str) -> str:
    """
    Finds the first occurrence of "(Select n)" in the input string where n is a number from 1 to 5.

    :param input_string: The string to search in.
    :return: The matched pattern as a string, or an empty string if no match is found.
    """
    match = re.search(r"\(Select [1-5]\)", input_string)
    return match.group(0) if match else ""


# Function to display a question and get the user's answers
def is_valid(answer: str, option_count:int, answer_count:int, last_is_bad_question_flag:bool=True) -> bool:
    if not answer:
        return False
    answers = answer.split(",")
    for number in answers:
        try:
            int(number)
        except ValueError:
            return False

    if answer_count == 1 and last_is_bad_question_flag and answers[0]==len(answers):
        # Don't count answer if the question is bad!
        return True


    # No out of bounds
    a = all(int(number) <= option_count for number in answers)
    # No negatives
    b = all(int(number) >= 0 for number in answers)
    # No under answers
    c = len(answers) == answer_count
    return (a and b and c)


def ask_question(question: dict[str, Any]) -> list[str]:
    clear_screen()
    # console.print(Panel(question["question"], title=f"Question {question['id']}", style="cyan"))
    question_text = question["question"]

    pattern = find_select_pattern(question_text)
    answer_count = len(question["answers"])

    if pattern:
        correct_select = f"(Select {answer_count})"
        # wrong one found
        if correct_select not in question_text:
            question_text = question_text.replace(pattern, correct_select)

    # Nothing found
    if "(Select" not in question_text:
        question_text = f"{question_text} (Select {answer_count})"

    if "(Select n)" in question_text:
        question_text = question_text.replace("(Select n)", f"(Select {answer_count})")

    question_panel = Align.center(Panel(Markdown(question_text)), vertical="middle")
    console.print(question_panel)

    table = Table(title="Options", style="green")
    table.add_column("Option Number", justify="center")
    table.add_column("Option Text", justify="left")

    for idx, option in enumerate(question["options"], 1):
        table.add_row(str(idx), option)

    table.add_row(str(len(question["options"]) + 1), BAD_QUESTION_TEXT)
    console.print(Align.center(table))

    answer = ""
    option_count = len(question["options"]) + 1
    answer_count = len(question["answers"])
    while not is_valid(answer, option_count,answer_count):
        answer = console.input(
            "[bold yellow]Enter your answer(s) as a comma-separated list (e.g., 1,2): [/bold yellow]"
        )
        msg  = is_valid(answer, option_count, answer_count)
        print(msg)

    # Map the input to the actual options
    selected = [
        question["options"][int(idx) - 1]
        for idx in answer.split(",")
        if idx.isdigit() and 1 <= int(idx) <= len(question["options"])
    ]
    return selected


# Function to calculate and display the results
def display_results(score: float, total: float, withhold_judgement: bool = False) -> None:
    # clear_screen()
    percent = (score / total) * 100
    passed = "Passed" if percent >= 70 else "Failed"

    if withhold_judgement:
        judgement = ""
    else:
        judgement = f"\n[green]{passed}[/green]"
    console.print(
        Panel(
            f"[bold yellow]Your Score: {score}/{total}({percent:.2f}%) {judgement}[/bold yellow]",
            title="Results",
            style="magenta",
        )
    )
    # play_sound("pass.mp3" if passed == "Passed" else "fail.mp3")

def save_session_file(session_file:str, state:list[dict[str, Any]]) -> None:
    with open(session_file, "w", encoding="utf-8") as file:
        data = {"questions": state}
        toml.dump(data, file)


def take_exam_now(question_file:str) -> None:
    """Main function to run the quiz"""
    session = None
    questions = load_questions(question_file)
    save_session_file("session.toml", questions)
    try:
        session = load_questions(question_file)
        interactive_question_and_answer(questions, session)
        save_session_file("session.toml", session)
    except KeyboardInterrupt:
        if session:
            save_session_file("session.toml", session)
        console.print("[bold red]Exiting the exam...[/bold red]")


def interactive_question_and_answer(questions, session):


    score = 0
    so_far = 0
    random.shuffle(questions)
    for question in questions:
        session_question = find_question(question, session)

        if session_question.get("user_score") == 1:
            # Already got it right!
            continue
        selected = ask_question(question)

        correct = set(question["answers"])
        nullable_alternatives = question.get("alternative_answers")
        if nullable_alternatives is None:
            alternatives = set()
        else:
            alternatives = set(nullable_alternatives)
        user_answers = set(selected)

        # clear_screen()
        console.print(
            Panel(
                f"[bold cyan]Correct Answer(s): {', '.join(correct)}\nYour Answer(s): {', '.join(user_answers)}[/bold cyan]",
                title="Answer Review",
                style="blue",
            )
        )

        if alternatives:
            console.print(Rule())
            console.print(
                Panel(
                    f"[bold cyan]Alternative Answer(s):{','.join(alternatives)}[/bold cyan]",
                    title="Alternative Answers",
                    style="blue",
                )
            )

        colored_explanations = []
        can_match_up = len(question["explanation"]) == len(question["options"])
        for place, explanation in enumerate(question["explanation"]):
            option = question["options"][place]

            if not can_match_up:
                # explanation count doesn't match option count
                colored_explanations.append(f"[bold green]{explanation}[/bold green]")
            elif option in question["answers"]:
                colored_explanations.append(f"[bold green]{explanation}[/bold green]")
            else:
                colored_explanations.append(f"[bold red]{explanation}[/bold red]")

        console.print(Panel("\n".join(colored_explanations), title="Explanation"))  # , style="blue"

        session_question["user_answers"] = list(user_answers)
        if user_answers in (correct, alternatives):
            console.print("[bold green]Correct![/bold green]", style="bold green")
            play_sound("correct.mp3")
            score += 1
            session_question["user_score"] = 1
        else:
            console.print("[bold red]Incorrect.[/bold red]", style="bold red")
            play_sound("incorrect.mp3")
            session_question["user_score"] = 0

        so_far += 1
        display_results(score, so_far, withhold_judgement=True)

        go_on = None
        while go_on not in ("", "bad"):
            go_on = console.input("[bold yellow]Press Enter to continue to the next question...[/bold yellow]")

        if go_on == "bad":
            session_question["defective"] = True
            save_session_file("session.toml", session)

    clear_screen()
    display_results(score, len(questions))
    save_session_file("session.toml", session)


def find_question(question, session):
    for q in session:
        if q["id"] == question["id"]:
            session_question = q
            break
    return session_question


if __name__ == "__main__":
    take_exam_now(question_file="cat_ownership_questions.toml")
