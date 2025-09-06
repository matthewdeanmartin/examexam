"""
Example toml


[[questions]]
question = "What is the primary purpose of Amazon Athena?"
id = "10fc5083-5528-4be1-a3cf-f377ae963dfc"

[[questions.options]]
text = "To perform ad-hoc querying on data stored in S3 using SQL."
explanation = "Amazon Athena allows users to run SQL queries directly on data in S3 without needing to manage any infrastructure. Correct."
is_correct = true

[[questions.options]]
text = "To manage relational databases on EC2."
explanation = "Amazon Athena is a serverless query service, and it does not manage databases on EC2. Incorrect."
is_correct = false
"""

from __future__ import annotations

import math
import os
import random
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, cast

import dotenv
import rtoml as toml
from rich.align import Align
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from scipy import stats

from examexam.constants import BAD_QUESTION_TEXT
from examexam.utils.toml_normalize import normalize_exam_for_toml

# Load environment variables (e.g., OPENAI_API_KEY)
dotenv.load_dotenv()

console = Console()


def load_questions(file_path: str) -> list[dict[str, Any]]:
    """Load questions from a file"""
    with open(file_path, encoding="utf-8") as file:
        data = toml.load(file)["questions"]
        return cast(list[dict[str, Any]], data)


def get_session_path(test_name: str) -> Path:
    """Get the session file path for a given test"""
    session_dir = Path(".session")
    session_dir.mkdir(exist_ok=True)
    return session_dir / f"{test_name}.toml"


def get_available_tests() -> list[str]:
    """Get list of available test files from /data/ folder"""
    data_dir = Path("data")
    if not data_dir.exists():
        console.print("[bold red]Error: /data/ folder not found![/bold red]")
        data_dir = Path(".")

    test_files = list(data_dir.glob("*.toml"))
    return [f.stem for f in test_files]


def select_test() -> str | None:
    """Let user select a test to take"""
    tests = get_available_tests()
    if not tests:
        console.print("[bold red]No test files found in /data/ folder![/bold red]")
        return None

    console.print("[bold blue]Available Tests:[/bold blue]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Number", style="dim", width=6)
    table.add_column("Test Name")

    for idx, test in enumerate(tests, 1):
        table.add_row(str(idx), test)

    console.print(table)

    while True:
        try:
            choice = Prompt.ask("Enter the test number", default="1")
            test_idx = int(choice) - 1
            if 0 <= test_idx < len(tests):
                return tests[test_idx]
            console.print("[bold red]Invalid choice. Please try again.[/bold red]")
        except ValueError:
            console.print("[bold red]Please enter a valid number.[/bold red]")


def check_resume_session(test_name: str) -> tuple[bool, list[dict[str, Any]] | None, datetime | None]:
    """Check if a session exists and ask if user wants to resume"""
    session_path = get_session_path(test_name)
    if not session_path.exists():
        return False, None, None

    try:
        with open(session_path, encoding="utf-8") as file:
            data = toml.load(file)
            session_data = data.get("questions", [])
            start_time = data.get("start_time")

        # Check if there's any progress
        completed = sum(1 for q in session_data if q.get("user_score") is not None)
        total = len(session_data)

        if completed == 0:
            return False, None, None

        console.print(f"[bold yellow]Found existing session for '{test_name}'[/bold yellow]")
        console.print(f"Progress: {completed}/{total} questions completed")

        start_dt = None
        if start_time:
            try:
                start_dt = datetime.fromisoformat(start_time)
                elapsed = datetime.now() - start_dt
                console.print(f"Started: {humanize_timedelta(elapsed)} ago")
            except (ValueError, TypeError):
                # Invalid start_time format, will use current time as fallback
                console.print("Started: Unknown time ago")

        resume = Confirm.ask("Do you want to resume this session?")
        if resume:
            return True, session_data, start_dt
        # User wants to start fresh
        session_path.unlink()  # Delete old session
        return False, None, None

    except Exception as e:
        console.print(f"[bold red]Error reading session file: {e}[/bold red]")
        return False, None, None


def humanize_timedelta(td: timedelta) -> str:
    """Convert timedelta to human readable format"""
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes > 0:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if seconds > 0 or not parts:
        parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")

    return " ".join(parts)


def calculate_time_estimates(session: list[dict[str, Any]], start_time: datetime) -> tuple[timedelta, timedelta | None]:
    """Calculate average time per question and estimated completion time, removing outliers"""
    completed_times = []

    for question in session:
        if "completion_time" in question and question.get("user_score") is not None:
            completion_dt = datetime.fromisoformat(question["completion_time"])
            question_start = datetime.fromisoformat(question.get("start_time", start_time.isoformat()))
            question_duration = completion_dt - question_start
            completed_times.append(question_duration.total_seconds())

    if len(completed_times) < 2:
        return timedelta(), None

    # Remove outliers (questions that took more than 3x the median)
    median_time = sorted(completed_times)[len(completed_times) // 2]
    filtered_times = [t for t in completed_times if t <= 3 * median_time]

    if not filtered_times:
        return timedelta(), None

    avg_seconds = sum(filtered_times) / len(filtered_times)
    avg_time_per_question = timedelta(seconds=avg_seconds)

    # Calculate remaining questions
    remaining = sum(1 for q in session if q.get("user_score") is None)
    estimated_time_left = timedelta(seconds=avg_seconds * remaining) if remaining > 0 else None

    return avg_time_per_question, estimated_time_left


def clear_screen() -> None:
    """Function to clear the terminal"""
    os.system("cls" if os.name == "nt" else "clear")  # nosec


def play_sound(_file: str) -> None:
    """Function to play sound effects"""
    # playsound(_file)


def find_select_pattern(input_string: str) -> str:
    """
    Finds the first occurrence of "(Select n)" in the input string where n is a number from 1 to 5.
    """
    match = re.search(r"\(Select [1-5]\)", input_string)
    return match.group(0) if match else ""


def is_valid(
    answer: str, option_count: int, answer_count: int, last_is_bad_question_flag: bool = True
) -> tuple[bool, str]:
    if not answer:
        return False, "Please enter an answer."

    answers = answer.split(",")

    # Check if all answers are valid numbers
    for number in answers:
        try:
            int(number)
        except ValueError:
            return False, f"'{number}' is not a valid number."

    # Special case for bad question flag
    if answer_count == 1 and last_is_bad_question_flag and len(answers) == 1 and int(answers[0]) == option_count:
        return True, ""

    # Check bounds
    for number in answers:
        num = int(number)
        if num < 1 or num > option_count:
            return False, f"Answer {num} is out of range (1-{option_count})."

    # Check answer count
    if len(answers) != answer_count:
        return (
            False,
            f"Please select exactly {answer_count} answer{'s' if answer_count != 1 else ''}, you selected {len(answers)}.",
        )

    return True, ""


def ask_question(question: dict[str, Any], options_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clear_screen()
    question_text = question["question"]

    pattern = find_select_pattern(question_text)
    answer_count = len(list(option for option in question["options"] if option.get("is_correct")))

    if pattern:
        correct_select = f"(Select {answer_count})"
        if correct_select not in question_text:
            question_text = question_text.replace(pattern, correct_select)

    if "(Select" not in question_text:
        question_text = f"{question_text} (Select {answer_count})"

    if "(Select n)" in question_text:
        question_text = question_text.replace("(Select n)", f"(Select {answer_count})")

    question_panel = Align.center(Panel(Markdown(question_text)), vertical="middle")
    console.print(question_panel)

    table = Table(title="Options", style="green")
    table.add_column("Option Number", justify="center")
    table.add_column("Option Text", justify="left")

    for idx, option in enumerate(options_list, 1):
        table.add_row(str(idx), option["text"])

    table.add_row(str(len(options_list) + 1), BAD_QUESTION_TEXT)
    console.print(Align.center(table))

    answer = ""
    option_count = len(options_list) + 1
    while True:
        answer = console.input(
            "[bold yellow]Enter your answer(s) as a comma-separated list (e.g., 1,2): [/bold yellow]"
        )
        is_valid_answer, error_msg = is_valid(answer, option_count, answer_count)
        if is_valid_answer:
            break
        console.print(f"[bold red]{error_msg}[/bold red]")

    selected = [
        options_list[int(idx) - 1] for idx in answer.split(",") if idx.isdigit() and 1 <= int(idx) <= len(options_list)
    ]
    return selected


def calculate_confidence_interval(score: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Calculate confidence interval for population proportion"""
    if total == 0:
        return 0.0, 0.0

    p = score / total  # sample proportion
    z = stats.norm.ppf((1 + confidence) / 2)  # z-score for confidence level

    # Standard error
    se = math.sqrt(p * (1 - p) / total)

    # Margin of error
    me = z * se

    # Confidence interval
    lower = max(0, p - me)
    upper = min(1, p + me)

    return lower, upper


def display_results(
    score: float,
    total: float,
    start_time: datetime,
    session: list[dict[str, Any]] = None,
    withhold_judgement: bool = False,
) -> None:
    percent = (score / total) * 100
    passed = "Passed" if percent >= 70 else "Failed"

    # Calculate timing
    elapsed = datetime.now() - start_time

    # Calculate confidence interval
    lower, upper = calculate_confidence_interval(int(score), int(total))

    # Format timing info
    total_time_str = humanize_timedelta(elapsed)

    # Calculate time estimates with outlier removal
    if session:
        avg_time_per_question, estimated_time_left = calculate_time_estimates(session, start_time)
        avg_time_str = humanize_timedelta(avg_time_per_question)

        time_info = f"Total Time: {total_time_str}\nAvg Time per Question: {avg_time_str}"
        if estimated_time_left and not withhold_judgement:
            time_info += f"\nEstimated Time to Complete: {humanize_timedelta(estimated_time_left)}"
    else:
        # Fallback to simple calculation
        time_per_question = elapsed / total if total > 0 else timedelta()
        avg_time_str = humanize_timedelta(time_per_question)
        time_info = f"Total Time: {total_time_str}\nTime per Question: {avg_time_str}"

    # Format confidence interval
    confidence_str = f"{lower * 100:.1f}%-{upper * 100:.1f}%, 95% confidence interval"

    if withhold_judgement:
        judgement = ""
    else:
        judgement = f"\n[green]{passed}[/green]"

    result_text = (
        f"[bold yellow]Your Score: {score}/{total} ({percent:.2f}%){judgement}\n"
        f"{time_info}\n"
        f"Population Estimate: {confidence_str}[/bold yellow]"
    )

    console.print(
        Panel(
            result_text,
            title="Results",
            style="magenta",
        )
    )


def save_session_file(session_file: Path, state: list[dict[str, Any]], start_time: datetime) -> None:
    with open(session_file, "w", encoding="utf-8") as file:
        data = {"questions": state, "start_time": start_time.isoformat(), "last_updated": datetime.now().isoformat()}
        toml.dump(normalize_exam_for_toml(data), file)


def take_exam_now(question_file: str = None) -> None:
    """Main function to run the quiz"""
    if question_file:
        # Legacy API - use provided file path
        test_path = Path(question_file)
        test_name = test_path.stem
        session_path = get_session_path(test_name)

        # Check for existing session
        resume_session, session_data, session_start_time = check_resume_session(test_name)

        if resume_session and session_data:
            session = session_data
            questions = load_questions(question_file)
            start_time = session_start_time or datetime.now()  # Fallback to current time
        else:
            questions = load_questions(question_file)
            session = questions.copy()
            start_time = datetime.now()
            save_session_file(session_path, session, start_time)
    else:
        # New interactive API
        test_name = select_test()
        if not test_name:
            return

        if (Path("data") / f"{test_name}.toml").exists():
            test_file = Path("data") / f"{test_name}.toml"
        else:
            test_file = f"{test_name}.toml"

        session_path = get_session_path(test_name)

        # Check for existing session
        resume_session, session_data, session_start_time = check_resume_session(test_name)

        if resume_session and session_data:
            session = session_data
            questions = load_questions(str(test_file))
            start_time = session_start_time or datetime.now()  # Fallback to current time
        else:
            questions = load_questions(str(test_file))
            session = questions.copy()
            start_time = datetime.now()
            save_session_file(session_path, session, start_time)
    try:
        interactive_question_and_answer(questions, session, session_path, start_time)
        save_session_file(session_path, session, start_time)
    except KeyboardInterrupt:
        save_session_file(session_path, session, start_time)
        console.print("[bold red]Exiting the exam...[/bold red]")


def interactive_question_and_answer(questions, session, session_path: Path, start_time: datetime):
    score = 0
    so_far = 0

    # Count already completed questions
    for question in session:
        if question.get("user_score") == 1:
            score += 1
            so_far += 1

    random.shuffle(questions)
    for question in questions:
        session_question = find_question(question, session)

        if session_question.get("user_score") == 1:
            continue

        # Record when this question started
        question_start_time = datetime.now()
        session_question["start_time"] = question_start_time.isoformat()

        options_list = list(question["options"])
        random.shuffle(options_list)
        try:
            selected = ask_question(question, options_list)
        except KeyboardInterrupt:
            display_results(score, len(questions), start_time, session)
            raise

        # Record completion time
        session_question["completion_time"] = datetime.now().isoformat()

        correct = {option["text"] for option in options_list if option.get("is_correct", False)}
        user_answers = {option["text"] for option in selected}

        # Only show comparison if answers differ
        if user_answers == correct:
            console.print(
                Panel(
                    "[bold green]✓ Correct![/bold green]",
                    title="Answer Review",
                    style="green",
                )
            )
        else:
            console.print(
                Panel(
                    f"[bold cyan]Correct Answer(s): {', '.join(correct)}\nYour Answer(s): {', '.join(user_answers)}[/bold cyan]",
                    title="Answer Review",
                    style="blue",
                )
            )

        # Create numbered explanations matching the original option order
        colored_explanations = []
        for idx, option in enumerate(options_list, 1):
            if option.get("is_correct", False):
                colored_explanations.append(f"{idx}. [bold green]{option['explanation']}[/bold green]")
            else:
                colored_explanations.append(f"{idx}. [bold red]{option['explanation']}[/bold red]")

        console.print(Panel("\n".join(colored_explanations), title="Explanation"))

        session_question["user_answers"] = list(user_answers)
        if user_answers == correct:
            play_sound("correct.mp3")
            score += 1
            session_question["user_score"] = 1
        else:
            console.print("[bold red]Incorrect.[/bold red]", style="bold red")
            play_sound("incorrect.mp3")
            session_question["user_score"] = 0

        so_far += 1
        display_results(score, so_far, start_time, session, withhold_judgement=True)

        go_on = None
        while go_on not in ("", "bad"):
            go_on = console.input("[bold yellow]Press Enter to continue to the next question...[/bold yellow]")

        if go_on == "bad":
            session_question["defective"] = True
            save_session_file(session_path, session, start_time)

    clear_screen()
    display_results(score, len(questions), start_time, session)
    save_session_file(session_path, session, start_time)
    return score


def find_question(question: dict[str, Any], session: list[dict[str, Any]]) -> dict[str, Any]:
    session_question = {}
    for q in session:
        if q["id"] == question["id"]:
            session_question = q
            break
    return session_question


if __name__ == "__main__":
    take_exam_now()
