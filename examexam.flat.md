# Contents of examexam source tree

## File: constants.py

```python
BAD_QUESTION_TEXT = "This is a bad question and is not answerable as posed."

```

## File: convert_to_pretty.py

```python
from __future__ import annotations

from typing import Any

import rtoml as toml
from markdown import markdown


def read_toml_file(file_path: str) -> list[dict[str, Any]]:
    with open(file_path, encoding="utf-8") as file:
        data = toml.load(file)
    return data.get("questions", [])


def generate_markdown(questions: list[dict[str, Any]]) -> str:
    markdown_content = ""
    for question in questions:
        markdown_content += f"### Question {question['id']}: {question['question']}\n\n"

        markdown_content += "#### Options:\n"
        for option in question.get("options", []):
            markdown_content += f"- {option.get('text', 'N/A')}\n"

        markdown_content += "\n#### Correct Answers:\n"
        correct_answers = [
            opt.get("text")
            for opt in question.get("options", [])
            if opt.get("is_correct")
        ]
        if not correct_answers:
            markdown_content += "- *No correct answer marked in source file.*\n"
        else:
            for answer in correct_answers:
                markdown_content += f"- {answer}\n"

        markdown_content += "\n#### Explanation:\n"
        for option in question.get("options", []):
            status = "Correct" if option.get("is_correct") else "Incorrect"
            explanation = option.get("explanation", "No explanation provided.")
            option_text = option.get("text", "N/A")
            markdown_content += f"- **{option_text}**: {explanation} *({status})*\n"

        markdown_content += "\n---\n\n"
    return markdown_content


def convert_markdown_to_html(markdown_content: str) -> str:
    html_content = markdown(markdown_content)
    return html_content


def write_to_file(content: str, file_path: str) -> None:
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)


def run(toml_file_path: str, markdown_file_path: str, html_file_path: str) -> None:
    questions = read_toml_file(toml_file_path)

    markdown_content = generate_markdown(questions)
    write_to_file(markdown_content, markdown_file_path)

    html_content = convert_markdown_to_html(markdown_content)
    write_to_file(html_content, html_file_path)

    print(f"Successfully created '{markdown_file_path}' and '{html_file_path}'.")

```

## File: generate_questions.py

```python
from __future__ import annotations

import logging
import os
import re
import uuid
from datetime import datetime

import dotenv
import rtoml as toml

from examexam.apis import Conversation
from examexam.apis.model_router import Router

dotenv.load_dotenv()


logger = logging.getLogger(__name__)


def create_new_conversation(exam_name: str, system_prompt: str) -> Conversation:
    system = f"You are an test maker for the '{exam_name}' exam."
    conversation = Conversation(system=system, mode="full", model="gpt4")
    return conversation


def generate_questions(
    prompt: str,
    n: int,
    conversation: Conversation,
    exam_name: str,
    service: str,
    model: str,
) -> dict[str, list[dict[str, str]]] | None:
    toml_content = None
    questions = None

    logger.info("Generating %d questions with prompt: %s", n, prompt)
    toml_schema = f"""[[questions]]
id = 1
question = "{exam_name} style question goes here? (Select n)"
options = ["answer 1",(five options!)]
explanation =["answer 1 is wrong because...", ...]
answers = ["answer 2"]
"""
    toml_schema = """[[questions]]
question = "Question for user here"

[[questions.options]]
text = "Some Correct answer. Must be first."
explanation = "Explanation. Must be before is_correct. Correct."
is_correct = true

[[questions.options]]
text = "Wrong Answer. Must be first."
explanation = "Explanation. Must be before is_correct. Incorrect."
is_correct = false
"""

    time_now = datetime.now()

    prompt = f"""Generate {n} medium difficulty certification exam questions. {prompt}.
    Follow the following TOML format:

    ```toml
    {toml_schema}
    ```
    One or more can be correct! 
    Five options. 
    Each explanation must end in  "Correct" or "Incorrect", e.g. "Instance storage is ephemeral. Correct".
    Do not use numbers or letters to represent the answers.
       [[questions.options]]
       text = "A. Answer"  # never do this.
       [[questions.options]]
       text = "1. Answer"  # never do this.
    Do not use "All of the above" or the like as an answer.
    """

    router = Router(conversation)
    time_then = datetime.now()
    retries = 0
    while not toml_content:
        if retries > 2:
            break
        content = router.call(prompt, model)
        toml_content = extract_toml(content)
        if toml_content is None:
            retries += 1
            continue

        try:
            questions = toml.loads(toml_content)
            retries += 1
        except TypeError as e:
            with open(f"error_{service}.txt", "w", encoding="utf-8") as error_file:
                error_file.write(toml_content)
            logger.error("Error loading TOML content: %s", e)
            continue

    logger.info("Time taken to generate questions: %s", time_then - time_now)
    return questions


def extract_toml(content: str) -> str | None:
    if not content:
        return None
    match = re.search(r"```toml\n(.*?)\n```", content, re.DOTALL)
    if match:
        logger.info("TOML content found in response.")
        return match.group(1)
    return None


def save_toml_to_file(toml_content: str, file_name: str) -> None:
    if os.path.exists(file_name):
        with open(file_name, encoding="utf-8") as file:
            existing_content = toml.load(file)
        existing_content["questions"].extend(toml.loads(toml_content)["questions"])
        with open(file_name, "w", encoding="utf-8") as file:
            toml.dump(existing_content, file)
    else:
        with open(file_name, "w", encoding="utf-8") as file:
            file.write(toml_content)
    print(f"TOML content saved to {file_name}")


def generate_questions_now(
    questions_per_toc_topic: int,
    file_name: str,
    exam_name: str,
    toc_file: str,
    model: str = "fakebot",
    system_prompt: str = "What would you like to do?",
) -> int:
    total_so_far = 0
    with open(toc_file, encoding="utf-8") as file:
        toc = file.readlines()
        toc = [line.strip() for line in toc]
        for service in toc:
            prompt = f"They must all be {service} questions."
            conversation = create_new_conversation(exam_name, system_prompt)

            questions = generate_questions(
                prompt, questions_per_toc_topic, conversation, exam_name, service, model
            )
            if not questions:
                continue

            for question in questions["questions"]:

                question["id"] = str(uuid.uuid4())

            total_so_far += len(questions["questions"])
            logger.info("Total questions so far: %d", total_so_far)
            toml_content = toml.dumps(questions)
            save_toml_to_file(toml_content, file_name)
    return total_so_far


if __name__ == "__main__":
    generate_questions_now(
        questions_per_toc_topic=10,
        file_name="personal_multiple_choice_tests.toml",
        exam_name="AWS Services Exam",
        toc_file="../example_inputs/personally_important.txt",
        model="gpt4",
        system_prompt="We are writing multiple choice tests.",
    )

```

## File: logging_config.py

```python
from __future__ import annotations
import os
from typing import Any


def generate_config(level: str = "DEBUG") -> dict[str, Any]:
    config: dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": True,
        "formatters": {
            "standard": {"format": "[%(levelname)s] %(name)s: %(message)s"},
            "colored": {
                "()": "colorlog.ColoredFormatter",
                "format": "%(log_color)s%(levelname)-8s%(reset)s %(green)s%(message)s",
            },
        },
        "handlers": {
            "default": {
                "level": "DEBUG",
                "formatter": "colored",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "examexam": {
                "handlers": ["default"],
                "level": "DEBUG",
                "propagate": False,
            }
        },
    }
    if os.environ.get("NO_COLOR") or os.environ.get("CI"):
        config["handlers"]["default"]["formatter"] = "standard"
    return config

```

## File: take_exam.py

```python
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
from rich.table import Table

from examexam.constants import BAD_QUESTION_TEXT

console = Console()


def load_questions(file_path: str) -> list[dict[str, Any]]:
    with open(file_path, encoding="utf-8") as file:
        data = toml.load(file)["questions"]
        return cast(list[dict[str, Any]], data)


def clear_screen() -> None:

    os.system("cls" if os.name == "nt" else "clear")



def play_sound(_file: str) -> None:



def find_select_pattern(input_string: str) -> str:
    match = re.search(r"\(Select [1-5]\)", input_string)
    return match.group(0) if match else ""



def is_valid(answer: str, option_count: int, answer_count: int, last_is_bad_question_flag: bool = True) -> bool:
    if not answer:
        return False
    answers = answer.split(",")
    for number in answers:
        try:
            int(number)
        except ValueError:
            return False

    if answer_count == 1 and last_is_bad_question_flag and answers[0] == len(answers):

        return True


    a = all(int(number) <= option_count for number in answers)

    b = all(int(number) >= 0 for number in answers)

    c = len(answers) == answer_count
    return (a and b and c)


def ask_question(question: dict[str, Any], options_list:list[dict[str,Any]]) -> list[dict[str, Any]]:
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


    while not is_valid(answer, option_count, answer_count):
        answer = console.input(
            "[bold yellow]Enter your answer(s) as a comma-separated list (e.g., 1,2): [/bold yellow]"
        )
        msg = is_valid(answer, option_count, answer_count)
        print(msg)


    selected = [
        options_list[int(idx) - 1]
        for idx in answer.split(",")
        if idx.isdigit() and 1 <= int(idx) <= len(options_list)
    ]
    return selected



def display_results(score: float, total: float, withhold_judgement: bool = False) -> None:

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



def save_session_file(session_file: str, state: list[dict[str, Any]]) -> None:
    with open(session_file, "w", encoding="utf-8") as file:
        data = {"questions": state}
        toml.dump(data, file)


def take_exam_now(question_file: str) -> None:
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

            continue
        options_list = list(question["options"])
        random.shuffle(options_list)
        selected = ask_question(question,options_list)

        correct = {option["text"] for option in options_list if option.get("is_correct", False)}





        user_answers = {option["text"] for option in selected}

        console.print(
            Panel(
                f"[bold cyan]Correct Answer(s): {', '.join(correct)}\nYour Answer(s): {', '.join(user_answers)}[/bold cyan]",
                title="Answer Review",
                style="blue",
            )
        )











        colored_explanations = []
        print(options_list)
        for option in options_list:
            if option.get("is_correct", False):
                colored_explanations.append(f"[bold green]{option["explanation"]}[/bold green]")
            else:
                colored_explanations.append(f"[bold red]{option["explanation"]}[/bold red]")

        console.print(Panel("\n".join(colored_explanations), title="Explanation"))


        session_question["user_answers"] = list(user_answers)

        if user_answers == correct:
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
    take_exam_now(question_file="personal_multiple_choice_tests.toml")

```

## File: validate_questions.py

```python
from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path
from typing import Any

import rtoml as toml

from examexam.apis import Conversation
from examexam.apis.model_router import Router


def read_questions(file_path: Path) -> list[dict[str, Any]]:
    with open(file_path, encoding="utf-8") as file:
        data = toml.load(file)
    return data.get("questions", [])


def ask_llm(
    question: str, options: list[str], answers: list[str], model: str, system: str
) -> list[str]:
    if "(Select" not in question:
        question = f"{question} (Select {len(answers)})"

    prompt = (
        f"Answer the following question in the format 'Answers: [option1 | option2 | ...]'.\n"
        f"Question: {question}\n"
        f"Options: {options}\n"
    )

    conversation = Conversation(system=system, mode="full", model=model)

    router = Router(conversation)
    answer = router.call(prompt, model)
    if answer is None:
        return []

    answer = answer.strip()
    if answer.startswith("Answers:"):
        return parse_answer(answer)
    raise ValueError(
        f"Unexpected response format, didn't start with Answers:, got {answer}"
    )


def parse_answer(answer: str) -> list[str]:
    if answer.startswith("Answers:"):
        answer = answer[8:]
        if (
            "','" in answer or "', '" in answer or '","' in answer or '", "' in answer
        ) and "|" not in answer:
            return parse_quote_lists(answer)

        if "[" in answer and "]" in answer:
            after_square_bracket = answer.split("[")[1]
            answer_part = after_square_bracket.split("]")[0]

            answer_part = answer_part.replace('", "', "|").strip('"')
            answers = answer_part.strip().strip("[]").split("|")
            return [ans.strip("'\" ").strip("'\" ") for ans in answers]
    return []


def parse_quote_lists(answer: str) -> list[str]:
    if "[" in answer and "]" in answer:
        after_square_bracket = answer.split("[")[1]
        answer_part = after_square_bracket.split("]")[0]

        if "', '" in answer_part or '","' in answer_part:
            answer_part = StringIO(answer_part)
            reader = csv.reader(answer_part, delimiter=",")
            answers = next(reader)
            return answers

        answer_part = answer_part.replace("â€˜", "").replace("â€™", "")

        answer_part = answer_part.replace('", "', "|").strip('"')
        answers = answer_part.strip("[] ").split("|")
        return [ans.strip("'\" ").strip("'\" ") for ans in answers]
    return []


def ask_if_bad_question(
    question: str, options: list[str], answers: list[str], model: str, exam_name: str
) -> tuple[str, str]:
    prompt = (
        f"Tell me if the following question is Good or Bad, e.g. would it be unfair to ask this on a test.\n"
        f"It is good if it has an answer, if it not every single option is an answer, if it is not opinion based, if it does not have weasel words such as best, optimal, primary which would "
        f"make many of the answers arguably true on some continuum of truth or opinion, or if the question is about *numerical* ephemeral truths, such as system limitations (max GB, etc) and UI defaults.\n\n"
        f"Question: {question}\n"
        f"Options: {options}\n"
        f"Answers: {answers}\n"
        f"\n"
        f"Think about the answer then write `---\nGood` or `---\nBad`\n"
    )

    system = f"You are a test reviewer for the '{exam_name}'."
    conversation = Conversation(system=system, mode="full", model=model)

    router = Router(conversation)
    answer = router.call(prompt, model)
    if answer is None:
        return "bad", "**** Bot returned None, maybe API failed ****"

    answer = answer.strip()
    if "---" in answer:
        return parse_good_bad(answer)
    raise ValueError(f"Unexpected response format, didn't contain ---:, got {answer}")


def parse_good_bad(answer):
    parts = answer.split("---")
    why = parts[0]
    good_bad = parts[1].strip(" \n").lower()
    if "good" in good_bad:
        return "good", why
    return "bad", why


def grade_test(
    questions: list[dict[str, Any]],
    responses: list[list[str]],
    good_bad: list[tuple[str, str]],
    file_path: Path,
    model: str,
) -> float:
    score = 0
    total = len(questions)

    for question, response in zip(questions, responses, strict=True):

        correct_answers = {
            opt["text"] for opt in question.get("options", []) if opt.get("is_correct")
        }
        given_answers = set(response)
        if correct_answers == given_answers:
            score += 1
        else:
            print(f"\nQuestion ID: {question['id']}")
            print(f"Question: {question['question']}")
            print(f"Correct Answers: {correct_answers}")
            print(f"Your Answers: {given_answers}")
            question[f"{model}_answers"] = list(given_answers)

    for question, opinion in zip(questions, good_bad, strict=True):
        good_bad_val, why = opinion
        question["good_bad"] = good_bad_val
        question["good_bad_why"] = why

    with open(file_path, "w", encoding="utf-8") as file:
        toml.dump({"questions": questions}, file)

    print(f"\nFinal Score: {score}/{total}")
    if total == 0:
        return 0
    return score / total


def validate_questions_now(
    file_name: str,
    exam_name: str,
    model: str = "claude",
) -> float:
    file_path = Path(file_name)
    questions = read_questions(file_path)

    responses = []
    opinions = []
    for question_data in questions:
        question = question_data["question"]
        options_list_of_dicts = question_data.get("options", [])

        option_texts = [opt.get("text", "") for opt in options_list_of_dicts]
        correct_answer_texts = [
            opt.get("text", "")
            for opt in options_list_of_dicts
            if opt.get("is_correct")
        ]

        print(f"Submitting question: {question}")
        response = ask_llm(question, option_texts, correct_answer_texts, model)
        print(f"Received answer: {response}")
        responses.append(response)

        good_bad = ask_if_bad_question(
            question, option_texts, correct_answer_texts, model, exam_name=exam_name
        )
        opinions.append(good_bad)

    return grade_test(questions, responses, opinions, file_path, model)


if __name__ == "__main__":
    validate_questions_now(model="claude")
    validate_questions_now(model="gpt4")

```

## File: __about__.py

```python
__all__ = [
    "__title__",
    "__version__",
    "__description__",
    "__author__",
    "__author_email__",
    "__keywords__",
    "__status__",
    "__license__",
    "__readme__",
    "__repository__",
    "__homepage__",
    "__documentation__",
]

__title__ = "examexam"
__version__ = "0.1.1"
__description__ = "Gather psychology tests for yourself to see how you're doing"
__author__ = "Matthew Martin"
__author_email__ = "matthewdeanmartin@gmail.com"
__keywords__ = ["exam", "multiple-choice"]
__status__ = "4 - Beta"
__license__ = "MIT"
__readme__ = "README.md"
__repository__ = "https://github.com/matthewdeanmartin/examexam"
__homepage__ = "https://github.com/matthewdeanmartin/examexam"
__documentation__ = "https://matthewdeanmartin.github.io/examexam/examexam/index.html"

```

## File: __init__.py

```python

```

## File: __main__.py

```python
from __future__ import annotations

import argparse
import logging
import logging.config
import sys
from collections.abc import Sequence

from examexam import logging_config
from examexam.convert_to_pretty import run as convert_questions_run
from examexam.generate_questions import generate_questions_now
from examexam.take_exam import take_exam_now
from examexam.validate_questions import validate_questions_now


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="examexam", description="A CLI for generating, taking, and managing exams."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        required=False,
        help="Enable detailed logging.",
    )
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands", required=True
    )

    take_parser = subparsers.add_parser("take", help="Take an exam from a TOML file.")
    take_parser.add_argument(
        "--question-file",
        type=str,
        required=True,
        help="Path to the TOML question file.",
    )

    generate_parser = subparsers.add_parser(
        "generate", help="Generate new exam questions using an LLM."
    )
    generate_parser.add_argument(
        "--exam-name",
        type=str,
        required=True,
        help="The official name of the exam (e.g., 'Certified Kubernetes Administrator').",
    )
    generate_parser.add_argument(
        "--toc-file",
        type=str,
        required=True,
        help="Path to a text file containing the table of contents or topics, one per line.",
    )
    generate_parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to the output TOML file where questions will be saved.",
    )
    generate_parser.add_argument(
        "-n",
        type=int,
        default=5,
        help="Number of questions to generate per topic (default: 5).",
    )
    generate_parser.add_argument(
        "--model",
        type=str,
        default="gpt4",
        help="Model to use for generating questions (e.g., 'gpt4', 'claude'). Default: gpt4",
    )

    validate_parser = subparsers.add_parser(
        "validate", help="Validate exam questions using an LLM."
    )
    validate_parser.add_argument(
        "--question-file",
        type=str,
        required=True,
        help="Path to the TOML question file to validate.",
    )
    validate_parser.add_argument(
        "--exam-name",
        type=str,
        required=True,
        help="The official name of the exam, for context during validation.",
    )
    validate_parser.add_argument(
        "--model",
        type=str,
        default="claude",
        help="Model to use for validation (default: claude).",
    )

    convert_parser = subparsers.add_parser(
        "convert", help="Convert a TOML question file to Markdown and HTML formats."
    )
    convert_parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to the input TOML question file.",
    )
    convert_parser.add_argument(
        "--output-base-name",
        type=str,
        required=True,
        help="Base name for the output .md and .html files (e.g., 'my-exam').",
    )

    args = parser.parse_args(args=argv)

    if args.verbose:
        config = logging_config.generate_config(level="DEBUG")
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.command == "take":
        take_exam_now(question_file=args.question_file)
    elif args.command == "generate":
        generate_questions_now(
            questions_per_toc_topic=args.n,
            file_name=args.output_file,
            exam_name=args.exam_name,
            toc_file=args.toc_file,
            model=args.model,
        )
    elif args.command == "validate":
        validate_questions_now(
            file_name=args.question_file, exam_name=args.exam_name, model=args.model
        )
    elif args.command == "convert":
        md_path = f"{args.output_base_name}.md"
        html_path = f"{args.output_base_name}.html"
        convert_questions_run(
            toml_file_path=args.input_file,
            markdown_file_path=md_path,
            html_file_path=html_path,
        )
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

```

## File: apis\anthropic_calls.py

```python
import logging
import os
import sys
import time

import anthropic
from retry import retry

from examexam.apis.conversation_model import Conversation
from examexam.utils.custom_exceptions import ExamExamTypeError
from examexam.utils.env_loader import load_env

LOGGER = logging.getLogger(__name__)


load_env()

ANTHROPIC_SUPPORTED_MODELS = [
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "claude-2.1",
    "claude-2.0",
    "claude-instant-1.2",
]


FREE_PLAN = int(60 / 5) + 1
TIER_ONE = int(60 / 50) + 1
TIER_TWO = 0


class AnthropicCaller:
    def __init__(
        self,
        model: str,
        tokens: int,
        conversation: Conversation,
    ):
        if not model:
            raise ExamExamTypeError("Model required.")
        self.client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
        )
        self.model = model
        self.supported_models = ANTHROPIC_SUPPORTED_MODELS
        self.tokens = tokens

        self.conversation = conversation

    @retry(
        exceptions=anthropic.RateLimitError,
        tries=3,
        delay=5,
        jitter=(0.15, 0.23),
        backoff=1.5,
        logger=LOGGER,
    )
    def single_completion(self, prompt: str) -> str:
        if not prompt:
            raise ValueError("Prompt cannot be empty")
        self.conversation.prompt(prompt)
        try:

            if "pytest" not in sys.modules:

                time.sleep(TIER_TWO)
            message = self.client.messages.create(
                max_tokens=self.tokens,
                messages=self.conversation.without_system(),
                model=self.model,
                system=self.conversation.system,
            )

            LOGGER.info(f"Actual Anthropic token count {message.usage}")
            core_response = message.content[0].text
            self.conversation.response(core_response)

            return core_response
        except anthropic.RateLimitError as e:
            self.conversation.pop()

            if "pytest" not in sys.modules:

                time.sleep(TIER_ONE)
            LOGGER.info(f"Anthropic rate limit {e}")
            LOGGER.warning("A 429 status code was received; we should back off a bit.")
            raise
        except:
            self.conversation.pop()
            raise

```

## File: apis\bedrock_calls.py

```python
import json
import logging
import os
from typing import Any, Optional

import boto3

import examexam.apis.llama_templates as llama_templates
from examexam.apis.bedrock_models import TitanAnswers, TitanResponse
from examexam.apis.conversation_model import Conversation
from examexam.utils.custom_exceptions import ExamExamTypeError
from examexam.utils.env_loader import load_env


LOGGER = logging.getLogger(__name__)

load_env()


class BedrockCaller:
    def __init__(self, conversation: Conversation) -> None:

        self.bedrock_runtime_client = boto3.client(
            "bedrock-runtime",
            region_name="us-east-1",
            aws_access_key_id=os.environ["ACCESS_KEY"],
            aws_secret_access_key=os.environ["SECRET_KEY"],
        )
        self.bedrock_client = boto3.client(
            "bedrock",
            region_name="us-east-1",
            aws_access_key_id=os.environ["ACCESS_KEY"],
            aws_secret_access_key=os.environ["SECRET_KEY"],
        )
        self.supported_models = [
            "amazon.titan-text-express-v1",
            "meta.llama2-70b-chat-v1",
            "ai21.j2-ultra-v1",
            "cohere.command-text-v14",
            "mistral.mixtral-8x7b-instruct-v0:1",
        ]

        self.llama_convo: Optional[llama_templates.LlamaConvo] = None

        self.conversation = conversation

    def list_foundation_models(self) -> list[str]:
        response = self.bedrock_client.list_foundation_models()
        models = response["modelSummaries"]
        return models

    def model_info(self, model_id: str) -> Any:
        return self.bedrock_client.get_foundation_model(modelIdentifier=model_id)

    def titan_call(self, prompt: str) -> str:
        self.conversation.prompt(prompt)
        model = "amazon.titan-text-express-v1"
        final_prompt = self.conversation.templatize_conversation()
        response = self.titan(final_prompt, model, max_token_count=7999)
        core_response = response.results[0].output_text.strip("\n ")
        self.conversation.response(core_response)
        return core_response

    def titan(
        self,
        prompt: str,
        model: str,
        max_token_count: int = 512,
        deterministic: bool = False,
    ) -> TitanResponse:

        if max_token_count > 8000:
            raise ExamExamTypeError("max_token_count must be less than 8000")

        if deterministic:
            temperature = 0.0
        else:
            temperature = 0.5

        initial_length = len(prompt)
        max_tokens = 8000
        tokens_in_characters = int((5 * max_tokens) * 0.95)
        if tokens_in_characters < initial_length:
            LOGGER.warning(
                f"Input text is too long. Truncating to {tokens_in_characters} characters"
            )

        final_prompt = prompt[:tokens_in_characters]
        payload = {
            "inputText": final_prompt,
            "textGenerationConfig": {
                "maxTokenCount": 8000,
                "stopSequences": [],
                "temperature": temperature,
                "topP": 1.0,
            },
        }
        body = json.dumps(payload)

        response = self.bedrock_runtime_client.invoke_model(
            body=body,
            modelId=model,
            accept="application/json",
            contentType="application/json",
        )
        raw = json.loads(response.get("body").read())
        tokens = raw["inputTextTokenCount"]
        response = TitanResponse(
            tokens,
            [
                TitanAnswers(
                    answer.get("tokenCount"),
                    answer.get("outputText"),
                    answer.get("completionReason"),
                )
                for answer in raw["results"]
            ],
        )
        return response

    def llama2(self, prompt: str, deterministic: bool = False) -> str:
        self.conversation.prompt(prompt)
        if self.llama_convo is None:
            self.llama_convo = llama_templates.LlamaConvo(self.conversation.system, "")
        context = self.llama_convo.render()
        prompt = self.llama_convo.next(context, prompt)

        model = "meta.llama2-70b-chat-v1"

        if deterministic:
            temperature = 0.0
        else:
            temperature = 0.5

        max_tokens = 4096
        initial_length = len(prompt)
        tokens_in_characters = int((3.9 * max_tokens) * 0.90)

        if tokens_in_characters < initial_length:
            LOGGER.warning(
                f"Input text is too long. Truncating to {tokens_in_characters} characters"
            )

        body = json.dumps(
            {
                "prompt": prompt[:tokens_in_characters],
                "max_gen_len": 2048,
                "temperature": temperature,
                "top_p": 0.9,
            }
        )

        response = self.bedrock_runtime_client.invoke_model(
            body=body,
            modelId=model,
            accept="application/json",
            contentType="application/json",
        )
        response_body = json.loads(response.get("body").read())

        core_response = response_body.get("generation")
        self.conversation.response(core_response)

        return core_response

    def jurassic_call(self, prompt: str, deterministic: bool = False) -> str:
        self.conversation.prompt(prompt)
        model = "ai21.j2-ultra-v1"

        if deterministic:

            temperature = 0.0

        else:

            temperature = 0.5

        max_tokens = 5000
        initial_length = len(prompt)
        tokens_in_characters = int((5 * max_tokens) * 0.90)

        if tokens_in_characters < initial_length:
            LOGGER.warning(
                f"Input text is too long. Truncating to {tokens_in_characters} characters"
            )

        body = json.dumps(
            {
                "prompt": prompt[:tokens_in_characters],
                "maxTokens": max_tokens,
                "temperature": temperature,
                "topP": 1,
            }
        )

        response = self.bedrock_runtime_client.invoke_model(
            body=body,
            modelId=model,
            accept="application/json",
            contentType="application/json",
        )

        response_body = json.loads(response.get("body").read())
        core_response = response_body["completions"][0]["data"]["text"]
        self.conversation.response(
            core_response,
        )

        return core_response

    def cohere_call(
        self,
        prompt: str,
        stop_sequences: Optional[list[str]] = None,
        deterministic: bool = False,
    ) -> str:
        self.conversation.prompt(prompt)
        model = "cohere.command-text-v14"

        if deterministic:

            temperature = 0.0
        else:

            temperature = 1.0

        max_tokens = 4096
        initial_length = len(prompt)
        tokens_in_characters = int((5 * max_tokens) * 0.90)

        if tokens_in_characters < initial_length:
            LOGGER.warning(
                f"Input text is too long. Truncating to {tokens_in_characters} characters"
            )

        parameters = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "p": 1,
            "k": 0,
            "num_generations": 1,
            "return_likelihoods": "NONE",
        }
        if stop_sequences:
            parameters["stop_sequences"] = stop_sequences
        body = json.dumps(parameters)

        response = self.bedrock_runtime_client.invoke_model(
            body=body,
            modelId=model,
            accept="application/json",
            contentType="application/json",
        )
        json_text = json.loads(response["body"].read())
        generations = json_text.get("generations")
        core_response = generations[0]["text"]
        self.conversation.response(core_response)
        return core_response

    def mixtral_8x7b_call(self, prompt: str) -> str:

        self.conversation.prompt(prompt)
        model = "mistral.mixtral-8x7b-instruct-v0:1"

        mixtral_prompt = "<s>"
        for exchange in self.conversation.conversation[:-1]:
            if exchange["role"] in ("user", "system"):
                mixtral_prompt += exchange["content"] + " [/INST]\n"
            elif exchange["role"] == "assistant":
                mixtral_prompt += exchange["content"]
        mixtral_prompt += "</s>\n"
        mixtral_prompt += f"[INST] {prompt} [INST]\n"

        max_tokens = 4096
        initial_length = len(prompt)
        tokens_in_characters = int((5 * max_tokens) * 0.90)

        if tokens_in_characters < initial_length:
            LOGGER.warning(
                f"Input text is too long. Truncating to {tokens_in_characters} characters"
            )
        final_prompt = mixtral_prompt[:tokens_in_characters]

        body = {
            "prompt": final_prompt,
            "max_tokens": max_tokens,
            "temperature": 0.5,
        }

        response = self.bedrock_runtime_client.invoke_model(
            modelId=model, body=json.dumps(body)
        )

        response_body = json.loads(response["body"].read())
        outputs = response_body.get("outputs")

        completions = [output["text"] for output in outputs]
        core_response = completions[0]
        self.conversation.response(core_response)

        return core_response

```

## File: apis\bedrock_models.py

```python
from dataclasses import dataclass, field


@dataclass
class TitanAnswers:
    token_count: int
    output_text: str
    completion_reason: str
    """FINISHED or LENGTH"""


@dataclass
class TitanResponse:
    input_text_token_count: int
    results: list[TitanAnswers] = field(default_factory=list)

```

## File: apis\conversation_model.py

```python
import logging
from typing import Literal, Optional

from examexam.utils.custom_exceptions import ExamExamTypeError

LOGGER = logging.getLogger(__name__)


class FatalConversationError(Exception):



ConversationMode = Literal["no_context", "minimal_context", "full"]


class Conversation:
    def __init__(self, system: str, mode: ConversationMode, model: Optional[str] = None) -> None:
        self.system = system
        self.model = model
        self.conversation: list[dict[str, str]] = [
            {
                "role": "system",
                "content": system,
            },
        ]
        self.mode: ConversationMode = mode
        self.stub_conversation = []

    def reinitialize_stub(self) -> None:
        self.stub_conversation = [
            {
                "role": "system",
                "content": self.system,
            },
        ]

    def prompt(self, prompt: str, role: str = "user") -> dict[str, str]:
        if self.conversation[-1]["role"] == role:
            raise FatalConversationError("Prompting the same role twice in a row")
        self.conversation.append(
            {
                "role": role,
                "content": prompt,
            },
        )
        return self.conversation[-1]

    def convert_to(self) -> list[dict[str, str]]:










































        return self.conversation

    def error(self, error: Exception) -> dict[str, str]:
        self.conversation.append(
            {"role": "examexam", "content": str(error)},
        )
        return self.conversation[-1]

    def response(self, response: str, role: str = "assistant") -> dict[str, str]:
        if self.conversation[-1]["role"] == role:
            raise FatalConversationError("Prompting the same role twice in a row")


        if self.model not in ("gpt3.5", "gpt4", "claude", "gemini-pro"):
            response = clean_text(response)
        self.conversation.append(
            {
                "role": role,
                "content": response,
            },
        )
        return self.conversation[-1]

    def pop(self) -> None:
        self.conversation.pop()

    def without_system(self) -> list[dict[str, str]]:
        return [_ for _ in self.conversation if _["role"] != "system"]

    def templatize_conversation(self) -> str:

        conversation = self.conversation

        entire_conversation = ""
        for exchange in conversation:
            exchange_content = exchange["content"]


            if exchange_content is None:
                exchange_content = "**** Bot returned None, maybe API failed ****"
            elif exchange_content.strip() == "":
                exchange_content = "**** Bot returned whitespace ****"
            elif not exchange_content:
                exchange_content = f"**** Bot returned falsy value {exchange_content} ****"

            if exchange["role"] == "user":
                entire_conversation += f"User: {exchange_content}\n"
            elif exchange["role"] == "assistant":
                entire_conversation += f"Assistant: {exchange_content}\n"
            elif exchange["role"] == "system" and exchange["content"]:
                entire_conversation += f"User: {exchange_content}\n"
            elif exchange["role"] == "examexam":
                entire_conversation += f"ERROR: {exchange_content}\n"
            else:
                raise ExamExamTypeError(f"Unknown role {exchange['role']}")
        return entire_conversation


def clean_text(text: str) -> str:
    if not text:
        return ""


    prefixes = [
        "Assistant:",
        "Assistant: ",
        "Assistant:\n",
        "Assistant: \n",
        "Assistant: \n\n",
    ]




    loops = 0
    while any(text.startswith(prefix) for prefix in prefixes):
        loops += 1
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix) :]
                text = text.strip()
        if loops > 10:


            break
    return text.strip()



```

## File: apis\fakebot_calls.py

```python


import logging
import random
from typing import Literal, Optional

from examexam.apis.conversation_model import Conversation
from examexam.utils.custom_exceptions import ExamExamTypeError
from examexam.utils.env_loader import load_env

load_env()

LOGGER = logging.getLogger(__name__)

DATA = ["Answers: [1,2]\n---Blah blah. Bad."]

FakeBotModels = Literal["fakebot"]


class FakeBotException(ValueError):



class FakeBotCaller:
    def __init__(
        self,
        model: str,
        conversation: Conversation,
        data: Optional[list[str]] = None,
        reliable: bool = False,
    ):
        self.conversation = conversation
        self.model = model
        self.supported_models = ["fakebot"]
        if self.model not in self.supported_models:
            raise ExamExamTypeError(f"Caller doesn't support that model : {self.model}")
        if data is None:
            self.data = DATA
        else:
            self.data = data
        self.reliable = reliable
        self.invocation_count = 0

    def completion(self, prompt: str) -> str:
        self.invocation_count += 1
        self.conversation.prompt(prompt)

        self.conversation.templatize_conversation()

        if not self.reliable:
            if random.random() < 0.1:


                raise FakeBotException("Fakebot has failed to return an answer, just like a real API.")

        core_response = random.choice(self.data)


        LOGGER.info(core_response.replace("\n", "\\n"))

        self.conversation.response(core_response)

        return core_response

```

## File: apis\google_calls.py

```python
from typing import Optional

import google.generativeai as genai
from google.generativeai import ChatSession

from examexam.apis.conversation_model import Conversation
from examexam.utils.env_loader import load_env

load_env()

INITIALIZED = False


def initialize_google(force_initialization: bool = False):

    global INITIALIZED
    if INITIALIZED and not force_initialization:
        return
    try:

        from google.colab import userdata

        GOOGLE_API_KEY = userdata.get("GOOGLE_API_KEY")
    except ImportError:
        import os

        GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    genai.configure(api_key=GOOGLE_API_KEY)
    INITIALIZED = True


class GoogleCaller:
    def __init__(
        self,
        model: str,
        conversation: Conversation,
    ):
        initialize_google()
        self.model = model
        self.client = genai.GenerativeModel(
            model_name=self.model, system_instruction=conversation.system
        )
        self.chat: Optional[ChatSession] = None

        self.supported_models = ["gemini-1.0-pro-001"]
        self.conversation = conversation

    def converse(self, prompt: str) -> str:
        self.conversation.prompt(prompt)
        if not self.chat:
            self.chat = self.client.start_chat()
        response = self.chat.send_message(prompt)
        core_response = response.text

        return core_response

    def single_completion(self, prompt: str) -> str:

        self.conversation.prompt(prompt)

        self.chat = self.client.start_chat()

        message = self.conversation.system + "\n" + prompt

        response = self.chat.send_message(message)
        core_response = response.text

        return core_response

```

## File: apis\halting_checker.py

```python
import logging
from collections.abc import Callable

LOGGER = logging.getLogger(__name__)


class FailureToHaltError(Exception):

    def __init__(self, message: str) -> None:
        super().__init__(message)


def call_limit(limit: int) -> Callable:

    def decorator(func: Callable) -> Callable:

        func.call_count = 0

        def wrapper(*args, **kwargs) -> Callable:
            func.call_count += 1

            try:
                name = func.__name__
            except AttributeError:
                name = "unknown function"
            LOGGER.info(f"{name} called {func.call_count} times")

            if func.call_count > limit:

                raise FailureToHaltError(
                    f"{name} has been called more than {limit} times"
                )

            return func(*args, **kwargs)

        return wrapper

    return decorator


if __name__ == "__main__":

    @call_limit(3)
    def my_function():
        print("Function executed")

    for i in range(5):

        try:
            my_function()
            print(i)
        except FailureToHaltError as e:
            print(e)
            break

```

## File: apis\llama_templates.py

```python
from examexam.utils.custom_exceptions import ExamExamTypeError


class LlamaConvo:
    template = """<s>[INST] <<SYS>>
{your_system_message}
<</SYS>>

{user_message_1} [/INST]"""

    def __init__(self, your_system_message: str, user_message: str) -> None:
        if your_system_message is None or user_message is None:
            raise ExamExamTypeError("No args can be None")
        self.your_system_message = your_system_message
        self.user_message = user_message

    def render(self) -> str:
        return self.template.format(
            your_system_message=self.your_system_message,
            user_message_1=self.user_message,
        )

    def next(self, history: str, user_reply: str) -> str:
        if not history.endswith("</s><s>[INST]"):
            history += "</s><s>[INST]"
        return history + f" {user_reply} [/INST]"

    def __str__(self) -> str:
        return self.render()

```

## File: apis\model_router.py

```python
import logging
import sys
from collections.abc import Callable
from typing import Any, Optional, Union

import examexam.apis.openai_calls as ai
from examexam.apis import anthropic_calls
from examexam.apis.anthropic_calls import AnthropicCaller
from examexam.apis.bedrock_calls import BedrockCaller
from examexam.apis.conversation_model import Conversation, FatalConversationError
from examexam.apis.fakebot_calls import FakeBotCaller, FakeBotException
from examexam.apis.google_calls import GoogleCaller
from examexam.utils.benchmark_utils import log_duration
from examexam.utils.custom_exceptions import ExamExamTypeError, ExamExamValueError

LOGGER = logging.getLogger(__name__)


DEFAULT_BOT = {
    "jurassic": "",
    "cohere": "",
    "mixtral": "",
    "gpt4": "gpt-4o-mini",
    "claude": "claude-3-haiku-20240307",
    "fakebot": "fakebot",
}


class Router:

    def __init__(self, conversation: Conversation):

        self.most_recent_python: Optional[str] = None
        self.most_recent_answer: Optional[str] = None
        self.most_recent_json: Union[dict[str, Any], list[Any], None] = None
        self.most_recent_just_code: Optional[list[str]] = None
        self.most_recent_exception: Optional[Exception] = None

        self.openai_caller: Optional[ai.OpenAICaller] = None
        self.fakebot_caller: Optional[FakeBotCaller] = None
        self.bedrock_caller: Optional[BedrockCaller] = None
        self.anthropic_caller: Optional[anthropic_calls.AnthropicCaller] = None
        self.standard_conversation: Conversation = conversation
        self.errors_so_far = 0
        """If there are too many errors, stop."""
        self.conversation_cannot_continue = False

    def reset(self) -> None:
        self.most_recent_python = None
        self.most_recent_answer = None
        self.most_recent_json = None
        self.most_recent_just_code = None
        self.most_recent_exception = None

    def call_until(
        self, request: str, model: str, stop_check: Callable
    ) -> Optional[str]:

        answer = self.call(request, model)
        while not stop_check(answer):
            answer = self.call(request, model)
        return answer

    @log_duration
    def call(self, request: str, model: str, essential: bool = False) -> Optional[str]:

        if self.conversation_cannot_continue:
            raise ExamExamValueError(
                "Conversation cannot continue, an essential exchange previously failed."
            )
        if not request:
            raise ExamExamValueError("Request cannot be empty")
        if len(request) < 5:
            LOGGER.warning(
                f"Request ('{request}') is less than 5 characters, unless this is a interactive chat, that is probably wrong."
            )
        self.reset()
        LOGGER.info(f"Calling {model} with request of length {len(request)}")

        try:
            if model == "titan":
                if self.bedrock_caller is None:
                    self.bedrock_caller = BedrockCaller(self.standard_conversation)
                self.standard_conversation = self.bedrock_caller.conversation
                answer = self.bedrock_caller.titan_call(request)
            elif model == "llama":
                if not self.bedrock_caller:
                    self.bedrock_caller = BedrockCaller(
                        conversation=self.standard_conversation
                    )
                self.standard_conversation = self.bedrock_caller.conversation
                answer = self.bedrock_caller.llama2(request)
            elif model == "jurassic":
                prompt = request
                if not self.bedrock_caller:
                    self.bedrock_caller = BedrockCaller(
                        conversation=self.standard_conversation
                    )
                self.standard_conversation = self.bedrock_caller.conversation
                answer = self.bedrock_caller.jurassic_call(prompt)
            elif model == "cohere":
                prompt = request
                if not self.bedrock_caller:
                    self.bedrock_caller = BedrockCaller(
                        conversation=self.standard_conversation
                    )
                self.standard_conversation = self.bedrock_caller.conversation
                answer = self.bedrock_caller.cohere_call(prompt)
            elif model in ("gpt4",):

                if not self.openai_caller:
                    self.openai_caller = ai.OpenAICaller(
                        model=DEFAULT_BOT[model],
                        conversation=self.standard_conversation,
                    )
                self.standard_conversation = self.openai_caller.conversation
                answer = self.openai_caller.completion(request)

            elif model == "claude":
                if not self.anthropic_caller:
                    self.anthropic_caller = anthropic_calls.AnthropicCaller(
                        model=DEFAULT_BOT[model],
                        tokens=4096,
                        conversation=self.standard_conversation,
                    )

                self.standard_conversation = self.anthropic_caller.conversation
                answer = self.anthropic_caller.single_completion(request)

            elif model == "mixtral":
                if not self.bedrock_caller:
                    self.bedrock_caller = BedrockCaller(
                        conversation=self.standard_conversation
                    )
                self.standard_conversation = self.bedrock_caller.conversation
                answer = self.bedrock_caller.mixtral_8x7b_call(request)
            elif model == "gemini-pro":
                prompt = request
                google_caller = GoogleCaller(
                    model=DEFAULT_BOT[model],
                    conversation=self.standard_conversation,
                )
                answer = google_caller.single_completion(prompt)
            elif model == "fakebot":
                if not self.fakebot_caller:
                    self.fakebot_caller = FakeBotCaller(
                        model=DEFAULT_BOT[model],
                        conversation=self.standard_conversation,
                    )

                self.standard_conversation = self.fakebot_caller.conversation
                answer = self.fakebot_caller.completion(request)

            else:
                raise ExamExamValueError(f"Unknown model {model}")
        except FatalConversationError:
            raise
        except FakeBotException as e:
            if self.standard_conversation:
                self.standard_conversation.error(e)

            self.most_recent_exception = e

            if essential:
                self.conversation_cannot_continue = True

            self.errors_so_far += 1
            LOGGER.error(e)
            LOGGER.error(f"Error calling {model} with...{request[:15]}")
            self.most_recent_answer = ""
            return None
        except Exception as e:
            if self.standard_conversation:
                self.standard_conversation.error(e)

            self.most_recent_exception = e

            if essential:

                self.conversation_cannot_continue = True

            if "pytest" in sys.modules:

                raise

            self.errors_so_far += 1
            LOGGER.error(e)
            LOGGER.error(f"Error calling {model} with...{request[:15]}")
            self.most_recent_answer = ""
            return None

        self.most_recent_answer = answer
        return answer

    def persist_caller(self, model: str) -> None:
        if model in ("gpt3.5", "gpt4"):
            self.openai_caller = ai.OpenAICaller(
                model=DEFAULT_BOT[model],
                conversation=self.standard_conversation,
            )
        elif model == "fakebot":
            self.fakebot_caller = FakeBotCaller(
                model=model, conversation=self.standard_conversation
            )
        elif model in ("mixtral", "titan"):
            self.bedrock_caller = BedrockCaller(conversation=self.standard_conversation)
        elif model == "claude":
            self.anthropic_caller = AnthropicCaller(
                model=DEFAULT_BOT[model],
                tokens=4096,
                conversation=self.standard_conversation,
            )
        else:
            raise ExamExamTypeError(
                "Unit testing needs a stateful bot, currently just openapi and fakebot"
            )

```

## File: apis\openai_calls.py

```python
import logging
import os
from typing import Literal

import openai

from examexam.apis.conversation_model import Conversation
from examexam.apis.halting_checker import call_limit
from examexam.utils.env_loader import load_env

CLIENT = None


load_env()

LOGGER = logging.getLogger(__name__)


def get_client(force_new_client: bool = False) -> openai.OpenAI:

    global CLIENT
    if CLIENT is None or force_new_client:

        client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        CLIENT = client
    return CLIENT


OpenAIModels = Literal["gpt-4o-mini"]
OPENAI_SUPPORTED_MODELS = ["gpt-4o-mini"]


class OpenAICaller:
    def __init__(
        self,
        model: str,
        conversation: Conversation,
    ):
        self.conversation = conversation

        self.model = model
        self.client = get_client()

        self.supported_models = OPENAI_SUPPORTED_MODELS

    @call_limit(500)
    def completion(self, prompt: str) -> str:
        self.conversation.prompt(prompt)

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=self.conversation.conversation,
        )

        if completion.usage:
            LOGGER.info(
                f"Prompt/completion/total tokens: {completion.usage.prompt_tokens}"
                f"/{completion.usage.completion_tokens}/{completion.usage.total_tokens}"
            )

        core_response = completion.choices[0].message.content or ""
        role = completion.choices[0].message.role or ""
        self.conversation.response(core_response, role)

        return core_response


if __name__ == "__main__":

    def example():
        conversation = Conversation(
            "You are a python 3.11 developer.", "full", model="test-model"
        )
        caller = OpenAICaller(model="gpt-3.5-turbo-1106", conversation=conversation)

        result = caller.completion(
            "How do I display a character in a console at a specific location?"
        )
        print(result)
        result = caller.completion("I see, now how to unit test that?")
        print(result)

    example()

```

## File: apis\raw_log.py

```python
def format_conversation_to_markdown(
    conversation: list[dict[str, str]],
    user_label: str = "User",
    assistant_label: str = "Assistant",
) -> str:

    markdown_lines = []

    for message in conversation:
        role = message.get("role", "").capitalize()
        content = message.get("content", "")

        if role.lower() == "user":
            label = user_label
        elif role.lower() == "assistant":
            label = assistant_label
        elif role.lower() == "examexam":
            label = "LLM Build Error Message"
        elif role.lower() == "system":
            label = "System"
        else:
            label = role

        if content is None:
            content = f"**** {label} returned None, maybe API failed ****"
        elif content.strip() == "":
            content = f"**** {label} returned whitespace ****"
        elif not content:
            content = f"**** {label} returned falsy value {content} ****"

        markdown_line = f"{label}: {content}"
        markdown_lines.append(markdown_line)

    return "\n".join(markdown_lines)

```

## File: apis\__init__.py

```python
__all__ = [
    "AnthropicCaller",
    "BedrockCaller",
    "Conversation",
    "FakeBotCaller",
    "GoogleCaller",
]

from examexam.apis.anthropic_calls import AnthropicCaller
from examexam.apis.bedrock_calls import BedrockCaller
from examexam.apis.conversation_model import Conversation
from examexam.apis.fakebot_calls import FakeBotCaller
from examexam.apis.google_calls import GoogleCaller

```

## File: utils\benchmark_utils.py

```python
import io
import logging
import sys
import time
from functools import wraps
from typing import Any


class CaptureOutput:

    def __init__(self) -> None:
        self.old_output = None
        self.captured_output = None

    def __enter__(self) -> io.StringIO:

        self.captured_output = io.StringIO()
        self.old_output = sys.stdout
        sys.stdout = self.captured_output
        return self.captured_output

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:

        if self.old_output:
            sys.stdout = self.old_output
        if self.captured_output:
            self.captured_output.close()


def log_duration(func: Any) -> Any:

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = (end_time - start_time) * 1000

        if duration < 500:
            logging.info(f"{func.__name__} execution time: {duration:.2f} milliseconds")
        elif duration < 60000:
            logging.info(
                f"{func.__name__} execution time: {duration / 1000:.2f} seconds"
            )
        else:
            logging.info(
                f"{func.__name__} execution time: {duration / 60000:.2f} minutes"
            )
        return result

    return wrapper

```

## File: utils\custom_exceptions.py

```python
class ExamExamValueError(ValueError):



class ExamExamTypeError(TypeError):


```

## File: utils\env_loader.py

```python
import os
import sys
from typing import Optional

import dotenv


def load_env(dotenv_path: Optional[str] = None) -> None:

    try:
        dotenv.load_dotenv(dotenv_path)

    except Exception as e:

        print(f"Error loading .env file: {e}")
        print("Continuing without .env file.")
    if "pytest" in sys.modules:
        forbidden = [
            "ACCESS_KEY",
            "SECRET_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
        ]
        for key in forbidden:
            if key in os.environ:
                os.environ.pop(key)

```

## File: utils\__init__.py

```python

```

