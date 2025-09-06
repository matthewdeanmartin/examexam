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

Rich-enhanced validator for multiple-choice TOML question sets.
- User-facing output uses Rich (progress bars, panels, tables).
- Developer logs go to logger.info / logger.debug via RichHandler.
- Avoids retry loops on fatal API errors (missing keys, auth, bad model).
"""

from __future__ import annotations

import csv
import logging
import os
from io import StringIO
from pathlib import Path
from time import perf_counter, sleep
from typing import Any

import dotenv
import rtoml as toml
from jinja2 import Environment, FileSystemLoader, PackageLoader
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from examexam.apis.conversation_and_router import Conversation, Router
from examexam.utils.custom_exceptions import ExamExamTypeError

# ----------------------------------------------------------------------------
# Env & logging setup
# ----------------------------------------------------------------------------
# Load environment variables (e.g., OPENAI_API_KEY / ANTHROPIC_API_KEY)
dotenv.load_dotenv()

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, markup=True, show_time=False, show_level=True)],
    )

console = Console()


# ----------------------------------------------------------------------------
# Jinja2 Template Loading
# ----------------------------------------------------------------------------
def get_jinja_env() -> Environment:
    """
    Initializes and returns a Jinja2 Environment.
    It checks for a local 'prompts' directory for development,
    otherwise it assumes the package is installed and uses the package loader.
    """
    # Dev mode: Check if a 'prompts' directory exists relative to this project's root
    dev_prompts_path = Path(__file__).parent.parent / "prompts"
    if dev_prompts_path.is_dir():
        # Running from source tree
        logger.debug("Loading Jinja2 templates from filesystem: %s", dev_prompts_path)
        loader = FileSystemLoader(dev_prompts_path)
        return Environment(loader=loader, autoescape=False)  # nosec
    else:
        # Installed package
        logger.debug("Loading Jinja2 templates from installed package 'examexam.prompts'")
        # The package name is 'examexam' and the templates are in a 'prompts' subdir
        return Environment(loader=PackageLoader("examexam", "prompts"), autoescape=False)  # nosec  # nosec


# Create a single environment instance to be used by the module
jinja_env = get_jinja_env()


# ----------------------------------------------------------------------------
# Fatal error detection & LLM helpers
# ----------------------------------------------------------------------------
class FatalLLMError(Exception):
    """Raised for obviously fatal misconfigurations (missing API keys, etc.)."""


def _fatal_precheck(model: str) -> None:
    """Detect common fatal cases before calling the LLM so we don't loop."""
    m = model.lower()
    if m in {"fakebot", "none", "noop"}:
        return
    if "gpt" in m and not os.getenv("OPENAI_API_KEY"):
        raise FatalLLMError("OPENAI_API_KEY is not set for OpenAI model.")
    if "claude" in m and not os.getenv("ANTHROPIC_API_KEY"):
        raise FatalLLMError("ANTHROPIC_API_KEY is not set for Claude model.")


def _is_fatal_message(msg: str) -> bool:
    msg = msg.lower()
    markers = [
        "api_key client option must be set",
        "no api key",
        "invalid api key",
        "unauthorized",
        "model not found",
        "does not exist or you do not have access",
        "access denied",
    ]
    return any(k in msg for k in markers)


def _llm_call(
    prompt: str,
    model: str,
    system: str,
    *,
    max_retries: int = 2,
    retry_delay_seconds: float = 1.25,
) -> str | None:
    """Make a guarded LLM call with minimal retries and fatal detection."""
    _fatal_precheck(model)

    conversation = Conversation(system=system)
    router = Router(conversation)

    attempts = 0
    while True:
        attempts += 1
        try:
            t0 = perf_counter()
            content = router.call(prompt, model)
            dt = perf_counter() - t0
            logger.debug("router.call returned len=%d in %.2fs", len(content or ""), dt)
            return content
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            logger.error("Error calling %s: %s", model, msg)
            if _is_fatal_message(msg):
                logger.error("Fatal error detected; will not retry.")
                return None
            if attempts > max_retries:
                logger.error("Exceeded max retries (%d); giving up.", max_retries)
                return None
            sleep(retry_delay_seconds)


# ----------------------------------------------------------------------------
# Core utilities (unchanged logic, richer logs)
# ----------------------------------------------------------------------------


def read_questions(file_path: Path) -> list[dict[str, Any]]:
    """Reads a TOML file and returns the list of questions."""
    logger.debug("Reading TOML questions from %s", file_path)
    with open(file_path, encoding="utf-8") as file:
        data = toml.load(file)
    questions = data.get("questions", [])
    logger.info("Loaded %d questions", len(questions))
    return questions


def parse_answer(answer: str) -> list[str]:
    """Parses the string response from the LLM to extract the answers."""
    if answer.startswith("Answers:"):
        answer = answer[8:]
        if ("','" in answer or "', '" in answer or '","' in answer or '", "' in answer) and "|" not in answer:
            return parse_quote_lists(answer)

        if "[" in answer and "]" in answer:
            after_square_bracket = answer.split("[")[1]
            answer_part = after_square_bracket.split("]")[0]

            answer_part = answer_part.replace('", "', "|").strip('"')
            answers = answer_part.strip().strip("[]").split("|")
            return [ans.strip("'\" ").strip("'\" ") for ans in answers]
    return []


def parse_quote_lists(answer: str) -> list[str]:
    """Helper function to parse comma-separated, quoted lists."""
    if "[" in answer and "]" in answer:
        after_square_bracket = answer.split("[")[1]
        answer_part = after_square_bracket.split("]")[0]

        if "', '" in answer_part or '","' in answer_part:
            answer_part_io = StringIO(answer_part)
            reader = csv.reader(answer_part_io, delimiter=",")
            answers = next(reader)
            return answers

        # Clean odd quotes sometimes returned by models
        answer_part = answer_part.replace("â€˜", "").replace("â€™", "")
        answer_part = answer_part.replace('", "', "|").strip('"')
        answers = answer_part.strip("[] ").split("|")
        return [ans.strip("'\" ").strip("'\" ") for ans in answers]
    return []


def ask_llm(question: str, options: list[str], answers: list[str], model: str, system: str) -> list[str]:
    """Asks the LLM to answer a given question."""
    if "(Select" not in question:
        question = f"{question} (Select {len(answers)})"

    try:
        template = jinja_env.get_template("answer_question.md.j2")
        prompt = template.render(question=question, options=options)
    except Exception as e:
        logger.error("Failed to load or render Jinja2 template 'answer_question.md.j2': %s", e)
        raise

    content = _llm_call(prompt, model=model, system=system)
    if content is None:
        logger.debug("ask_llm returned None content; treating as no answer")
        return []

    content = content.strip()
    logger.debug("ask_llm raw content: %r", content[:200])
    if content.startswith("Answers:"):
        parsed = parse_answer(content)
        logger.debug("ask_llm parsed answers: %s", parsed)
        return parsed
    raise ExamExamTypeError(f"Unexpected response format, didn't start with Answers:, got {content[:120]!r}")


def ask_if_bad_question(question: str, options: list[str], answers: list[str], model: str) -> tuple[str, str]:
    """Asks the LLM to evaluate if a question is Good or Bad."""
    try:
        template = jinja_env.get_template("evaluate_question.md.j2")
        prompt = template.render(question=question, options=options, answers=answers)
    except Exception as e:
        logger.error("Failed to load or render Jinja2 template 'evaluate_question.md.j2': %s", e)
        raise

    system = "You are a test reviewer and are validating questions."
    content = _llm_call(prompt, model=model, system=system)
    if content is None:
        return "bad", "**** Bot returned None, maybe API failed ****"

    content = content.strip()
    logger.debug("ask_if_bad_question raw content: %r", content[:200])
    if "---" in content:
        return parse_good_bad(content)
    raise ExamExamTypeError(f"Unexpected response format, didn't contain '---'. got {content[:120]!r}")


def parse_good_bad(answer: str) -> tuple[str, str]:
    """Parses the good/bad response from the LLM."""
    parts = answer.split("---")
    why = parts[0]
    good_bad = parts[1].strip(" \n").lower()
    if "good" in good_bad:
        return "good", why
    return "bad", why


# You will need this helper function inside grade_test or at the module level


def _is_array_of_tables(val: Any) -> bool:
    return isinstance(val, list) and (len(val) == 0 or isinstance(val[0], dict))


# ----------------------------------------------------------------------------
# Grading & Orchestration with Rich progress
# ----------------------------------------------------------------------------


def grade_test(
    questions: list[dict[str, Any]],
    responses: list[list[str]],
    good_bad: list[tuple[str, str]],
    file_path: Path,
    model: str,
) -> float:
    """Grades the LLM's performance and writes results to a TOML file."""
    score = 0
    total = len(questions)
    questions_to_write: list[dict[str, Any]] = []
    failures: list[tuple[str, str, set[str], set[str]]] = []  # (id, question, correct, given)

    for question, response, opinion in zip(questions, responses, good_bad, strict=True):
        correct_answers = {opt["text"] for opt in question.get("options", []) if opt.get("is_correct")}
        given_answers = set(response)

        if correct_answers == given_answers:
            score += 1
        else:
            failures.append(
                (
                    question.get("id", "<no-id>"),
                    question.get("question", "<no-question>"),
                    correct_answers,
                    given_answers,
                )
            )

        # Build new question dict without mutating original, scalars first.
        new_question_data = {k: v for k, v in question.items() if not _is_array_of_tables(v)}
        new_question_data[f"{model}_answers"] = sorted(list(given_answers))
        new_question_data["good_bad"], new_question_data["good_bad_why"] = opinion

        for k, v in question.items():
            if _is_array_of_tables(v):
                new_question_data[k] = v

        questions_to_write.append(new_question_data)

    # Write results next to the original file
    out_path = file_path
    with open(out_path, "w", encoding="utf-8") as file:
        toml.dump({"questions": questions_to_write}, file)

    # Pretty print summary
    console.print(Panel.fit(f"Final Score: [bold]{score}[/] / [bold]{total}[/]", title="Grading Summary"))

    if failures:
        table = Table(title=f"Incorrect ({len(failures)})", show_lines=False)
        table.add_column("ID", no_wrap=True, overflow="fold")
        table.add_column("Question", overflow="fold")
        table.add_column("Correct", overflow="fold")
        table.add_column("Given", overflow="fold")
        for qid, qtext, correct, given in failures[:25]:  # show top 25 to keep output readable
            table.add_row(str(qid), qtext, " | ".join(sorted(correct)), " | ".join(sorted(given)))
        if len(failures) > 25:
            table.caption = f"(+{len(failures) - 25} more not shown)"
        console.print(table)

    return 0 if total == 0 else score / total


def validate_questions_now(
    file_name: str,
    model: str = "claude",
) -> float:
    """Main function to orchestrate the validation process with Rich progress."""
    file_path = Path(file_name)
    if not file_path.exists():
        console.print(Panel.fit(f"[red]TOML file not found:[/] {file_name}", title="Error"))
        return 0.0

    questions = read_questions(file_path)
    total = len(questions)
    if total == 0:
        console.print(Panel.fit("[yellow]No questions found in TOML.[/]", title="Nothing to validate"))
        return 0.0

    console.rule("[bold]Exam Question Validation")
    console.print(f"Validating [bold]{total}[/] questions using model [italic]{model}[/]…\n")

    responses: list[list[str]] = []
    opinions: list[tuple[str, str]] = []

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("ETA:"),
        TimeRemainingColumn(),
        expand=True,
        console=console,
        transient=False,
    )

    with progress:
        overall_task = progress.add_task("Overall", total=total)

        for idx, question_data in enumerate(questions, start=1):
            q = question_data.get("question", "<no-question>")
            opts = question_data.get("options", [])
            option_texts = [opt.get("text", "") for opt in opts]
            correct_answer_texts = [opt.get("text", "") for opt in opts if opt.get("is_correct")]

            # Show per-question task
            desc = f"{idx}/{total} answering"
            q_task = progress.add_task(desc, total=2)  # step 1: answer, step 2: review

            try:
                resp = ask_llm(q, option_texts, correct_answer_texts, model, system="You are test evaluator.")
                responses.append(resp)
            except ExamExamTypeError as e:
                logger.error("ask_llm parse error: %s", e)
                responses.append([])
            finally:
                progress.advance(q_task)

            try:
                op = ask_if_bad_question(q, option_texts, correct_answer_texts, model)
                opinions.append(op)
            except ExamExamTypeError as e:
                logger.error("ask_if_bad_question parse error: %s", e)
                opinions.append(("bad", "**** parse error ****"))
            finally:
                progress.update(q_task, description=f"{idx}/{total} reviewed")
                progress.advance(q_task)
                progress.advance(overall_task)

    score = grade_test(questions, responses, opinions, file_path, model)
    console.rule()
    return score


if __name__ == "__main__":
    # Example usage; update paths as needed.
    validate_questions_now(
        file_name="personal_multiple_choice_tests.toml",
        model="claude",
    )
