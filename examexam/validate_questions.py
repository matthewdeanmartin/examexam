"""Example toml


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
- User-facing output uses the FrontendUI protocol (Rich for CLI).
- Developer logs go to logger.info / logger.debug via RichHandler.
- Avoids retry loops on fatal API errors (missing keys, auth, bad model).
"""

from __future__ import annotations

import csv
import logging
import os
import re
from io import StringIO
from pathlib import Path
from time import perf_counter, sleep
from typing import TYPE_CHECKING, Any

import dotenv
import rtoml as toml
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from examexam.apis.conversation_and_router import Conversation, Router
from examexam.apis.fatal_errors import fatal_if_misconfigured, is_fatal_message
from examexam.jinja_management import jinja_env

if TYPE_CHECKING:
    from examexam.ui_protocol import FrontendUI

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
        handlers=[
            RichHandler(
                rich_tracebacks=True, markup=True, show_time=False, show_level=True
            )
        ],
    )


# ----------------------------------------------------------------------------
# LLM helpers (fatal-error detection shared with generate_questions.py via
# examexam.apis.fatal_errors)
# ----------------------------------------------------------------------------


def _llm_call(
    prompt: str,
    model: str,
    system: str,
    *,
    max_retries: int = 2,
    retry_delay_seconds: float = 1.25,
) -> str | None:
    """Make a guarded LLM call with minimal retries and fatal detection."""
    fatal_if_misconfigured(model)

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
            if is_fatal_message(msg):
                logger.error("Fatal error detected; will not retry.")
                return None
            if attempts > max_retries:
                logger.error("Exceeded max retries (%d); giving up.", max_retries)
                return None
            sleep(retry_delay_seconds)


# ----------------------------------------------------------------------------
# Core utilities (unchanged logic, richer logs)
# ----------------------------------------------------------------------------


BANNED_OPTION_PATTERNS = (
    "all of the above",
    "none of the above",
)


def check_question_form(question: dict[str, Any]) -> list[str]:
    """Deterministic, non-LLM structural checks on a single question.

    Returns a list of problem descriptions; an empty list means the question
    passed every check. These run before any LLM call so obviously malformed
    questions are caught for free.
    """
    problems: list[str] = []
    text = question.get("question", "")
    options = question.get("options", [])

    if not text or not str(text).strip():
        problems.append("Question text is empty.")

    if len(options) < 2:
        problems.append(f"Question has fewer than 2 options ({len(options)}).")

    correct_options = [opt for opt in options if opt.get("is_correct")]
    incorrect_options = [opt for opt in options if not opt.get("is_correct")]

    if not correct_options:
        problems.append("Question has no option marked is_correct.")
    if not incorrect_options:
        problems.append(
            "Question has no incorrect options (every option is marked is_correct)."
        )

    # "Select N" phrasing in the stem should match the actual count of correct options.
    match = re.search(r"select\s+(\d+|one|two|three|four|five)", text, re.IGNORECASE)
    if match:
        word_to_num = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
        raw = match.group(1).lower()
        expected_count = word_to_num.get(raw, int(raw) if raw.isdigit() else None)
        if expected_count is not None and expected_count != len(correct_options):
            problems.append(
                f"Stem says 'select {raw}' but {len(correct_options)} option(s) are marked is_correct."
            )

    option_texts = [str(opt.get("text", "")).strip().lower() for opt in options]
    for pattern in BANNED_OPTION_PATTERNS:
        if any(pattern in text_ for text_ in option_texts):
            problems.append(f"Option text contains banned pattern: {pattern!r}.")

    seen: set[str] = set()
    for text_ in option_texts:
        if text_ and text_ in seen:
            problems.append(f"Duplicate option text within question: {text_!r}.")
        seen.add(text_)

    return problems


def _normalize_for_dedupe(text: str) -> str:
    """Lowercase and strip punctuation/whitespace for a cheap duplicate check."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def find_duplicate_questions(questions: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """Finds pairs of questions with identical normalized text, across the whole bank.

    This is intentionally a simple exact-match-after-normalization check, not
    fuzzy/embedding similarity — cheap, deterministic, and catches the common
    case of the LLM regenerating a near-verbatim question for another topic.
    """
    normalized: dict[str, int] = {}
    duplicates: list[tuple[int, int]] = []
    for idx, question in enumerate(questions):
        key = _normalize_for_dedupe(str(question.get("question", "")))
        if not key:
            continue
        if key in normalized:
            duplicates.append((normalized[key], idx))
        else:
            normalized[key] = idx
    return duplicates


def run_deterministic_checks(questions: list[dict[str, Any]]) -> dict[int, list[str]]:
    """Runs all deterministic (non-LLM) checks and returns problems keyed by question index."""
    problems_by_index: dict[int, list[str]] = {}
    for idx, question in enumerate(questions):
        problems = check_question_form(question)
        if problems:
            problems_by_index[idx] = problems

    for first_idx, second_idx in find_duplicate_questions(questions):
        problems_by_index.setdefault(second_idx, []).append(
            f"Duplicate of question at index {first_idx} (near-identical text)."
        )

    return problems_by_index


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
        answer_part = answer_part.replace("\u201c", "").replace("\u201d", "")
        answer_part = answer_part.replace('", "', "|").strip('"')
        answers = answer_part.strip("[] ").split("|")
        return [ans.strip("'\" ").strip("'\" ") for ans in answers]
    return []


_TOML_FENCE_RE = re.compile(
    r"(?ms)^(?:`{3,}|~{3,})\s*(?:toml)?\s*\n(.*?)\n(?:`{3,}|~{3,})\s*$"
)


def _extract_toml_block(content: str) -> dict[str, Any] | None:
    """Extract and parse the first fenced (or bare) TOML block in a response."""
    candidates: list[str] = []
    match = _TOML_FENCE_RE.search(content)
    if match:
        candidates.append(match.group(1))
    candidates.append(content)  # fall back to treating the whole response as TOML

    for candidate in candidates:
        try:
            return toml.loads(candidate)
        except Exception as e:  # noqa: BLE001
            logger.debug("Structured parse candidate failed: %s", e)
    return None


class ValidationInconclusive(Exception):
    """Raised when an LLM validation response could not be parsed after retries."""


def ask_llm(
    question: str,
    options: list[str],
    answers: list[str],
    model: str,
    system: str,
    *,
    max_retries: int = 2,
) -> list[str]:
    """Asks the LLM to answer a given question, expecting a structured TOML response.

    Retries with a corrective prompt on parse failure (mirrors the pattern used
    in generate_questions.py) instead of silently returning an empty answer set.
    """
    if "(Select" not in question:
        question = f"{question} (Select {len(answers)})"

    try:
        template = jinja_env.get_template("answer_question.md.j2")
        original_prompt = template.render(question=question, options=options)
    except Exception as e:
        logger.error(
            "Failed to load or render Jinja2 template 'answer_question.md.j2': %s", e
        )
        raise

    prompt = original_prompt
    last_content = ""
    for attempt in range(max_retries + 1):
        content = _llm_call(prompt, model=model, system=system)
        if content is None:
            raise ValidationInconclusive("LLM call returned no content after retries.")

        last_content = content.strip()
        logger.debug(
            "ask_llm raw content (attempt %d): %r", attempt + 1, last_content[:200]
        )
        parsed = _extract_toml_block(last_content)
        if parsed is not None and isinstance(parsed.get("answers"), list):
            return [str(a) for a in parsed["answers"]]

        logger.warning(
            "Attempt %d: could not parse structured answer from response.", attempt + 1
        )
        prompt = (
            "Your previous response was not a valid TOML block with an `answers` array. "
            'Respond with ONLY a TOML code block like ```toml\\nanswers = ["..."]\\n``` and nothing else.\n\n'
            + original_prompt
        )

    raise ValidationInconclusive(
        f"Could not parse a structured answer after {max_retries + 1} attempts, last content: {last_content[:120]!r}"
    )


def ask_if_bad_question(
    question: str,
    options: list[str],
    answers: list[str],
    model: str,
    *,
    max_retries: int = 2,
) -> tuple[str, str]:
    """Asks the LLM to evaluate if a question is Good or Bad, expecting structured TOML."""
    try:
        template = jinja_env.get_template("evaluate_question.md.j2")
        original_prompt = template.render(
            question=question, options=options, answers=answers
        )
    except Exception as e:
        logger.error(
            "Failed to load or render Jinja2 template 'evaluate_question.md.j2': %s", e
        )
        raise

    system = "You are a test reviewer and are validating questions."
    prompt = original_prompt
    last_content = ""
    for attempt in range(max_retries + 1):
        content = _llm_call(prompt, model=model, system=system)
        if content is None:
            raise ValidationInconclusive("LLM call returned no content after retries.")

        last_content = content.strip()
        logger.debug(
            "ask_if_bad_question raw content (attempt %d): %r",
            attempt + 1,
            last_content[:200],
        )
        parsed = _extract_toml_block(last_content)
        if parsed is not None and isinstance(parsed.get("verdict"), str):
            verdict = parsed["verdict"].strip().lower()
            if verdict in ("good", "bad"):
                return verdict, str(parsed.get("reason", ""))

        logger.warning(
            "Attempt %d: could not parse structured verdict from response.", attempt + 1
        )
        prompt = (
            'Your previous response was not a valid TOML block with a `verdict` of "good" or "bad". '
            'Respond with ONLY a TOML code block like ```toml\\nverdict = "good"\\nreason = "..."\\n``` and nothing else.\n\n'
            + original_prompt
        )

    raise ValidationInconclusive(
        f"Could not parse a structured verdict after {max_retries + 1} attempts, last content: {last_content[:120]!r}"
    )


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
    responses: list[list[str] | None],
    good_bad: list[tuple[str, str] | None],
    file_path: Path,
    model: str,
    ui: FrontendUI,
    form_problems: dict[int, list[str]] | None = None,
) -> float:
    """Grades the LLM's performance and writes results to a TOML file.

    A `None` entry in `responses`/`good_bad` means that question's validation
    was inconclusive (the LLM response couldn't be parsed even after retries) —
    it is recorded distinctly and excluded from the score, rather than being
    counted as simply "wrong".
    """
    form_problems = form_problems or {}
    score = 0
    scored_total = 0
    total = len(questions)
    questions_to_write: list[dict[str, Any]] = []
    failures: list[tuple[str, str, set[str], set[str]]] = (
        []
    )  # (id, question, correct, given)
    inconclusive: list[str] = []

    for idx, (question, response, opinion) in enumerate(
        zip(questions, responses, good_bad, strict=True)
    ):
        correct_answers = {
            opt["text"] for opt in question.get("options", []) if opt.get("is_correct")
        }
        new_question_data = {
            k: v for k, v in question.items() if not _is_array_of_tables(v)
        }

        if response is None or opinion is None:
            inconclusive.append(question.get("id", "<no-id>"))
            new_question_data[f"{model}_answers"] = None
            new_question_data["good_bad"] = "inconclusive"
            new_question_data["good_bad_why"] = (
                "Validation LLM response could not be parsed after retries."
            )
        else:
            given_answers = set(response)
            scored_total += 1
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
            new_question_data[f"{model}_answers"] = sorted(given_answers)
            new_question_data["good_bad"], new_question_data["good_bad_why"] = opinion

        new_question_data["form_problems"] = form_problems.get(idx, [])

        for k, v in question.items():
            if _is_array_of_tables(v):
                new_question_data[k] = v

        questions_to_write.append(new_question_data)

    # Write results next to the original file
    out_path = file_path
    with open(out_path, "w", encoding="utf-8") as file:
        toml.dump({"questions": questions_to_write}, file)

    # Pretty print summary
    ui.show_panel(
        f"Final Score: {score} / {scored_total} (of {total} total)",
        title="Grading Summary",
    )

    if inconclusive:
        ui.show_message(
            f"Inconclusive ({len(inconclusive)}, excluded from score): {', '.join(inconclusive[:25])}"
            + (f" (+{len(inconclusive) - 25} more)" if len(inconclusive) > 25 else "")
        )

    if form_problems:
        form_lines = [f"Deterministic form issues ({len(form_problems)} question(s)):"]
        for idx, problems in list(form_problems.items())[:25]:
            qid = questions[idx].get("id", "<no-id>")
            form_lines.append(f"  {qid}: {'; '.join(problems)}")
        if len(form_problems) > 25:
            form_lines.append(f"  (+{len(form_problems) - 25} more not shown)")
        ui.show_message("\n".join(form_lines))

    if failures:
        # For non-Rich UIs, show as simple text; Rich UI will still get nice output
        failure_lines = [f"Incorrect ({len(failures)}):"]
        for qid, qtext, correct, given in failures[:25]:
            failure_lines.append(f"  {qid}: {qtext[:60]}...")
            failure_lines.append(f"    Correct: {' | '.join(sorted(correct))}")
            failure_lines.append(f"    Given:   {' | '.join(sorted(given))}")
        if len(failures) > 25:
            failure_lines.append(f"  (+{len(failures) - 25} more not shown)")
        ui.show_message("\n".join(failure_lines))

    return 0 if scored_total == 0 else score / scored_total


def validate_questions_now(
    file_name: str,
    model: str = "claude",
    ui: FrontendUI | None = None,
) -> float:
    """Main function to orchestrate the validation process with Rich progress."""
    # Default to Rich CLI if no UI provided
    if ui is None:
        from examexam.frontends.rich_ui import RichUI

        ui = RichUI()

    file_path = Path(file_name)
    if not file_path.exists():
        ui.show_panel(f"TOML file not found: {file_name}", title="Error", style="red")
        return 0.0

    questions = read_questions(file_path)
    total = len(questions)
    if total == 0:
        ui.show_panel(
            "No questions found in TOML.", title="Nothing to validate", style="yellow"
        )
        return 0.0

    ui.show_rule("Exam Question Validation")
    ui.show_message(f"Validating {total} questions using model {model}...")

    form_problems = run_deterministic_checks(questions)
    if form_problems:
        ui.show_message(
            f"Deterministic form checks flagged {len(form_problems)}/{total} question(s) "
            "(see form_problems in output file)."
        )

    responses: list[list[str] | None] = []
    opinions: list[tuple[str, str] | None] = []

    console = Console()
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("\u2022"),
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
            correct_answer_texts = [
                opt.get("text", "") for opt in opts if opt.get("is_correct")
            ]

            # Show per-question task
            desc = f"{idx}/{total} answering"
            q_task = progress.add_task(desc, total=2)  # step 1: answer, step 2: review

            try:
                resp = ask_llm(
                    q,
                    option_texts,
                    correct_answer_texts,
                    model,
                    system="You are test evaluator.",
                )
                responses.append(resp)
            except ValidationInconclusive as e:
                logger.error("ask_llm inconclusive: %s", e)
                responses.append(None)
            finally:
                progress.advance(q_task)

            try:
                op = ask_if_bad_question(q, option_texts, correct_answer_texts, model)
                opinions.append(op)
            except ValidationInconclusive as e:
                logger.error("ask_if_bad_question inconclusive: %s", e)
                opinions.append(None)
            finally:
                progress.update(q_task, description=f"{idx}/{total} reviewed")
                progress.advance(q_task)
                progress.advance(overall_task)

    score = grade_test(
        questions,
        responses,
        opinions,
        file_path,
        model,
        ui,
        form_problems=form_problems,
    )
    ui.show_rule()
    return score


if __name__ == "__main__":
    # Example usage; update paths as needed.
    validate_questions_now(
        file_name="personal_multiple_choice_tests.toml",
        model="claude",
    )
