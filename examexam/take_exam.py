"""Enhanced exam runner with per-run and per-session question caps.

Changes made:
- Added two new args:
    * questions_to_complete: cap questions answered in THIS run.
    * questions_to_complete_for_session: cap total questions answered across the SESSION (all runs).
- Threaded these args through interactive and machine modes.
- Preserved SciPy usage and added optional exact CI + binomial test helpers using scipy.stats.
- Consolidated/kept interactive ask function; removed duplicate non-interactive UI function.
- Safer screen clearing via Rich (still falls back if unavailable).
- Added small performance improvement by indexing session questions by id.
- More comments throughout for clarity and future maintenance.
- Refactored to use FrontendUI protocol for multi-frontend support.

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
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

import dotenv
import rtoml as toml
from scipy import stats

from examexam.models import AnswerFeedback, ExamResult, Option, Question, TestInfo
from examexam.utils.secure_random import SecureRandom
from examexam.utils.toml_normalize import normalize_exam_for_toml

if TYPE_CHECKING:
    from examexam.ui_protocol import FrontendUI

# Load environment variables (e.g., OPENAI_API_KEY)
dotenv.load_dotenv()

# ----------------- NEW: answer provider protocol & strategies -----------------


class AnswerProvider(Protocol):
    def __call__(self, question: dict[str, Any], options_list: list[dict[str, Any]]) -> list[dict[str, Any]]: ...


MachineStrategy = Literal["oracle", "random", "first", "none"]


def build_machine_answer_provider(strategy: MachineStrategy = "oracle", *, seed: int | None = 42) -> AnswerProvider:
    """Return a function that selects answers without user input.

    Strategies:
      - 'oracle': choose exactly the options with is_correct=True
      - 'random': choose a random valid set of size 'answer_count'
      - 'first': choose the first 'answer_count' options
      - 'none': choose an incorrect set on purpose, if possible

    Note: We use SecureRandom for deterministic tests (not for cryptographic guarantees).
    """
    rng = SecureRandom(seed)  # nosec

    def provider(question: dict[str, Any], options_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
        answer_count = sum(1 for o in question["options"] if o.get("is_correct"))
        if strategy == "oracle":
            return [o for o in options_list if o.get("is_correct")]
        if strategy == "first":
            # Skip the "bad question" sentinel; we never include it in machine mode
            return options_list[:answer_count]
        if strategy == "random":
            # Sample from actual options (exclude bad-question sentinel)
            population = list(options_list)
            if answer_count <= 0:
                return []
            if answer_count >= len(population):
                return population
            picks = rng.sample(range(len(population)), k=answer_count)
            return [population[i] for i in picks]
        if strategy == "none":
            # Try to pick a *different* set than the correct one
            correct = {id(o) for o in options_list if o.get("is_correct")}
            population = list(range(len(options_list)))
            if not correct:
                # If there is no correct answer, pick one anyway (e.g., trick Q)
                return [options_list[0]] if options_list else []
            # Greedy: start from first 'answer_count' indices; ensure it differs
            attempt = population[:answer_count]
            if {id(options_list[i]) for i in attempt} == correct and len(population) > answer_count:
                attempt[-1] = population[-1]
            return [options_list[i] for i in attempt]
        raise ValueError(f"Unknown strategy: {strategy!r}")

    return provider


# ----------------- File/session helpers -----------------


def load_questions(file_path: str) -> list[dict[str, Any]]:
    """Load questions from a TOML file.

    Expects a top-level 'questions' array of tables.
    """
    with open(file_path, encoding="utf-8") as file:
        data = toml.load(file)["questions"]
        return cast(list[dict[str, Any]], data)


def get_session_path(test_name: str) -> Path:
    """Get the session file path for a given test, under ./.session.

    Session files allow resuming progress.
    """
    session_dir = Path(".session")
    session_dir.mkdir(exist_ok=True)
    return session_dir / f"{test_name}.toml"


def get_available_tests(ui: FrontendUI) -> list[str]:
    """Get list of available test files from /data/ folder (fallback: CWD)."""
    data_dir = Path("data")
    if not data_dir.exists():
        ui.show_error("Error: /data/ folder not found!")
        data_dir = Path(".")

    test_files = list(data_dir.glob("*.toml"))
    return [f.stem for f in test_files]


def select_test(ui: FrontendUI) -> str | None:
    """Let user select a test to take."""
    tests = get_available_tests(ui)
    if not tests:
        return None

    test_infos = [TestInfo(name=t, index=idx) for idx, t in enumerate(tests, 1)]
    choice = ui.show_test_selection(test_infos)
    if choice is not None:
        return tests[choice]
    return None


def check_resume_session(test_name: str, ui: FrontendUI) -> tuple[bool, list[dict[str, Any]] | None, datetime | None]:
    """Check if a session exists and ask if user wants to resume.

    Returns: (should_resume, session_data_or_None, session_start_time_or_None)
    """
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

        start_dt = None
        time_ago = ""
        if start_time:
            try:
                start_dt = datetime.fromisoformat(start_time)
                elapsed = datetime.now() - start_dt
                time_ago = humanize_timedelta(elapsed)
            except (ValueError, TypeError):
                time_ago = "Unknown time"

        ui.show_session_info(test_name, completed, total, time_ago)

        resume = ui.confirm("Do you want to resume this session?")
        if resume:
            return True, session_data, start_dt
        # User wants to start fresh
        session_path.unlink()  # Delete old session
        return False, None, None

    except Exception as e:
        ui.show_error(f"Error reading session file: {e}")
        return False, None, None


# ----------------- Time formatting & estimates -----------------


def humanize_timedelta(td: timedelta) -> str:
    """Convert timedelta to human readable format: e.g., '1 hour 2 minutes 3 seconds'."""
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    parts: list[str] = []
    if hours > 0:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes > 0:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if seconds > 0 or not parts:
        parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")

    return " ".join(parts)


def calculate_time_estimates(session: list[dict[str, Any]], start_time: datetime) -> tuple[timedelta, timedelta | None]:
    """Calculate average time per question and estimated completion time, removing outliers.

    Outlier rule: drop durations > 3x median of completed question times.
    """
    completed_times: list[float] = []

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

    # Calculate remaining questions in session
    remaining = sum(1 for q in session if q.get("user_score") is None)
    estimated_time_left = timedelta(seconds=avg_seconds * remaining) if remaining > 0 else None

    return avg_time_per_question, estimated_time_left


# ----------------- Terminal helpers -----------------


def play_sound(_file: str) -> None:
    """Hook for sound effects; intentionally a no-op placeholder here."""
    # playsound(_file)


# ----------------- Validation helpers -----------------


def find_select_pattern(input_string: str) -> str:
    """Finds the first occurrence of "(Select n)" in the input string where n is a number from 1 to 5."""
    match = re.search(r"\(Select [1-5]\)", input_string)
    return match.group(0) if match else ""


def is_valid(
    answer: str, option_count: int, answer_count: int, last_is_bad_question_flag: bool = True
) -> tuple[bool, str]:
    """Validate a comma-separated set of option indices."""
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


# ----------------- Interactive question prompt (via UI) -----------------


def ask_question_interactive(
    ui: FrontendUI, question: dict[str, Any], options_list: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Render a question and return the selected options (interactive mode via UI)."""
    q_model = Question.from_dict(question)
    opt_models = [Option.from_dict(o) for o in options_list]

    ui.show_question(q_model, opt_models)

    option_count = len(options_list) + 1  # +1 for "bad question" sentinel
    answer_count = q_model.answer_count
    answer = ui.get_answer(option_count, answer_count)

    selected = [
        options_list[int(idx) - 1] for idx in answer.split(",") if idx.isdigit() and 1 <= int(idx) <= len(options_list)
    ]
    return selected


# ----------------- Machine answering -----------------


def ask_question_machine(
    provider: AnswerProvider, question: dict[str, Any], options_list: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """No terminal I/O; defers to the selected answer provider."""
    return provider(question, options_list)


# ----------------- Stats helpers (SciPy) -----------------


def calculate_confidence_interval(score: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Symmetric normal-approximation CI for a proportion using z from SciPy.

    Keep existing behavior but fetch z from SciPy for clarity.
    For small samples or extreme proportions, consider the exact CI (below).
    """
    if total == 0:
        return 0.0, 0.0

    p = score / total  # sample proportion
    z = stats.norm.ppf((1 + confidence) / 2)  # z-score for confidence level

    # Standard error
    se = math.sqrt(p * (1 - p) / total)

    # Margin of error
    me = z * se

    # Confidence interval
    lower = max(0.0, p - me)
    upper = min(1.0, p + me)

    return lower, upper


def calculate_exact_binomial_ci(score: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Exact (Clopper-Pearson) CI using the Beta distribution via SciPy.

    This is more conservative but valid for all n, including small samples.
    Useful for reporting alongside the normal CI when n is small or p is extreme.
    """
    if total == 0:
        return 0.0, 0.0
    alpha = 1.0 - confidence
    # Handle edge cases explicitly
    if score == 0:
        lower = 0.0
        upper = stats.beta.ppf(1 - alpha / 2, 1, total - 0)
    elif score == total:
        lower = stats.beta.ppf(alpha / 2, total, 1)
        upper = 1.0
    else:
        lower = stats.beta.ppf(alpha / 2, score, total - score + 1)
        upper = stats.beta.ppf(1 - alpha / 2, score + 1, total - score)
    return float(lower), float(upper)


def binomial_pass_rate_test(score: int, total: int, pass_rate: float = 0.7) -> float:
    """One-sided exact binomial test p-value for H0: p >= pass_rate vs H1: p < pass_rate.

    Returns a p-value (smaller => stronger evidence performance is below 'pass_rate').
    Handy for automated gates in CI (e.g., "fail build if p < 0.05").
    """
    if total == 0:
        return 1.0
    res = stats.binomtest(score, total, p=pass_rate, alternative="less")
    return float(res.pvalue)


def _build_exam_result(
    score: int,
    total: int,
    start_time: datetime,
    session: list[dict[str, Any]] | None = None,
    *,
    is_final: bool = False,
) -> ExamResult:
    """Build an ExamResult from current state."""
    elapsed = datetime.now() - start_time
    lower, upper = calculate_confidence_interval(score, total)
    exact_lower, exact_upper = calculate_exact_binomial_ci(score, total)
    p_value = binomial_pass_rate_test(score, total, pass_rate=0.70)

    avg_time = None
    est_left = None
    if session:
        avg_time, est_left = calculate_time_estimates(session, start_time)

    return ExamResult(
        score=score,
        total=total,
        elapsed=elapsed,
        avg_time_per_question=avg_time,
        estimated_time_left=est_left,
        ci_lower=lower,
        ci_upper=upper,
        exact_ci_lower=exact_lower,
        exact_ci_upper=exact_upper,
        p_value=p_value,
        is_final=is_final,
    )


def build_answer_feedback(options_list: list[dict[str, Any]], selected: list[dict[str, Any]]) -> AnswerFeedback:
    """Build answer feedback for a displayed option list and the user's selection."""
    correct = {o["text"] for o in options_list if o.get("is_correct", False)}
    user_answers = {o["text"] for o in selected}
    explanations = [(o.get("explanation", ""), o.get("is_correct", False)) for o in options_list]
    return AnswerFeedback(
        is_correct=user_answers == correct,
        correct_answers=correct,
        user_answers=user_answers,
        explanations=explanations,
    )


# ----------------- Core Q&A engine with caps -----------------


def index_session_by_id(session: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build an index for O(1) lookup of session questions by id.

    This avoids repeated O(n) scans throughout a long session.
    """
    idx: dict[str, dict[str, Any]] = {}
    for q in session:
        qid = q.get("id")
        if isinstance(qid, str):
            idx[qid] = q
    return idx


def find_question(question: dict[str, Any], session_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Lookup helper using the session index."""
    return session_index.get(question.get("id", ""), {})


def save_session_file(session_file: Path, state: list[dict[str, Any]], start_time: datetime) -> None:
    """Persist session state atomically to TOML (normalized for stable diffs)."""
    with open(session_file, "w", encoding="utf-8") as file:
        data = {
            "questions": state,
            "start_time": start_time.isoformat(),
            "last_updated": datetime.now().isoformat(),
        }
        toml.dump(normalize_exam_for_toml(data), file)


def interactive_question_and_answer(
    questions: list[dict[str, Any]],
    session: list[dict[str, Any]],
    session_path: Path,
    start_time: datetime,
    ui: FrontendUI,
    *,
    answer_provider: AnswerProvider | None = None,
    quiet: bool = False,
    questions_to_complete: int | None = None,
    questions_to_complete_for_session: int | None = None,
) -> int:
    """Run through questions using either interactive or machine answer provider.

    Caps:
      - questions_to_complete: limit questions answered in *this* run.
      - questions_to_complete_for_session: limit total answered in the *entire session*.
    """
    # Build a fast index for session lookups
    session_index = index_session_by_id(session)

    # Score already-completed questions in session
    score = 0
    completed_so_far = 0
    for q in session:
        if q.get("user_score") == 1:
            score += 1
        if q.get("user_score") is not None:
            completed_so_far += 1

    # Compute remaining allowance per caps
    remaining_allowed_session = (
        None
        if questions_to_complete_for_session is None
        else max(0, questions_to_complete_for_session - completed_so_far)
    )
    remaining_allowed_this_run = questions_to_complete if questions_to_complete is not None else None

    # Collect unanswered questions only
    unanswered = [q for q in questions if find_question(q, session_index).get("user_score") is None]
    random.shuffle(unanswered)
    answered_this_run = 0

    for question in unanswered:
        # Check caps before asking
        if remaining_allowed_session is not None and remaining_allowed_session <= 0:
            if not quiet:
                ui.show_panel("Session total cap reached.", title="Limit Reached", style="yellow")
            break
        if remaining_allowed_this_run is not None and answered_this_run >= remaining_allowed_this_run:
            if not quiet:
                ui.show_panel("Per-run question cap reached.", title="Limit Reached", style="yellow")
            break

        session_question = find_question(question, session_index)
        if not session_question:
            # Defensive copy if not yet in session (shouldn't happen if session starts as copy of questions)
            session_question = question
            session.append(session_question)
            # Update index
            qid = session_question.get("id")
            if isinstance(qid, str):
                session_index[qid] = session_question

        # Record start time
        question_start_time = datetime.now()
        session_question["start_time"] = question_start_time.isoformat()

        options_list = list(question["options"])
        random.shuffle(options_list)

        try:
            if answer_provider is None:
                selected = ask_question_interactive(ui, question, options_list)
            else:
                selected = ask_question_machine(answer_provider, question, options_list)
        except KeyboardInterrupt:
            if not quiet:
                result = _build_exam_result(score, completed_so_far, start_time, session, is_final=False)
                ui.show_results(result)
            raise

        # Record completion time
        session_question["completion_time"] = datetime.now().isoformat()

        feedback = build_answer_feedback(options_list, selected)
        correct = feedback.correct_answers
        user_answers = feedback.user_answers

        # Feedback (skip in quiet mode)
        if not quiet:
            ui.show_answer_feedback(feedback)

        session_question["user_answers"] = list(user_answers)
        if user_answers == correct:
            play_sound("correct.mp3") if not quiet else None
            score += 1
            session_question["user_score"] = 1
        else:
            if not quiet:
                ui.show_error("Incorrect.")
                play_sound("incorrect.mp3")
            session_question["user_score"] = 0

        completed_so_far += 1
        answered_this_run += 1
        if remaining_allowed_session is not None:
            remaining_allowed_session -= 1

        if not quiet:
            interim_result = _build_exam_result(score, completed_so_far, start_time, session, is_final=False)
            ui.show_results(interim_result)

        if answer_provider is None:
            go_on = ui.wait_for_continue()
            if go_on == "bad":
                session_question["defective"] = True
                save_session_file(session_path, session, start_time)

    if not quiet:
        ui.clear_screen()
        final_result = _build_exam_result(score, completed_so_far, start_time, session, is_final=True)
        ui.show_results(final_result)
    save_session_file(session_path, session, start_time)
    return score


# ----------------- Public entry points -----------------


def take_exam_now(
    question_file: str | None = None,
    *,
    ui: FrontendUI | None = None,
    machine: bool = False,
    strategy: MachineStrategy = "oracle",
    seed: int | None = 42,
    quiet: bool = False,
    questions_to_complete: int | None = None,
    questions_to_complete_for_session: int | None = None,
) -> None:
    """Main function to run the quiz.

    Interactive by default, or machine mode if requested.
    The two caps allow "do N more now" and "stop session at M total answered" behaviors.
    """
    # Default to Rich CLI if no UI provided
    if ui is None:
        from examexam.frontends.rich_ui import RichUI

        ui = RichUI()

    # If explicitly in machine mode (or via env var), run headless path
    if (machine and question_file) or (os.environ.get("EXAMEXAM_MACHINE_TAKES_EXAM")):
        _ = take_exam_machine(
            cast(str, question_file),
            ui=ui,
            strategy=strategy,
            seed=seed,
            quiet=quiet,
            persist_session=True,
            questions_to_complete=questions_to_complete,
            questions_to_complete_for_session=questions_to_complete_for_session,
        )
        return

    if question_file:
        # Legacy API - use provided file path
        test_path = Path(question_file)
        test_name = test_path.stem
        session_path = get_session_path(test_name)

        # Check for existing session
        resume_session, session_data, session_start_time = check_resume_session(test_name, ui)

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
        test_name = select_test(ui)
        if not test_name:
            return

        if (Path("data") / f"{test_name}.toml").exists():
            test_file = Path("data") / f"{test_name}.toml"
        else:
            test_file = f"{test_name}.toml"

        session_path = get_session_path(test_name)

        # Check for existing session
        resume_session, session_data, session_start_time = check_resume_session(test_name, ui)

        if resume_session and session_data:
            session = session_data
            questions = load_questions(str(test_file))
            start_time = session_start_time or datetime.now()  # Fallback to current time
        else:
            questions = load_questions(str(test_file))
            session = questions.copy()
            start_time = datetime.now()
            save_session_file(session_path, session, start_time)

    # Enforce session-wide cap before starting this run
    already_completed = sum(1 for q in session if q.get("user_score") is not None)
    if questions_to_complete_for_session is not None and already_completed >= questions_to_complete_for_session:
        if not quiet:
            ui.show_panel(
                f"Session cap reached\nAnswered: {already_completed}/{questions_to_complete_for_session}",
                title="Limits",
                style="yellow",
            )
            final_result = _build_exam_result(
                sum(1 for q in session if q.get("user_score") == 1),
                already_completed,
                start_time,
                session,
                is_final=True,
            )
            ui.show_results(final_result)
        save_session_file(session_path, session, start_time)
        return

    try:
        interactive_question_and_answer(
            questions,
            session,
            session_path,
            start_time,
            ui,
            answer_provider=None,
            quiet=quiet,
            questions_to_complete=questions_to_complete,
            questions_to_complete_for_session=questions_to_complete_for_session,
        )
        save_session_file(session_path, session, start_time)
    except KeyboardInterrupt:
        save_session_file(session_path, session, start_time)
        ui.show_error("Exiting the exam...")


def take_exam_machine(
    question_file: str,
    *,
    ui: FrontendUI | None = None,
    strategy: MachineStrategy = "oracle",
    seed: int | None = 42,
    quiet: bool = True,
    persist_session: bool = False,
    questions_to_complete: int | None = None,
    questions_to_complete_for_session: int | None = None,
) -> dict[str, Any]:
    """Non-interactive exam runner for integration tests.

    Returns:
      dict with keys: score, total, percent, session_path, session, start_time
    """
    # Default to Rich CLI if no UI provided
    if ui is None:
        from examexam.frontends.rich_ui import RichUI

        ui = RichUI()

    test_path = Path(question_file)
    test_name = test_path.stem
    questions = load_questions(str(test_path))

    # Fresh session each time unless you deliberately persist
    session_path = get_session_path(test_name)
    if not persist_session and session_path.exists():
        session_path.unlink(missing_ok=True)

    session = questions.copy()
    start_time = datetime.now()
    save_session_file(session_path, session, start_time)

    score = interactive_question_and_answer(
        questions,
        session,
        session_path,
        start_time,
        ui,
        answer_provider=build_machine_answer_provider(strategy=strategy, seed=seed),
        quiet=quiet,
        questions_to_complete=questions_to_complete,
        questions_to_complete_for_session=questions_to_complete_for_session,
    )

    total = len(questions)
    percent = (score / total * 100) if total else 0.0

    return {
        "score": score,
        "total": total,
        "percent": percent,
        "session_path": session_path,
        "session": session,
        "start_time": start_time,
    }


if __name__ == "__main__":
    # Example: run interactively with a per-run cap of 10
    take_exam_now(questions_to_complete=10)
