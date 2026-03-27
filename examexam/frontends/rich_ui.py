"""Rich CLI frontend for examexam.

This implements the FrontendUI protocol using the Rich library,
preserving the existing terminal-based user experience.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

from rich.align import Align
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from examexam.constants import BAD_QUESTION_TEXT
from examexam.models import AnswerFeedback, ExamResult, Option, Question, TestInfo


class RichUI:
    """Rich-based CLI frontend implementing the FrontendUI protocol."""

    def __init__(self) -> None:
        self.console = Console()

    # ---- Core ----

    def show_message(self, message: str, *, style: str = "") -> None:
        if style:
            self.console.print(f"[{style}]{message}[/{style}]")
        else:
            self.console.print(message)

    def show_error(self, message: str) -> None:
        self.console.print(f"[bold red]{message}[/bold red]")

    def confirm(self, message: str, *, default: bool = True) -> bool:
        return Confirm.ask(message, default=default)

    def get_input(self, prompt: str) -> str:
        return Prompt.ask(prompt)

    # ---- Exam taking ----

    def show_test_selection(self, tests: list[TestInfo]) -> int | None:
        if not tests:
            self.console.print("[bold red]No test files found in /data/ folder![/bold red]")
            return None

        self.console.print("[bold blue]Available Tests:[/bold blue]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Number", style="dim", width=6)
        table.add_column("Test Name")

        for test in tests:
            table.add_row(str(test.index), test.name)

        self.console.print(table)

        while True:
            try:
                choice = Prompt.ask("Enter the test number", default="1")
                test_idx = int(choice) - 1
                if 0 <= test_idx < len(tests):
                    return test_idx
                self.console.print("[bold red]Invalid choice. Please try again.[/bold red]")
            except ValueError:
                self.console.print("[bold red]Please enter a valid number.[/bold red]")

    def show_session_info(self, test_name: str, completed: int, total: int, time_ago: str) -> None:
        self.console.print(f"[bold yellow]Found existing session for '{test_name}'[/bold yellow]")
        self.console.print(f"Progress: {completed}/{total} questions completed")
        if time_ago:
            self.console.print(f"Started: {time_ago} ago")

    def show_question(self, question: Question, options: list[Option], question_number: int | None = None) -> None:
        self.clear_screen()
        question_text = question.question
        answer_count = question.answer_count

        # Fix "(Select n)" patterns
        import re

        pattern_match = re.search(r"\(Select [1-5]\)", question_text)
        if pattern_match:
            correct_select = f"(Select {answer_count})"
            if correct_select not in question_text:
                question_text = question_text.replace(pattern_match.group(0), correct_select)

        if "(Select" not in question_text:
            question_text = f"{question_text} (Select {answer_count})"

        if "(Select n)" in question_text:
            question_text = question_text.replace("(Select n)", f"(Select {answer_count})")

        question_panel = Align.center(Panel(Markdown(question_text)), vertical="middle")
        self.console.print(question_panel)

        table = Table(title="Options", style="green")
        table.add_column("Option Number", justify="center")
        table.add_column("Option Text", justify="left")

        for idx, option in enumerate(options, 1):
            table.add_row(str(idx), option.text)

        table.add_row(str(len(options) + 1), BAD_QUESTION_TEXT)
        self.console.print(Align.center(table))

    def get_answer(self, option_count: int, answer_count: int) -> str:
        while True:
            answer = self.console.input(
                "[bold yellow]Enter your answer(s) as a comma-separated list (e.g., 1,2): [/bold yellow]"
            )
            from examexam.take_exam import is_valid

            valid, error_msg = is_valid(answer, option_count, answer_count)
            if valid:
                return answer
            self.console.print(f"[bold red]{error_msg}[/bold red]")

    def show_answer_feedback(self, feedback: AnswerFeedback) -> None:
        if feedback.is_correct:
            self.console.print(Panel("[bold green]\u2713 Correct![/bold green]", title="Answer Review", style="green"))
        else:
            self.console.print(
                Panel(
                    f"[bold cyan]Correct Answer(s): {', '.join(feedback.correct_answers)}\nYour Answer(s): {', '.join(feedback.user_answers)}[/bold cyan]",
                    title="Answer Review",
                    style="blue",
                )
            )

        colored_explanations: list[str] = []
        for idx, (explanation, is_correct) in enumerate(feedback.explanations, 1):
            if is_correct:
                colored_explanations.append(f"{idx}. [bold green]{explanation}[/bold green]")
            else:
                colored_explanations.append(f"{idx}. [bold red]{explanation}[/bold red]")
        self.console.print(Panel("\n".join(colored_explanations), title="Explanation"))

    def show_results(self, result: ExamResult) -> None:
        from examexam.take_exam import humanize_timedelta

        percent = result.percent
        total_time_str = humanize_timedelta(result.elapsed)

        time_info = f"Total Time: {total_time_str}"
        if result.avg_time_per_question is not None:
            time_info += f"\nAvg Time per Question: {humanize_timedelta(result.avg_time_per_question)}"
        if result.estimated_time_left is not None and not result.is_final:
            time_info += f"\nEstimated Time to Complete: {humanize_timedelta(result.estimated_time_left)}"

        confidence_str = f"Normal 95% CI: {result.ci_lower * 100:.1f}%-{result.ci_upper * 100:.1f}% | Exact 95% CI: {result.exact_ci_lower * 100:.1f}%-{result.exact_ci_upper * 100:.1f}%"
        pvalue_str = f"Binomial test vs 70% pass rate (one-sided): p={result.p_value:.3f}"

        if result.is_final:
            passed = "Passed" if result.passed else "Failed"
            judgement = f"\n[green]{passed}[/green]"
        else:
            judgement = ""

        result_text = f"[bold yellow]Your Score: {result.score}/{result.total} ({percent:.2f}%){judgement}\n{time_info}\n{confidence_str}\n{pvalue_str}[/bold yellow]"

        self.console.print(
            Panel(
                result_text,
                title="Results",
                style="magenta",
            )
        )

    def wait_for_continue(self) -> str:
        go_on: str | None = None
        while go_on not in ("", "bad"):
            go_on = self.console.input("[bold yellow]Press Enter to continue to the next question...[/bold yellow]")
        return go_on or ""

    def clear_screen(self) -> None:
        try:
            self.console.clear()
        except Exception:
            os.system("cls" if os.name == "nt" else "clear")  # nosec

    # ---- Progress ----

    def progress_start(self, total: int, description: str = "") -> str:
        task_id = str(uuid.uuid4())[:8]
        # For the Rich CLI, progress is handled inline via show_message
        # The Rich progress bars in generate/validate are kept in those modules
        # since they need the Rich Progress context manager pattern.
        # This simpler interface is for general progress reporting.
        if description:
            self.console.print(f"{description} (0/{total})")
        return task_id

    def progress_update(self, task_id: str, advance: int = 1, description: str = "") -> None:
        if description:
            self.console.print(description)

    def progress_finish(self, task_id: str) -> None:
        pass

    # ---- Display ----

    def show_panel(self, content: str, *, title: str = "", style: str = "") -> None:
        kwargs: dict[str, Any] = {}
        if title:
            kwargs["title"] = title
        if style:
            kwargs["style"] = style
        self.console.print(Panel(content, **kwargs))

    def show_markdown(self, content: str) -> None:
        self.console.print(Markdown(content))

    def show_rule(self, title: str = "") -> None:
        if title:
            self.console.rule(f"[bold]{title}[/bold]")
        else:
            self.console.rule()

    # ---- Lifecycle ----

    def run(self, callback: Any = None) -> None:
        # CLI is synchronous, no event loop needed
        pass

    def shutdown(self) -> None:
        pass
