"""UI Protocol for examexam multi-frontend support.

All frontends (CLI, GUI, TUI, Web) implement this protocol.
Business logic calls these methods instead of using Rich directly.
"""

from __future__ import annotations

from typing import Any, Protocol

from examexam.models import AnswerFeedback, ExamResult, Option, Question, TestInfo


class FrontendUI(Protocol):
    """Protocol that all examexam frontends must implement.

    Methods are grouped by capability:
    - Core: messages, errors, confirmation
    - Exam: question display, answer collection, feedback
    - Progress: progress bar management
    - Display: panels, tables, markdown
    """

    # ---- Core ----

    def show_message(self, message: str, *, style: str = "") -> None:
        """Display a general message to the user."""
        ...

    def show_error(self, message: str) -> None:
        """Display an error message."""
        ...

    def confirm(self, message: str, *, default: bool = True) -> bool:
        """Ask the user a yes/no question. Returns their choice."""
        ...

    def get_input(self, prompt: str) -> str:
        """Get free-text input from the user."""
        ...

    # ---- Exam taking ----

    def show_test_selection(self, tests: list[TestInfo]) -> int | None:
        """Show available tests and let user pick one. Returns index or None."""
        ...

    def show_session_info(self, test_name: str, completed: int, total: int, time_ago: str) -> None:
        """Show information about an existing session."""
        ...

    def show_question(self, question: Question, options: list[Option], question_number: int | None = None) -> None:
        """Display a question and its shuffled options."""
        ...

    def get_answer(self, option_count: int, answer_count: int) -> str:
        """Get the user's answer (comma-separated option numbers)."""
        ...

    def show_answer_feedback(self, feedback: AnswerFeedback) -> None:
        """Show whether the answer was correct, with explanations."""
        ...

    def show_results(self, result: ExamResult) -> None:
        """Show exam results (interim or final)."""
        ...

    def wait_for_continue(self) -> str:
        """Wait for user to press Enter (or type 'bad'). Returns the input."""
        ...

    def clear_screen(self) -> None:
        """Clear the display."""
        ...

    # ---- Progress ----

    def progress_start(self, total: int, description: str = "") -> str:
        """Start a progress operation. Returns a task_id for updates."""
        ...

    def progress_update(self, task_id: str, advance: int = 1, description: str = "") -> None:
        """Update a progress operation."""
        ...

    def progress_finish(self, task_id: str) -> None:
        """Mark a progress operation as complete."""
        ...

    # ---- Display ----

    def show_panel(self, content: str, *, title: str = "", style: str = "") -> None:
        """Show content in a bordered panel."""
        ...

    def show_markdown(self, content: str) -> None:
        """Render and display markdown content."""
        ...

    def show_rule(self, title: str = "") -> None:
        """Show a horizontal rule/separator."""
        ...

    # ---- Lifecycle ----

    def run(self, callback: Any = None) -> None:
        """Start the frontend event loop (for GUI frontends). No-op for CLI."""
        ...

    def shutdown(self) -> None:
        """Clean up resources."""
        ...
