"""Tkinter GUI frontend for examexam.

Implements the FrontendUI protocol using tkinter for a native desktop GUI.
Runs the exam-taking flow in a separate thread so the tkinter event loop stays responsive.
"""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext, simpledialog, ttk
from typing import Any

from examexam.constants import BAD_QUESTION_TEXT
from examexam.models import AnswerFeedback, ExamResult, Option, Question, TestInfo


class TkinterUI:
    """Tkinter-based GUI frontend implementing the FrontendUI protocol."""

    def __init__(self) -> None:
        self._root: tk.Tk | None = None
        self._answer_var: tk.StringVar | None = None
        self._answer_event = threading.Event()
        self._confirm_result: bool = True
        self._confirm_event = threading.Event()
        self._input_result: str = ""
        self._input_event = threading.Event()
        self._continue_event = threading.Event()
        self._continue_result: str = ""
        self._selection_result: int | None = None
        self._selection_event = threading.Event()
        self._main_frame: ttk.Frame | None = None
        self._log_text: scrolledtext.ScrolledText | None = None
        self._initialized = False

    def _ensure_root(self) -> tk.Tk:
        if self._root is None:
            self._root = tk.Tk()
            self._root.title("ExamExam")
            self._root.geometry("900x700")
            self._root.configure(bg="#f0f0f0")

            # Style
            style = ttk.Style()
            style.theme_use("clam")
            style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"), background="#f0f0f0")
            style.configure("Question.TLabel", font=("Segoe UI", 12), background="#f0f0f0", wraplength=800)
            style.configure("Correct.TLabel", foreground="green", font=("Segoe UI", 11))
            style.configure("Incorrect.TLabel", foreground="red", font=("Segoe UI", 11))
            style.configure("Score.TLabel", font=("Segoe UI", 14, "bold"), background="#f0f0f0")

            self._main_frame = ttk.Frame(self._root, padding=20)
            self._main_frame.pack(fill=tk.BOTH, expand=True)

            self._initialized = True
        return self._root

    def _clear_main_frame(self) -> None:
        if self._main_frame:
            for widget in self._main_frame.winfo_children():
                widget.destroy()

    def _schedule(self, func, *args):
        """Schedule a function to run on the main tkinter thread."""
        root = self._ensure_root()
        root.after(0, func, *args)

    # ---- Core ----

    def show_message(self, message: str, *, style: str = "") -> None:
        def _show():
            if self._log_text:
                self._log_text.insert(tk.END, message + "\n")
                self._log_text.see(tk.END)
        if self._root and self._initialized:
            self._schedule(_show)
        # else: silently ignore pre-init messages

    def show_error(self, message: str) -> None:
        def _show():
            if self._root:
                messagebox.showerror("Error", message, parent=self._root)
        if self._root and self._initialized:
            self._schedule(_show)

    def confirm(self, message: str, *, default: bool = True) -> bool:
        self._confirm_event.clear()

        def _ask():
            self._confirm_result = messagebox.askyesno("Confirm", message, parent=self._root)
            self._confirm_event.set()

        self._schedule(_ask)
        self._confirm_event.wait()
        return self._confirm_result

    def get_input(self, prompt: str) -> str:
        self._input_event.clear()

        def _ask():
            result = simpledialog.askstring("Input", prompt, parent=self._root)
            self._input_result = result or ""
            self._input_event.set()

        self._schedule(_ask)
        self._input_event.wait()
        return self._input_result

    # ---- Exam taking ----

    def show_test_selection(self, tests: list[TestInfo]) -> int | None:
        if not tests:
            self.show_error("No test files found!")
            return None

        self._selection_event.clear()

        def _show():
            self._clear_main_frame()
            frame = self._main_frame

            ttk.Label(frame, text="Select a Test", style="Title.TLabel").pack(pady=(0, 15))

            listbox = tk.Listbox(frame, font=("Segoe UI", 12), height=min(len(tests), 15), width=60)
            for test in tests:
                listbox.insert(tk.END, f"  {test.index}. {test.name}")
            listbox.pack(pady=10)
            listbox.selection_set(0)

            def on_select():
                sel = listbox.curselection()
                if sel:
                    self._selection_result = sel[0]
                else:
                    self._selection_result = 0
                self._selection_event.set()

            ttk.Button(frame, text="Start Exam", command=on_select).pack(pady=10)

        self._schedule(_show)
        self._selection_event.wait()
        return self._selection_result

    def show_session_info(self, test_name: str, completed: int, total: int, time_ago: str) -> None:
        # This will be followed by a confirm() call, so just show info
        self.show_message(f"Session found for '{test_name}': {completed}/{total} completed ({time_ago} ago)")

    def show_question(self, question: Question, options: list[Option], question_number: int | None = None) -> None:
        self._answer_event.clear()

        def _show():
            self._clear_main_frame()
            frame = self._main_frame

            # Question text
            q_text = question.question
            answer_count = question.answer_count
            if "(Select" not in q_text:
                q_text = f"{q_text} (Select {answer_count})"

            ttk.Label(frame, text=q_text, style="Question.TLabel").pack(pady=(0, 15), anchor="w")

            ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=5)

            # Options as checkbuttons
            self._option_vars: list[tk.BooleanVar] = []
            for idx, opt in enumerate(options, 1):
                var = tk.BooleanVar(value=False)
                self._option_vars.append(var)
                cb = ttk.Checkbutton(frame, text=f"{idx}. {opt.text}", variable=var)
                cb.pack(anchor="w", pady=2, padx=20)

            # Bad question option
            bad_var = tk.BooleanVar(value=False)
            self._option_vars.append(bad_var)
            cb = ttk.Checkbutton(frame, text=f"{len(options) + 1}. {BAD_QUESTION_TEXT}", variable=bad_var)
            cb.pack(anchor="w", pady=2, padx=20)

            ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=10)

            def on_submit():
                selected = [str(i + 1) for i, var in enumerate(self._option_vars) if var.get()]
                self._input_result = ",".join(selected) if selected else ""
                self._answer_event.set()

            ttk.Button(frame, text="Submit Answer", command=on_submit).pack(pady=10)

        self._schedule(_show)

    def get_answer(self, option_count: int, answer_count: int) -> str:
        # Wait for the submit button from show_question
        while True:
            self._answer_event.wait()
            self._answer_event.clear()
            answer = self._input_result

            from examexam.take_exam import is_valid
            valid, error_msg = is_valid(answer, option_count, answer_count)
            if valid:
                return answer

            # Show error and re-enable
            def _show_err(msg=error_msg):
                messagebox.showwarning("Invalid Answer", msg, parent=self._root)
                # Re-arm the submit (user can click again)
            self._schedule(_show_err)
            # Loop back and wait for another submit

    def show_answer_feedback(self, feedback: AnswerFeedback) -> None:
        def _show():
            self._clear_main_frame()
            frame = self._main_frame

            if feedback.is_correct:
                ttk.Label(frame, text="\u2713 Correct!", style="Correct.TLabel",
                          font=("Segoe UI", 16, "bold")).pack(pady=(0, 10))
            else:
                ttk.Label(frame, text="\u2717 Incorrect", style="Incorrect.TLabel",
                          font=("Segoe UI", 16, "bold")).pack(pady=(0, 10))
                ttk.Label(frame, text=f"Correct: {', '.join(feedback.correct_answers)}",
                          wraplength=800).pack(anchor="w", padx=20)
                ttk.Label(frame, text=f"Your answer: {', '.join(feedback.user_answers)}",
                          wraplength=800).pack(anchor="w", padx=20)

            ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=10)

            # Explanations
            ttk.Label(frame, text="Explanations:", font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=10, pady=(5, 5))
            for idx, (explanation, is_correct) in enumerate(feedback.explanations, 1):
                color = "green" if is_correct else "red"
                lbl = ttk.Label(frame, text=f"{idx}. {explanation}", wraplength=780,
                                foreground=color, font=("Segoe UI", 10))
                lbl.pack(anchor="w", padx=30, pady=2)

        self._schedule(_show)

    def show_results(self, result: ExamResult) -> None:
        from examexam.take_exam import humanize_timedelta

        def _show():
            self._clear_main_frame()
            frame = self._main_frame

            percent = result.percent
            title = "Final Results" if result.is_final else "Current Progress"
            ttk.Label(frame, text=title, style="Title.TLabel").pack(pady=(0, 10))

            score_text = f"Score: {result.score}/{result.total} ({percent:.1f}%)"
            ttk.Label(frame, text=score_text, style="Score.TLabel").pack(pady=5)

            if result.is_final:
                status = "PASSED" if result.passed else "FAILED"
                color = "green" if result.passed else "red"
                ttk.Label(frame, text=status, foreground=color,
                          font=("Segoe UI", 18, "bold")).pack(pady=5)

            # Progress bar
            progress_bar = ttk.Progressbar(frame, length=400, mode="determinate",
                                           maximum=result.total, value=result.score)
            progress_bar.pack(pady=10)

            # Time info
            time_str = f"Time elapsed: {humanize_timedelta(result.elapsed)}"
            ttk.Label(frame, text=time_str).pack(anchor="w", padx=20)

            if result.avg_time_per_question:
                ttk.Label(frame, text=f"Avg per question: {humanize_timedelta(result.avg_time_per_question)}").pack(
                    anchor="w", padx=20)

            if result.estimated_time_left and not result.is_final:
                ttk.Label(frame, text=f"Est. remaining: {humanize_timedelta(result.estimated_time_left)}").pack(
                    anchor="w", padx=20)

            # CI info
            ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=10)
            ci_text = f"95% CI: {result.ci_lower * 100:.1f}%-{result.ci_upper * 100:.1f}% (normal) | {result.exact_ci_lower * 100:.1f}%-{result.exact_ci_upper * 100:.1f}% (exact)"
            ttk.Label(frame, text=ci_text, font=("Segoe UI", 9)).pack(anchor="w", padx=20)
            ttk.Label(frame, text=f"Binomial test vs 70%: p={result.p_value:.3f}",
                      font=("Segoe UI", 9)).pack(anchor="w", padx=20)

        self._schedule(_show)

    def wait_for_continue(self) -> str:
        self._continue_event.clear()

        def _show():
            if self._main_frame:
                btn_frame = ttk.Frame(self._main_frame)
                btn_frame.pack(pady=15)

                def on_continue():
                    self._continue_result = ""
                    self._continue_event.set()

                def on_bad():
                    self._continue_result = "bad"
                    self._continue_event.set()

                ttk.Button(btn_frame, text="Next Question", command=on_continue).pack(side=tk.LEFT, padx=5)
                ttk.Button(btn_frame, text="Mark as Bad Question", command=on_bad).pack(side=tk.LEFT, padx=5)

        self._schedule(_show)
        self._continue_event.wait()
        return self._continue_result

    def clear_screen(self) -> None:
        if self._root and self._initialized:
            self._schedule(self._clear_main_frame)

    # ---- Progress ----

    def progress_start(self, total: int, description: str = "") -> str:
        import uuid
        task_id = str(uuid.uuid4())[:8]
        self.show_message(f"{description} (0/{total})" if description else f"Progress: 0/{total}")
        return task_id

    def progress_update(self, task_id: str, advance: int = 1, description: str = "") -> None:
        if description:
            self.show_message(description)

    def progress_finish(self, task_id: str) -> None:
        pass

    # ---- Display ----

    def show_panel(self, content: str, *, title: str = "", style: str = "") -> None:
        def _show():
            if self._root:
                messagebox.showinfo(title or "ExamExam", content, parent=self._root)

        if self._root and self._initialized:
            self._schedule(_show)

    def show_markdown(self, content: str) -> None:
        def _show():
            self._clear_main_frame()
            frame = self._main_frame
            text_widget = scrolledtext.ScrolledText(frame, wrap=tk.WORD, font=("Segoe UI", 11),
                                                     width=90, height=30)
            text_widget.insert(tk.END, content)
            text_widget.configure(state="disabled")
            text_widget.pack(fill=tk.BOTH, expand=True)

        self._schedule(_show)

    def show_rule(self, title: str = "") -> None:
        self.show_message(f"{'=' * 40} {title} {'=' * 40}" if title else "=" * 80)

    # ---- Lifecycle ----

    def run(self, callback: Any = None) -> None:
        """Start the tkinter event loop.

        If callback is provided, it runs in a background thread
        so the GUI stays responsive. When the callback completes,
        a "Done" button appears so the user can review results
        before the window closes.
        """
        root = self._ensure_root()

        def _on_close():
            """Handle window close button — unblock any waiting threads."""
            # Unblock all waiting events so worker thread can exit
            self._answer_event.set()
            self._confirm_event.set()
            self._input_event.set()
            self._continue_event.set()
            self._selection_event.set()
            if self._root:
                self._root.quit()

        root.protocol("WM_DELETE_WINDOW", _on_close)

        if callback:
            def _run_callback():
                try:
                    callback()
                except KeyboardInterrupt:
                    pass
                except Exception as e:
                    _ = e # something keeps removing this, ruff?
                    if self._root:
                        self._schedule(
                            lambda: messagebox.showerror("Error", str(e), parent=self._root) # noqa
                        )
                finally:
                    # When done, add a "Close" button so user can review final results
                    def _add_close_button():
                        if self._main_frame and self._root:
                            ttk.Button(
                                self._main_frame, text="Close", command=self._root.quit
                            ).pack(pady=15)
                    if self._root:
                        self._schedule(_add_close_button)

            thread = threading.Thread(target=_run_callback, daemon=True)
            thread.start()

        root.mainloop()

    def shutdown(self) -> None:
        if self._root:
            self._root.quit()
            self._root.destroy()
            self._root = None
