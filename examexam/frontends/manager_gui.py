"""Tkinter management GUI for examexam.

Three-panel layout:
  Left   – command buttons (Generate, Validate, Research, Study Plan, Convert,
            Config, Doctor, API Keys, Take Exam)
  Middle – content area: scrolled output + input widgets for the selected command
  Right  – help / cheat-sheet panel

All long-running operations execute in a background daemon thread so the
tkinter event loop stays responsive.
"""

from __future__ import annotations

import os
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Any

# ── Catppuccin Mocha-inspired palette ──────────────────────────────────────
_CLR_BG = "#1e1e2e"
_CLR_BG_ALT = "#252536"
_CLR_FG = "#cdd6f4"
_CLR_ACCENT = "#89b4fa"
_CLR_SIDEBAR = "#181825"
_CLR_BTN = "#313244"
_CLR_BTN_ACTIVE = "#45475a"
_CLR_OK = "#22c55e"
_CLR_WARN = "#eab308"
_CLR_ERR = "#ef4444"
_CLR_DIM = "#9ca3af"
_CLR_BORDER = "#45475a"

_FONT_UI = ("Segoe UI", 10)
_FONT_MONO = ("Consolas", 10)
_FONT_HEADING = ("Segoe UI", 11, "bold")
_FONT_BTN = ("Segoe UI", 10)

# ── Help texts per panel ───────────────────────────────────────────────────
_HELP = {
    "home": """\
ExamExam Manager

Use the command buttons on the left
to manage your exam question sets.

Quick-start
-----------
1. Write a toc.txt with one topic
   per line.
2. Click Generate to create questions.
3. Click Validate to sanity-check them.
4. Click Take Exam to start a quiz.

Keyboard shortcuts
------------------
Ctrl+Q  Quit
""",
    "generate": """\
Generate Questions

Creates TOML question files from a
table-of-contents text file.

Input file  – plain text, one topic
              per line.
Output file – saved as .toml

--n  Questions per topic (default 5).

Tip: keep TOC topics short and specific
for best LLM output.
""",
    "validate": """\
Validate Questions

Sends every question in a TOML file
to the LLM and checks:
  - factual accuracy
  - correct answer labels
  - explanation quality

Results are written back to the file
with a 'validated' flag.
""",
    "research": """\
Research / Study Guide

Generates a detailed study guide for
a single topic using the LLM.

Output is printed to the content area
and can be copied from there.
""",
    "study_plan": """\
Study Plan

Reads a toc.txt file and produces a
week-by-week study plan covering all
topics.

Useful for scheduling exam prep over
several weeks.
""",
    "convert": """\
Convert Questions

Converts a TOML question file to:
  - Markdown (.md)
  - HTML (.html)

Useful for sharing or printing a
human-readable version of the exam.
""",
    "config": """\
Configuration

Shows the current examexam.toml
settings and lets you edit them.

Config precedence:
1. Environment variables
   (EXAMEXAM_GENERAL_DEFAULT_N)
2. examexam.toml
3. Built-in defaults

Click "Create Default Config" to
generate a starter examexam.toml.
""",
    "api_keys": """\
API Keys

Displays API key status from your
environment and .env file.

Keys are masked – only the first and
last 4 characters are shown.

To set a key, either:
  • Add it to a .env file in the
    current directory, or
  • Set an environment variable before
    launching examexam.

Supported providers:
  OPENAI_API_KEY
  ANTHROPIC_API_KEY
  GOOGLE_API_KEY
  MISTRAL_API_KEY
  AWS_ACCESS_KEY_ID
""",
    "doctor": """\
Doctor

Prints a full diagnostic report:
  - Python version & platform
  - Installed package versions
  - API key presence (masked)
  - Config file locations
  - Available TOML question files

Share this report when asking for
tech support.
""",
    "take_exam": """\
Take Exam

Launches the interactive exam-taking
GUI in a new window.

Select a TOML question file, then
answer questions one by one.

Your progress is saved after each
question so you can resume later.
""",
}

_CHEATSHEET = """\
── Cheat Sheet ─────────────────

examexam generate
  --toc-file topics.txt
  --output-file q.toml
  -n 5

examexam validate
  --question-file q.toml

examexam research
  --topic "AWS S3"

examexam study-plan
  --toc-file topics.txt

examexam convert
  --input-file q.toml
  --output-base-name my-exam

examexam take
  --question-file q.toml

examexam doctor

─── Model flags ────────────────
--model-provider  openai|anthropic
                  |google|mistral
--model-class     fast|frontier

─── Global flags ───────────────
--frontend  cli|gui|tui|web
--verbose
"""


# ── Background runner ──────────────────────────────────────────────────────


class _BackgroundRunner:
    """Runs a callable in a daemon thread and posts results back via root.after."""

    def __init__(self, root: tk.Tk) -> None:
        self._root = root

    def run(
        self,
        func,
        *,
        args: tuple = (),
        on_success=None,
        on_error=None,
    ) -> None:
        def _worker():
            try:
                result = func(*args)
            except Exception as exc:
                if on_error:
                    self._root.after(0, on_error, exc)
                return
            if on_success:
                self._root.after(0, on_success, result)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()


# ── Output queue helper ────────────────────────────────────────────────────


class _StreamToQueue:
    """File-like object that puts lines into a queue for the GUI to poll."""

    def __init__(self, q: queue.Queue[str]) -> None:
        self._q = q

    def write(self, text: str) -> None:
        if text:
            self._q.put(text)

    def flush(self) -> None:
        pass


# ── Widget helpers ─────────────────────────────────────────────────────────


def _make_output(parent: tk.Widget, height: int = 20) -> scrolledtext.ScrolledText:
    w = scrolledtext.ScrolledText(
        parent,
        height=height,
        font=_FONT_MONO,
        bg=_CLR_BG_ALT,
        fg=_CLR_FG,
        insertbackground=_CLR_FG,
        relief=tk.FLAT,
        wrap=tk.WORD,
    )
    return w


def _output_append(widget: scrolledtext.ScrolledText, text: str) -> None:
    widget.configure(state="normal")
    widget.insert(tk.END, text)
    widget.see(tk.END)
    widget.configure(state="disabled")


def _output_clear(widget: scrolledtext.ScrolledText) -> None:
    widget.configure(state="normal")
    widget.delete("1.0", tk.END)
    widget.configure(state="disabled")


def _make_label(parent, text: str, **kw) -> tk.Label:
    return tk.Label(parent, text=text, bg=_CLR_BG, fg=_CLR_FG, font=_FONT_UI, anchor="w", **kw)


def _make_entry(parent, textvariable=None, width=40) -> ttk.Entry:
    e = ttk.Entry(parent, textvariable=textvariable, width=width, font=_FONT_MONO)
    return e


def _make_btn(parent, text: str, command, **kw) -> tk.Button:
    defaults = dict(
        bg=_CLR_BTN,
        fg=_CLR_FG,
        activebackground=_CLR_BTN_ACTIVE,
        activeforeground=_CLR_FG,
        font=_FONT_BTN,
        relief=tk.FLAT,
        padx=8,
        pady=4,
        cursor="hand2",
    )
    defaults.update(kw)
    return tk.Button(parent, text=text, command=command, **defaults)


def _pick_file(title="Select file", filetypes=None) -> str:
    filetypes = filetypes or [("All files", "*.*")]
    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    return path or ""


def _pick_save(title="Save as", defaultextension="", filetypes=None) -> str:
    filetypes = filetypes or [("All files", "*.*")]
    path = filedialog.asksaveasfilename(title=title, defaultextension=defaultextension, filetypes=filetypes)
    return path or ""


# ── Base panel ─────────────────────────────────────────────────────────────


class _BasePanel(tk.Frame):
    def __init__(
        self, parent: tk.Widget, runner: _BackgroundRunner, status_var: tk.StringVar, help_key: str = "home"
    ) -> None:
        super().__init__(parent, bg=_CLR_BG)
        self._runner = runner
        self._status = status_var
        self._help_key = help_key

    @property
    def help_text(self) -> str:
        return _HELP.get(self._help_key, "")


# ── Home panel ─────────────────────────────────────────────────────────────


class HomePanel(_BasePanel):
    def __init__(self, parent, runner, status_var):
        super().__init__(parent, runner, status_var, "home")
        tk.Label(
            self,
            text="ExamExam Manager",
            bg=_CLR_BG,
            fg=_CLR_ACCENT,
            font=("Segoe UI", 18, "bold"),
        ).pack(pady=(40, 10))
        tk.Label(
            self,
            text="Select a command from the left panel to get started.",
            bg=_CLR_BG,
            fg=_CLR_DIM,
            font=_FONT_UI,
        ).pack(pady=5)


# ── Generate panel ─────────────────────────────────────────────────────────


class GeneratePanel(_BasePanel):
    def __init__(self, parent, runner, status_var):
        super().__init__(parent, runner, status_var, "generate")
        self._build()

    def _build(self):
        tk.Label(self, text="Generate Questions", bg=_CLR_BG, fg=_CLR_ACCENT, font=_FONT_HEADING).pack(
            anchor="w", padx=12, pady=(12, 4)
        )

        form = tk.Frame(self, bg=_CLR_BG)
        form.pack(fill=tk.X, padx=12, pady=4)

        # TOC file
        _make_label(form, "TOC file:").grid(row=0, column=0, sticky="w", pady=3)
        self._toc_var = tk.StringVar()
        _make_entry(form, self._toc_var, width=38).grid(row=0, column=1, sticky="ew", padx=4)
        _make_btn(form, "Browse", self._browse_toc).grid(row=0, column=2, padx=2)

        # Output file
        _make_label(form, "Output file:").grid(row=1, column=0, sticky="w", pady=3)
        self._out_var = tk.StringVar()
        _make_entry(form, self._out_var, width=38).grid(row=1, column=1, sticky="ew", padx=4)
        _make_btn(form, "Browse", self._browse_out).grid(row=1, column=2, padx=2)

        # N questions
        _make_label(form, "Questions per topic:").grid(row=2, column=0, sticky="w", pady=3)
        self._n_var = tk.StringVar(value="5")
        _make_entry(form, self._n_var, width=10).grid(row=2, column=1, sticky="w", padx=4)

        # Provider
        _make_label(form, "Model provider:").grid(row=3, column=0, sticky="w", pady=3)
        self._provider_var = tk.StringVar(value="openai")
        ttk.Combobox(
            form,
            textvariable=self._provider_var,
            values=["openai", "anthropic", "google", "mistral"],
            width=15,
            state="readonly",
        ).grid(row=3, column=1, sticky="w", padx=4)

        form.columnconfigure(1, weight=1)

        btn_row = tk.Frame(self, bg=_CLR_BG)
        btn_row.pack(anchor="w", padx=12, pady=8)
        _make_btn(btn_row, "Generate", self._run, bg=_CLR_ACCENT, fg=_CLR_BG).pack(side=tk.LEFT, padx=4)

        self._out_widget = _make_output(self)
        self._out_widget.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)

    def _browse_toc(self):
        path = _pick_file("Select TOC file", [("Text files", "*.txt"), ("All", "*.*")])
        if path:
            self._toc_var.set(path)
            if not self._out_var.get():
                self._out_var.set(str(Path(path).with_suffix(".toml")))

    def _browse_out(self):
        path = _pick_save("Save questions as", ".toml", [("TOML files", "*.toml")])
        if path:
            self._out_var.set(path)

    def _run(self):
        toc = self._toc_var.get().strip()
        out = self._out_var.get().strip()
        n_str = self._n_var.get().strip()
        provider = self._provider_var.get()

        if not toc:
            messagebox.showwarning("Missing input", "Please select a TOC file.")
            return
        try:
            n = int(n_str)
        except ValueError:
            messagebox.showwarning("Bad value", "Questions per topic must be an integer.")
            return

        if not out:
            out = str(Path(toc).with_suffix(".toml"))
            self._out_var.set(out)

        _output_clear(self._out_widget)
        self._status.set("Generating questions...")

        def _fetch():
            from examexam.apis.conversation_and_router import pick_model
            from examexam.frontends.manager_gui import _QueueUI
            from examexam.generate_questions import generate_questions_now

            q: queue.Queue[str] = queue.Queue()
            ui = _QueueUI(q)
            generate_questions_now(
                questions_per_toc_topic=n,
                file_name=out,
                toc_file=toc,
                model=pick_model("", provider, "fast"),
                system_prompt="You are a test maker.",
                ui=ui,
            )
            return q

        def _success(q):
            while not q.empty():
                _output_append(self._out_widget, q.get())
            _output_append(self._out_widget, f"\nDone. Saved to: {out}\n")
            self._status.set("Generate complete.")

        def _error(exc):
            _output_append(self._out_widget, f"\nError: {exc}\n")
            self._status.set("Error during generate.")

        self._runner.run(_fetch, on_success=_success, on_error=_error)


# ── Validate panel ─────────────────────────────────────────────────────────


class ValidatePanel(_BasePanel):
    def __init__(self, parent, runner, status_var):
        super().__init__(parent, runner, status_var, "validate")
        self._build()

    def _build(self):
        tk.Label(self, text="Validate Questions", bg=_CLR_BG, fg=_CLR_ACCENT, font=_FONT_HEADING).pack(
            anchor="w", padx=12, pady=(12, 4)
        )
        form = tk.Frame(self, bg=_CLR_BG)
        form.pack(fill=tk.X, padx=12, pady=4)

        _make_label(form, "Question file:").grid(row=0, column=0, sticky="w", pady=3)
        self._file_var = tk.StringVar()
        _make_entry(form, self._file_var, width=40).grid(row=0, column=1, sticky="ew", padx=4)
        _make_btn(form, "Browse", self._browse).grid(row=0, column=2, padx=2)

        _make_label(form, "Model provider:").grid(row=1, column=0, sticky="w", pady=3)
        self._provider_var = tk.StringVar(value="openai")
        ttk.Combobox(
            form,
            textvariable=self._provider_var,
            values=["openai", "anthropic", "google", "mistral"],
            width=15,
            state="readonly",
        ).grid(row=1, column=1, sticky="w", padx=4)

        form.columnconfigure(1, weight=1)

        btn_row = tk.Frame(self, bg=_CLR_BG)
        btn_row.pack(anchor="w", padx=12, pady=8)
        _make_btn(btn_row, "Validate", self._run, bg=_CLR_ACCENT, fg=_CLR_BG).pack(side=tk.LEFT, padx=4)

        self._out_widget = _make_output(self)
        self._out_widget.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)

    def _browse(self):
        path = _pick_file("Select question file", [("TOML files", "*.toml"), ("All", "*.*")])
        if path:
            self._file_var.set(path)

    def _run(self):
        qfile = self._file_var.get().strip()
        if not qfile:
            messagebox.showwarning("Missing input", "Please select a question file.")
            return

        provider = self._provider_var.get()
        _output_clear(self._out_widget)
        self._status.set("Validating...")

        def _fetch():
            from examexam.apis.conversation_and_router import pick_model
            from examexam.frontends.manager_gui import _QueueUI
            from examexam.validate_questions import validate_questions_now

            q: queue.Queue[str] = queue.Queue()
            ui = _QueueUI(q)
            validate_questions_now(file_name=qfile, model=pick_model("", provider, "fast"), ui=ui)
            return q

        def _success(q):
            while not q.empty():
                _output_append(self._out_widget, q.get())
            _output_append(self._out_widget, "\nValidation complete.\n")
            self._status.set("Validate complete.")

        def _error(exc):
            _output_append(self._out_widget, f"\nError: {exc}\n")
            self._status.set("Error during validate.")

        self._runner.run(_fetch, on_success=_success, on_error=_error)


# ── Research panel ─────────────────────────────────────────────────────────


class ResearchPanel(_BasePanel):
    def __init__(self, parent, runner, status_var):
        super().__init__(parent, runner, status_var, "research")
        self._build()

    def _build(self):
        tk.Label(self, text="Research / Study Guide", bg=_CLR_BG, fg=_CLR_ACCENT, font=_FONT_HEADING).pack(
            anchor="w", padx=12, pady=(12, 4)
        )
        form = tk.Frame(self, bg=_CLR_BG)
        form.pack(fill=tk.X, padx=12, pady=4)

        _make_label(form, "Topic:").grid(row=0, column=0, sticky="w", pady=3)
        self._topic_var = tk.StringVar()
        _make_entry(form, self._topic_var, width=50).grid(row=0, column=1, sticky="ew", padx=4)

        _make_label(form, "Model provider:").grid(row=1, column=0, sticky="w", pady=3)
        self._provider_var = tk.StringVar(value="openai")
        ttk.Combobox(
            form,
            textvariable=self._provider_var,
            values=["openai", "anthropic", "google", "mistral"],
            width=15,
            state="readonly",
        ).grid(row=1, column=1, sticky="w", padx=4)

        form.columnconfigure(1, weight=1)

        btn_row = tk.Frame(self, bg=_CLR_BG)
        btn_row.pack(anchor="w", padx=12, pady=8)
        _make_btn(btn_row, "Research", self._run, bg=_CLR_ACCENT, fg=_CLR_BG).pack(side=tk.LEFT, padx=4)

        self._out_widget = _make_output(self)
        self._out_widget.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)

    def _run(self):
        topic = self._topic_var.get().strip()
        if not topic:
            messagebox.showwarning("Missing input", "Please enter a topic.")
            return
        provider = self._provider_var.get()
        _output_clear(self._out_widget)
        self._status.set(f"Researching: {topic[:40]}...")

        def _fetch():
            from examexam.apis.conversation_and_router import pick_model
            from examexam.frontends.manager_gui import _QueueUI
            from examexam.generate_topic_research import generate_topic_research_now

            q: queue.Queue[str] = queue.Queue()
            ui = _QueueUI(q)
            generate_topic_research_now(topic=topic, model=pick_model("", provider, "fast"), ui=ui)
            return q

        def _success(q):
            while not q.empty():
                _output_append(self._out_widget, q.get())
            self._status.set("Research complete.")

        def _error(exc):
            _output_append(self._out_widget, f"\nError: {exc}\n")
            self._status.set("Error during research.")

        self._runner.run(_fetch, on_success=_success, on_error=_error)


# ── Study Plan panel ───────────────────────────────────────────────────────


class StudyPlanPanel(_BasePanel):
    def __init__(self, parent, runner, status_var):
        super().__init__(parent, runner, status_var, "study_plan")
        self._build()

    def _build(self):
        tk.Label(self, text="Study Plan", bg=_CLR_BG, fg=_CLR_ACCENT, font=_FONT_HEADING).pack(
            anchor="w", padx=12, pady=(12, 4)
        )
        form = tk.Frame(self, bg=_CLR_BG)
        form.pack(fill=tk.X, padx=12, pady=4)

        _make_label(form, "TOC file:").grid(row=0, column=0, sticky="w", pady=3)
        self._toc_var = tk.StringVar()
        _make_entry(form, self._toc_var, width=40).grid(row=0, column=1, sticky="ew", padx=4)
        _make_btn(form, "Browse", self._browse).grid(row=0, column=2, padx=2)

        _make_label(form, "Model provider:").grid(row=1, column=0, sticky="w", pady=3)
        self._provider_var = tk.StringVar(value="openai")
        ttk.Combobox(
            form,
            textvariable=self._provider_var,
            values=["openai", "anthropic", "google", "mistral"],
            width=15,
            state="readonly",
        ).grid(row=1, column=1, sticky="w", padx=4)

        form.columnconfigure(1, weight=1)

        btn_row = tk.Frame(self, bg=_CLR_BG)
        btn_row.pack(anchor="w", padx=12, pady=8)
        _make_btn(btn_row, "Generate Study Plan", self._run, bg=_CLR_ACCENT, fg=_CLR_BG).pack(side=tk.LEFT, padx=4)

        self._out_widget = _make_output(self)
        self._out_widget.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)

    def _browse(self):
        path = _pick_file("Select TOC file", [("Text files", "*.txt"), ("All", "*.*")])
        if path:
            self._toc_var.set(path)

    def _run(self):
        toc = self._toc_var.get().strip()
        if not toc:
            messagebox.showwarning("Missing input", "Please select a TOC file.")
            return
        provider = self._provider_var.get()
        _output_clear(self._out_widget)
        self._status.set("Generating study plan...")

        def _fetch():
            from examexam.apis.conversation_and_router import pick_model
            from examexam.frontends.manager_gui import _QueueUI
            from examexam.generate_study_plan import generate_study_plan_now

            q: queue.Queue[str] = queue.Queue()
            ui = _QueueUI(q)
            generate_study_plan_now(toc_file=toc, model=pick_model("", provider, "fast"), ui=ui)
            return q

        def _success(q):
            while not q.empty():
                _output_append(self._out_widget, q.get())
            self._status.set("Study plan complete.")

        def _error(exc):
            _output_append(self._out_widget, f"\nError: {exc}\n")
            self._status.set("Error during study plan.")

        self._runner.run(_fetch, on_success=_success, on_error=_error)


# ── Convert panel ──────────────────────────────────────────────────────────


class ConvertPanel(_BasePanel):
    def __init__(self, parent, runner, status_var):
        super().__init__(parent, runner, status_var, "convert")
        self._build()

    def _build(self):
        tk.Label(self, text="Convert Questions", bg=_CLR_BG, fg=_CLR_ACCENT, font=_FONT_HEADING).pack(
            anchor="w", padx=12, pady=(12, 4)
        )
        form = tk.Frame(self, bg=_CLR_BG)
        form.pack(fill=tk.X, padx=12, pady=4)

        _make_label(form, "Input TOML:").grid(row=0, column=0, sticky="w", pady=3)
        self._in_var = tk.StringVar()
        _make_entry(form, self._in_var, width=40).grid(row=0, column=1, sticky="ew", padx=4)
        _make_btn(form, "Browse", self._browse_in).grid(row=0, column=2, padx=2)

        _make_label(form, "Output base name:").grid(row=1, column=0, sticky="w", pady=3)
        self._base_var = tk.StringVar()
        _make_entry(form, self._base_var, width=40).grid(row=1, column=1, sticky="ew", padx=4)

        form.columnconfigure(1, weight=1)

        btn_row = tk.Frame(self, bg=_CLR_BG)
        btn_row.pack(anchor="w", padx=12, pady=8)
        _make_btn(btn_row, "Convert", self._run, bg=_CLR_ACCENT, fg=_CLR_BG).pack(side=tk.LEFT, padx=4)

        self._out_widget = _make_output(self, height=10)
        self._out_widget.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)

    def _browse_in(self):
        path = _pick_file("Select question TOML", [("TOML files", "*.toml"), ("All", "*.*")])
        if path:
            self._in_var.set(path)
            if not self._base_var.get():
                self._base_var.set(str(Path(path).with_suffix("")))

    def _run(self):
        infile = self._in_var.get().strip()
        base = self._base_var.get().strip()
        if not infile or not base:
            messagebox.showwarning("Missing input", "Please fill in input file and output base name.")
            return
        _output_clear(self._out_widget)
        self._status.set("Converting...")

        def _fetch():
            from examexam.convert_to_pretty import run as convert_run

            convert_run(
                toml_file_path=infile,
                markdown_file_path=f"{base}.md",
                html_file_path=f"{base}.html",
            )
            return f"Converted to:\n  {base}.md\n  {base}.html\n"

        def _success(msg):
            _output_append(self._out_widget, msg)
            self._status.set("Convert complete.")

        def _error(exc):
            _output_append(self._out_widget, f"\nError: {exc}\n")
            self._status.set("Error during convert.")

        self._runner.run(_fetch, on_success=_success, on_error=_error)


# ── Config panel ───────────────────────────────────────────────────────────


class ConfigPanel(_BasePanel):
    def __init__(self, parent, runner, status_var):
        super().__init__(parent, runner, status_var, "config")
        self._build()

    def _build(self):
        tk.Label(self, text="Configuration", bg=_CLR_BG, fg=_CLR_ACCENT, font=_FONT_HEADING).pack(
            anchor="w", padx=12, pady=(12, 4)
        )

        btn_row = tk.Frame(self, bg=_CLR_BG)
        btn_row.pack(anchor="w", padx=12, pady=4)
        _make_btn(btn_row, "Show Config", self._show_config).pack(side=tk.LEFT, padx=4)
        _make_btn(btn_row, "Create Default Config", self._create_default).pack(side=tk.LEFT, padx=4)
        _make_btn(btn_row, "Open in Editor", self._open_editor).pack(side=tk.LEFT, padx=4)

        self._out_widget = _make_output(self)
        self._out_widget.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)
        self._show_config()

    def _show_config(self):
        _output_clear(self._out_widget)

        def _fetch():
            from examexam.config import DEFAULT_CONFIG_FILENAME, config

            config.load()
            lines = [f"Config file: {DEFAULT_CONFIG_FILENAME}\n\n"]
            lines.append("[general]\n")
            for key in (
                "default_n",
                "preferred_cheap_model",
                "preferred_frontier_model",
                "use_frontier_model",
                "override_model",
            ):
                val = config.get(f"general.{key}", "(not set)")
                lines.append(f"  {key} = {val!r}\n")
            return "".join(lines)

        def _success(text):
            _output_append(self._out_widget, text)
            self._status.set("Config loaded.")

        def _error(exc):
            _output_append(self._out_widget, f"Error: {exc}\n")

        self._runner.run(_fetch, on_success=_success, on_error=_error)

    def _create_default(self):
        from examexam.config import DEFAULT_CONFIG_FILENAME, create_default_config_if_not_exists

        created = create_default_config_if_not_exists()
        if created:
            _output_append(self._out_widget, f"Created {DEFAULT_CONFIG_FILENAME}\n")
        else:
            _output_append(self._out_widget, f"{DEFAULT_CONFIG_FILENAME} already exists.\n")

    def _open_editor(self):
        from examexam.config import DEFAULT_CONFIG_FILENAME

        cfg_path = Path(DEFAULT_CONFIG_FILENAME)
        if not cfg_path.exists():
            messagebox.showinfo("Not found", f"{DEFAULT_CONFIG_FILENAME} does not exist. Create it first.")
            return
        import subprocess
        import sys

        if sys.platform == "win32":
            os.startfile(str(cfg_path.resolve()))
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(cfg_path.resolve())])
        else:
            subprocess.Popen(["xdg-open", str(cfg_path.resolve())])


# ── API Keys panel ─────────────────────────────────────────────────────────


class ApiKeysPanel(_BasePanel):
    def __init__(self, parent, runner, status_var):
        super().__init__(parent, runner, status_var, "api_keys")
        self._build()

    def _build(self):
        tk.Label(self, text="API Keys", bg=_CLR_BG, fg=_CLR_ACCENT, font=_FONT_HEADING).pack(
            anchor="w", padx=12, pady=(12, 4)
        )
        tk.Label(
            self,
            text="Keys are masked for security. Values from environment and .env file.",
            bg=_CLR_BG,
            fg=_CLR_DIM,
            font=_FONT_UI,
        ).pack(anchor="w", padx=12)

        btn_row = tk.Frame(self, bg=_CLR_BG)
        btn_row.pack(anchor="w", padx=12, pady=6)
        _make_btn(btn_row, "Refresh", self._show_keys).pack(side=tk.LEFT, padx=4)

        self._out_widget = _make_output(self, height=20)
        self._out_widget.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)
        self._show_keys()

    def _show_keys(self):
        _output_clear(self._out_widget)
        from examexam.doctor import _env_key_status

        keys = [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
            "MISTRAL_API_KEY",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_DEFAULT_REGION",
            "COHERE_API_KEY",
            "AI21_API_KEY",
        ]
        lines = [f"{'Key':<32} {'Status'}\n", "-" * 60 + "\n"]
        for key in keys:
            lines.append(f"{key:<32} {_env_key_status(key)}\n")

        # .env file presence
        env_path = Path(".env")
        lines.append(f"\n.env file: {env_path.resolve()} ({'exists' if env_path.exists() else 'not found'})\n")

        for line in lines:
            _output_append(self._out_widget, line)
        self._status.set("API key status refreshed.")


# ── Doctor panel ───────────────────────────────────────────────────────────


class DoctorPanel(_BasePanel):
    def __init__(self, parent, runner, status_var):
        super().__init__(parent, runner, status_var, "doctor")
        self._build()

    def _build(self):
        tk.Label(self, text="Doctor", bg=_CLR_BG, fg=_CLR_ACCENT, font=_FONT_HEADING).pack(
            anchor="w", padx=12, pady=(12, 4)
        )
        tk.Label(
            self,
            text="Diagnostic report for tech support.",
            bg=_CLR_BG,
            fg=_CLR_DIM,
            font=_FONT_UI,
        ).pack(anchor="w", padx=12)

        btn_row = tk.Frame(self, bg=_CLR_BG)
        btn_row.pack(anchor="w", padx=12, pady=6)
        _make_btn(btn_row, "Run Diagnostics", self._run).pack(side=tk.LEFT, padx=4)
        _make_btn(btn_row, "Copy to Clipboard", self._copy).pack(side=tk.LEFT, padx=4)

        self._out_widget = _make_output(self)
        self._out_widget.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)

        self._last_report = ""
        self._run()

    def _run(self):
        _output_clear(self._out_widget)
        self._status.set("Running diagnostics...")

        def _fetch():
            from examexam.doctor import run_doctor

            return run_doctor()

        def _success(report):
            self._last_report = report
            _output_append(self._out_widget, report)
            self._status.set("Diagnostics complete.")

        def _error(exc):
            _output_append(self._out_widget, f"Error: {exc}\n")
            self._status.set("Error during diagnostics.")

        self._runner.run(_fetch, on_success=_success, on_error=_error)

    def _copy(self):
        if self._last_report and self._out_widget.winfo_toplevel():
            self._out_widget.winfo_toplevel().clipboard_clear()
            self._out_widget.winfo_toplevel().clipboard_append(self._last_report)
            self._status.set("Report copied to clipboard.")


# ── Take Exam launcher panel ───────────────────────────────────────────────


class TakeExamPanel(_BasePanel):
    def __init__(self, parent, runner, status_var):
        super().__init__(parent, runner, status_var, "take_exam")
        self._build()

    def _build(self):
        tk.Label(self, text="Take Exam", bg=_CLR_BG, fg=_CLR_ACCENT, font=_FONT_HEADING).pack(
            anchor="w", padx=12, pady=(12, 4)
        )
        tk.Label(
            self,
            text="Select a question file and launch the exam.",
            bg=_CLR_BG,
            fg=_CLR_DIM,
            font=_FONT_UI,
        ).pack(anchor="w", padx=12)

        form = tk.Frame(self, bg=_CLR_BG)
        form.pack(fill=tk.X, padx=12, pady=8)

        _make_label(form, "Question file:").grid(row=0, column=0, sticky="w", pady=3)
        self._file_var = tk.StringVar()
        _make_entry(form, self._file_var, width=42).grid(row=0, column=1, sticky="ew", padx=4)
        _make_btn(form, "Browse", self._browse).grid(row=0, column=2, padx=2)
        form.columnconfigure(1, weight=1)

        tk.Label(self, text="Frontend:", bg=_CLR_BG, fg=_CLR_FG, font=_FONT_UI).pack(anchor="w", padx=12, pady=(8, 0))
        self._frontend_var = tk.StringVar(value="gui")
        frontend_row = tk.Frame(self, bg=_CLR_BG)
        frontend_row.pack(anchor="w", padx=24, pady=4)
        for val, label in [("cli", "CLI (terminal)"), ("gui", "GUI (Tkinter)"), ("web", "Web (browser)")]:
            ttk.Radiobutton(frontend_row, text=label, variable=self._frontend_var, value=val).pack(side=tk.LEFT, padx=8)

        btn_row = tk.Frame(self, bg=_CLR_BG)
        btn_row.pack(anchor="w", padx=12, pady=12)
        _make_btn(btn_row, "Launch Exam", self._launch, bg=_CLR_OK, fg="#000000").pack(side=tk.LEFT, padx=4)

        self._out_widget = _make_output(self, height=8)
        self._out_widget.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)

    def _browse(self):
        path = _pick_file("Select question TOML", [("TOML files", "*.toml"), ("All", "*.*")])
        if path:
            self._file_var.set(path)

    def _launch(self):
        qfile = self._file_var.get().strip()
        frontend = self._frontend_var.get()
        _output_clear(self._out_widget)

        if frontend == "gui":
            # Launch the tkinter exam UI in a new top-level process/thread
            self._status.set("Launching exam GUI...")
            import subprocess
            import sys

            cmd = [sys.executable, "-m", "examexam", "--frontend", "gui", "take"]
            if qfile:
                cmd += ["--question-file", qfile]
            try:
                subprocess.Popen(cmd)
                _output_append(self._out_widget, "Exam GUI launched in new window.\n")
            except Exception as exc:
                _output_append(self._out_widget, f"Error: {exc}\n")
            self._status.set("Exam launched.")
        elif frontend == "web":
            self._status.set("Launching web exam...")
            import subprocess
            import sys

            cmd = [sys.executable, "-m", "examexam", "--frontend", "web", "take"]
            if qfile:
                cmd += ["--question-file", qfile]
            try:
                subprocess.Popen(cmd)
                _output_append(self._out_widget, "Web exam launched. Open http://localhost:8000 in your browser.\n")
            except Exception as exc:
                _output_append(self._out_widget, f"Error: {exc}\n")
            self._status.set("Web exam launched.")
        else:
            # CLI – run in background thread, stream output
            self._status.set("Running CLI exam...")
            import subprocess
            import sys

            cmd = [sys.executable, "-m", "examexam", "take"]
            if qfile:
                cmd += ["--question-file", qfile]
            _output_append(self._out_widget, f"Launching: {' '.join(cmd)}\n")
            _output_append(self._out_widget, "CLI exam runs in terminal. Check your terminal window.\n")
            subprocess.Popen(cmd, creationflags=0x10 if sys.platform == "win32" else 0)
            self._status.set("CLI exam launched.")


# ── Queue-based UI adapter ─────────────────────────────────────────────────


class _QueueUI:
    """Minimal FrontendUI that puts messages into a queue for later display."""

    def __init__(self, q: queue.Queue[str]) -> None:
        self._q = q

    def show_message(self, message: str, *, style: str = "") -> None:
        self._q.put(message + "\n")

    def show_error(self, message: str) -> None:
        self._q.put(f"ERROR: {message}\n")

    def confirm(self, message: str, *, default: bool = True) -> bool:
        return default

    def get_input(self, prompt: str) -> str:
        return ""

    def show_test_selection(self, tests):
        return None

    def show_session_info(self, *a, **kw):
        pass

    def show_question(self, *a, **kw):
        pass

    def get_answer(self, *a, **kw):
        return ""

    def show_answer_feedback(self, *a, **kw):
        pass

    def show_results(self, *a, **kw):
        pass

    def wait_for_continue(self) -> str:
        return ""

    def clear_screen(self) -> None:
        pass

    def progress_start(self, total: int, description: str = "") -> str:
        self._q.put(f"{description} (0/{total})\n")
        return "task"

    def progress_update(self, task_id: str, advance: int = 1, description: str = "") -> None:
        if description:
            self._q.put(description + "\n")

    def progress_finish(self, task_id: str) -> None:
        pass

    def show_panel(self, content: str, *, title: str = "", style: str = "") -> None:
        if title:
            self._q.put(f"[{title}]\n")
        self._q.put(content + "\n")

    def show_markdown(self, content: str) -> None:
        self._q.put(content + "\n")

    def show_rule(self, title: str = "") -> None:
        self._q.put(("-" * 40) + (f" {title} " if title else "") + "\n")

    def run(self, callback: Any = None) -> None:
        pass

    def shutdown(self) -> None:
        pass


# ── Main app ───────────────────────────────────────────────────────────────


class ManagerApp:
    """Three-panel manager GUI for examexam."""

    _BUTTONS = [
        ("home", "Home"),
        ("generate", "Generate Questions"),
        ("validate", "Validate Questions"),
        ("research", "Research / Study Guide"),
        ("study_plan", "Study Plan"),
        ("convert", "Convert Questions"),
        ("take_exam", "Take Exam"),
        None,  # separator
        ("config", "Configuration"),
        ("api_keys", "API Keys"),
        ("doctor", "Doctor"),
    ]

    _PANELS = {
        "home": HomePanel,
        "generate": GeneratePanel,
        "validate": ValidatePanel,
        "research": ResearchPanel,
        "study_plan": StudyPlanPanel,
        "convert": ConvertPanel,
        "take_exam": TakeExamPanel,
        "config": ConfigPanel,
        "api_keys": ApiKeysPanel,
        "doctor": DoctorPanel,
    }

    def __init__(self) -> None:
        self._root = tk.Tk()
        self._root.title("ExamExam Manager")
        self._root.geometry("1100x700")
        self._root.minsize(900, 600)
        self._root.configure(bg=_CLR_BG)

        self._status_var = tk.StringVar(value="Ready")
        self._runner = _BackgroundRunner(self._root)
        self._current_panel: _BasePanel | None = None
        self._current_key: str = ""

        self._apply_styles()
        self._build_ui()
        self._show_panel("home")
        self._root.bind("<Control-q>", lambda _e: self._root.quit())

    def _apply_styles(self) -> None:
        style = ttk.Style(self._root)
        style.theme_use("clam")
        style.configure("TEntry", fieldbackground=_CLR_BG_ALT, foreground=_CLR_FG, insertcolor=_CLR_FG)
        style.configure("TCombobox", fieldbackground=_CLR_BG_ALT, foreground=_CLR_FG)
        style.configure("TRadiobutton", background=_CLR_BG, foreground=_CLR_FG)
        style.configure("TSeparator", background=_CLR_BORDER)

    def _build_ui(self) -> None:
        # ── Left sidebar ────────────────────────────────────────────────
        sidebar = tk.Frame(self._root, bg=_CLR_SIDEBAR, width=200)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        tk.Label(
            sidebar,
            text="ExamExam",
            bg=_CLR_SIDEBAR,
            fg=_CLR_ACCENT,
            font=("Segoe UI", 13, "bold"),
        ).pack(pady=(18, 12), padx=10)

        ttk.Separator(sidebar, orient="horizontal").pack(fill=tk.X, padx=8, pady=4)

        self._sidebar_buttons: dict[str, tk.Button] = {}
        for item in self._BUTTONS:
            if item is None:
                ttk.Separator(sidebar, orient="horizontal").pack(fill=tk.X, padx=8, pady=8)
                continue
            key, label = item
            btn = tk.Button(
                sidebar,
                text=label,
                command=lambda k=key: self._show_panel(k),
                bg=_CLR_SIDEBAR,
                fg=_CLR_FG,
                activebackground=_CLR_BTN_ACTIVE,
                activeforeground=_CLR_FG,
                relief=tk.FLAT,
                font=_FONT_BTN,
                anchor="w",
                padx=12,
                pady=6,
                cursor="hand2",
            )
            btn.pack(fill=tk.X, padx=4, pady=1)
            self._sidebar_buttons[key] = btn

        # ── Middle content area ────────────────────────────────────────
        self._content_frame = tk.Frame(self._root, bg=_CLR_BG)
        self._content_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # ── Right help panel ───────────────────────────────────────────
        right = tk.Frame(self._root, bg=_CLR_BG_ALT, width=220)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)

        tk.Label(right, text="Help", bg=_CLR_BG_ALT, fg=_CLR_ACCENT, font=_FONT_HEADING).pack(
            anchor="w", padx=10, pady=(12, 4)
        )

        self._help_text = scrolledtext.ScrolledText(
            right,
            bg=_CLR_BG_ALT,
            fg=_CLR_FG,
            font=("Segoe UI", 9),
            relief=tk.FLAT,
            wrap=tk.WORD,
            state="disabled",
        )
        self._help_text.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        ttk.Separator(right, orient="horizontal").pack(fill=tk.X, padx=6, pady=4)

        tk.Label(right, text="Cheat Sheet", bg=_CLR_BG_ALT, fg=_CLR_ACCENT, font=_FONT_HEADING).pack(
            anchor="w", padx=10, pady=4
        )

        cheat = scrolledtext.ScrolledText(
            right,
            bg=_CLR_BG_ALT,
            fg=_CLR_DIM,
            font=("Consolas", 8),
            relief=tk.FLAT,
            wrap=tk.WORD,
            state="normal",
            height=18,
        )
        cheat.insert(tk.END, _CHEATSHEET)
        cheat.configure(state="disabled")
        cheat.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        # ── Status bar ────────────────────────────────────────────────
        status_bar = tk.Frame(self._root, bg=_CLR_SIDEBAR, height=24)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        tk.Label(
            status_bar,
            textvariable=self._status_var,
            bg=_CLR_SIDEBAR,
            fg=_CLR_DIM,
            font=("Segoe UI", 9),
            anchor="w",
        ).pack(side=tk.LEFT, padx=10)

    def _show_panel(self, key: str) -> None:
        if self._current_panel is not None:
            self._current_panel.destroy()

        # Highlight active sidebar button
        for k, btn in self._sidebar_buttons.items():
            if k == key:
                btn.configure(bg=_CLR_BTN, fg=_CLR_ACCENT)
            else:
                btn.configure(bg=_CLR_SIDEBAR, fg=_CLR_FG)

        cls = self._PANELS.get(key, HomePanel)
        panel = cls(self._content_frame, self._runner, self._status_var)
        panel.pack(fill=tk.BOTH, expand=True)
        self._current_panel = panel
        self._current_key = key

        # Update help text
        help_txt = panel.help_text
        self._help_text.configure(state="normal")
        self._help_text.delete("1.0", tk.END)
        self._help_text.insert(tk.END, help_txt)
        self._help_text.configure(state="disabled")

    def run(self) -> None:
        self._root.mainloop()


def launch_manager() -> None:
    """Entry point: launch the management GUI."""
    app = ManagerApp()
    app.run()
