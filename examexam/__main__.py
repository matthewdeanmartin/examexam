from __future__ import annotations

import argparse
import logging
import logging.config
import sys
from collections.abc import Sequence
from pathlib import Path

import argcomplete
import dotenv

from examexam import __about__, logging_config
from examexam.apis.conversation_and_router import FRONTIER_MODELS, pick_model
from examexam.convert_to_pretty import run as convert_questions_run
from examexam.frontends import FRONTEND_CHOICES, get_frontend
from examexam.generate_questions import generate_questions_now
from examexam.generate_study_plan import generate_study_plan_now
from examexam.generate_topic_research import generate_topic_research_now
from examexam.jinja_management import deploy_for_customization
from examexam.take_exam import take_exam_now
from examexam.utils.cli_suggestions import SmartParser
from examexam.utils.update_checker import start_background_update_check
from examexam.validate_questions import validate_questions_now

# Load environment variables (e.g., OPENAI_API_KEY)
dotenv.load_dotenv()


def add_model_args(parser) -> None:
    models = list(_ for _ in FRONTIER_MODELS.keys())
    models_string = ", ".join(models)
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Exact model is to use. Default: blank meaning use provider/class",
    )

    parser.add_argument(
        "--model-provider",
        type=str,
        default="openai",
        help=f"Model provider to use for generating questions (e.g., {models_string}). Default: openai",
    )

    parser.add_argument(
        "--model-class",
        type=str,
        default="fast",
        help="'frontier' or 'fast' model Default: fast",
    )


def _interactive_launcher() -> int:
    """Show an interactive terminal menu when no subcommand is given."""
    import importlib.util

    _has_tk = importlib.util.find_spec("tkinter") is not None

    if _has_tk:
        return _launch_interactive_tk()
    return _launch_interactive_cli()


def _launch_interactive_tk() -> int:
    """Show a small Tkinter launcher dialog."""
    import tkinter as tk

    # Catppuccin Mocha colours (matching manager_gui.py)
    _BG = "#1e1e2e"
    _FG = "#cdd6f4"
    _ACCENT = "#89b4fa"
    _BTN = "#313244"
    _BTN_ACTIVE = "#45475a"
    _OK = "#22c55e"

    root = tk.Tk()
    root.title(f"ExamExam {__about__.__version__}")
    root.geometry("420x340")
    root.resizable(False, False)
    root.configure(bg=_BG)

    # Centre on screen
    root.update_idletasks()
    x = (root.winfo_screenwidth() - 420) // 2
    y = (root.winfo_screenheight() - 340) // 2
    root.geometry(f"+{x}+{y}")

    choice = tk.StringVar(value="")

    tk.Label(root, text="ExamExam", bg=_BG, fg=_ACCENT, font=("Segoe UI", 22, "bold")).pack(pady=(28, 4))
    tk.Label(root, text="What would you like to do?", bg=_BG, fg=_FG, font=("Segoe UI", 11)).pack(pady=(0, 18))

    def _pick(value: str) -> None:
        choice.set(value)
        root.quit()

    def _btn(parent, text, value, bg=_BTN):
        return tk.Button(
            parent,
            text=text,
            command=lambda: _pick(value),
            bg=bg,
            fg=_FG,
            activebackground=_BTN_ACTIVE,
            activeforeground=_FG,
            relief=tk.FLAT,
            font=("Segoe UI", 11),
            width=24,
            pady=8,
            cursor="hand2",
        )

    _btn(root, "Take an Exam", "take", bg=_OK).pack(pady=4)
    _btn(root, "Manage Exams (GUI)", "manage").pack(pady=4)
    _btn(root, "Manage Exams (CLI)", "manage_cli").pack(pady=4)
    _btn(root, "Doctor / Diagnostics", "doctor").pack(pady=4)

    root.protocol("WM_DELETE_WINDOW", root.quit)
    root.mainloop()

    action = choice.get()
    root.destroy()

    if action == "take":
        from examexam.frontends.tkinter_ui import TkinterUI

        ui = TkinterUI()
        ui.run(callback=lambda: take_exam_now(ui=ui))
    elif action == "manage":
        from examexam.frontends.manager_gui import launch_manager

        launch_manager()
    elif action == "manage_cli":
        # Fall through to the CLI with the Rich UI – show help
        from rich.console import Console

        Console().print(
            "[bold cyan]ExamExam CLI[/bold cyan]\n\n"
            "Run [yellow]examexam --help[/yellow] to see all commands, or use the manager GUI."
        )
        return 0
    elif action == "doctor":
        from examexam.doctor import run_doctor

        print(run_doctor())
    return 0


def _launch_interactive_cli() -> int:
    """Fallback interactive launcher for headless/no-tkinter environments."""
    print(f"\nExamExam {__about__.__version__}")
    print("=" * 40)
    print("What would you like to do?")
    print("  1. Take an exam")
    print("  2. Generate questions")
    print("  3. Doctor / diagnostics")
    print("  4. Show help")
    print("  q. Quit")
    print()
    try:
        choice = input("Enter choice: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return 0

    if choice == "1":
        from examexam.frontends.rich_ui import RichUI

        ui = RichUI()
        take_exam_now(ui=ui)
    elif choice == "2":
        print("\nRun: examexam generate --toc-file <topics.txt>")
    elif choice == "3":
        from examexam.doctor import run_doctor

        print(run_doctor())
    elif choice == "4":
        print("\nRun: examexam --help")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Main function for the command-line interface."""
    start_background_update_check("examexam", __about__.__version__)
    parser = SmartParser(
        prog=__about__.__title__,
        description="A CLI for generating, taking, and managing exams.",
        formatter_class=argparse.RawTextHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__about__.__version__}")

    parser.add_argument("--verbose", action="store_true", required=False, help="Enable detailed logging.")

    parser.add_argument(
        "--frontend",
        type=str,
        default="cli",
        choices=FRONTEND_CHOICES,
        help="UI frontend to use: cli (Rich terminal), gui (Tkinter), tui (Textual), web (browser). Default: cli",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- Take Command ---
    take_parser = subparsers.add_parser("take", help="Take an exam from a TOML file.")
    take_parser.add_argument(
        "--question-file", type=str, default="", required=False, help="Path to the TOML question file."
    )

    # --- Generate Command ---
    generate_parser = subparsers.add_parser("generate", help="Generate new exam questions using an LLM.")

    generate_parser.add_argument(
        "--toc-file",
        type=str,
        required=True,
        help="Path to a text file containing the table of contents or topics, one per line.",
    )
    generate_parser.add_argument(
        "--output-file",
        type=str,
        required=False,
        help="Path to the output TOML file where questions will be saved.",
    )
    generate_parser.add_argument(
        "-n",
        type=int,
        default=5,
        help="Number of questions to generate per topic (default: 5).",
    )

    add_model_args(generate_parser)

    # --- Validate Command ---
    validate_parser = subparsers.add_parser("validate", help="Validate exam questions using an LLM.")
    validate_parser.add_argument(
        "--question-file",
        type=str,
        required=True,
        help="Path to the TOML question file to validate.",
    )
    add_model_args(validate_parser)

    # --- Convert Command ---
    convert_parser = subparsers.add_parser("convert", help="Convert a TOML question file to Markdown and HTML formats.")
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

    # --- Research Command ---
    research_parser = subparsers.add_parser("research", help="Generate a study guide for a specific topic.")
    research_parser.add_argument(
        "--topic",
        type=str,
        required=True,
        help="The topic to generate a study guide for.",
    )
    add_model_args(research_parser)

    # --- Study Plan Command ---
    study_plan_parser = subparsers.add_parser(
        "study-plan", help="Generate a study plan for a list of topics from a file."
    )
    study_plan_parser.add_argument(
        "--toc-file",
        type=str,
        required=True,
        help="Path to a text file containing the topics, one per line.",
    )
    add_model_args(study_plan_parser)

    # --- Customize Command ---
    customize_parser = subparsers.add_parser(
        "customize", help="Deploy Jinja2 templates to a local directory for customization."
    )
    customize_parser.add_argument(
        "--target-dir",
        type=str,
        default=".",
        help="The directory where the 'prompts' folder will be created (default: current directory).",
    )
    customize_parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite of existing templates, even if they have been modified by the user.",
    )

    # --- Doctor Command ---
    subparsers.add_parser("doctor", help="Print diagnostic information for tech support.")

    # --- Manager Command ---
    subparsers.add_parser("manager", help="Launch the exam management GUI.")

    argcomplete.autocomplete(parser)

    args = parser.parse_args(args=argv)

    # ── No subcommand: launch interactive launcher ─────────────────────────
    if not args.command:
        return _interactive_launcher()

    if args.verbose:
        config = logging_config.generate_config()
        logging.config.dictConfig(config)
    else:
        # Configure a basic logger for user-facing messages
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    # ── Doctor command ────────────────────────────────────────────────────
    if args.command == "doctor":
        from examexam.doctor import run_doctor

        print(run_doctor())
        return 0

    # ── Manager command ───────────────────────────────────────────────────
    if args.command == "manager":
        from examexam.frontends.manager_gui import launch_manager

        launch_manager()
        return 0

    # Instantiate the selected frontend
    ui = get_frontend(args.frontend)
    needs_event_loop = args.frontend in ("gui", "tui")

    if args.frontend == "web":
        if args.command != "take":
            parser.error("The web frontend currently supports only the 'take' command.")
        configure_exam = getattr(ui, "configure_take_exam", None)
        if configure_exam is None:
            raise RuntimeError("Selected web frontend cannot be configured for exam taking.")
        configure_exam(args.question_file or None)
        ui.run()
        return 0

    def _run_command() -> None:
        """Execute the selected command. Runs on main thread (CLI) or worker thread (GUI/TUI)."""
        if args.command == "take":
            if hasattr(args, "question_file") and args.question_file:
                take_exam_now(question_file=args.question_file, ui=ui)
            else:
                take_exam_now(ui=ui)
        elif args.command == "generate":
            toc_file = args.toc_file
            if not toc_file.endswith(".txt"):
                toc_file_base = toc_file + ".txt"
            else:
                toc_file_base = toc_file
            model = pick_model(args.model, args.model_provider, args.model_class)

            generate_questions_now(
                questions_per_toc_topic=args.n,
                file_name=args.output_file or toc_file_base.replace(".txt", ".toml"),
                toc_file=args.toc_file,
                model=model,
                system_prompt="You are a test maker.",
                ui=ui,
            )
        elif args.command == "validate":
            model = pick_model(args.model, args.model_provider, args.model_class)
            validate_questions_now(file_name=args.question_file, model=model, ui=ui)
        elif args.command == "research":
            model = pick_model(args.model, args.model_provider, args.model_class)
            generate_topic_research_now(topic=args.topic, model=model, ui=ui)
        elif args.command == "study-plan":
            model = pick_model(args.model, args.model_provider, args.model_class)
            generate_study_plan_now(toc_file=args.toc_file, model=model, ui=ui)
        elif args.command == "convert":
            md_path = f"{args.output_base_name}.md"
            html_path = f"{args.output_base_name}.html"
            convert_questions_run(
                toml_file_path=args.input_file,
                markdown_file_path=md_path,
                html_file_path=html_path,
            )
        elif args.command == "customize":
            target_path = Path(args.target_dir)
            logging.info(f"Deploying templates to '{target_path.resolve()}/prompts'...")
            deploy_for_customization(target_dir=target_path, force=args.force)
        else:
            parser.print_help()

    if needs_event_loop:
        # GUI/TUI frontends need the main thread for their event loop.
        # Business logic runs in a background worker thread via ui.run(callback=...).
        ui.run(callback=_run_command)
    else:
        # CLI runs synchronously on the main thread.
        _run_command()

    return 0


if __name__ == "__main__":
    sys.exit(main())
