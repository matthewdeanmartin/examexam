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
from examexam.apis.conversation_and_router import FRONTIER_MODELS
from examexam.convert_to_pretty import run as convert_questions_run
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


def main(argv: Sequence[str] | None = None) -> int:
    """Main function for the command-line interface."""
    start_background_update_check("examexam", __about__.__version__)
    parser = SmartParser(
        prog=__about__.__title__,
        description="A CLI for generating, taking, and managing exams.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__about__.__version__}")

    parser.add_argument("--verbose", action="store_true", required=False, help="Enable detailed logging.")
    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)

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

    models = list(_ for _ in FRONTIER_MODELS.keys())
    models_string = ", ".join(models)
    generate_parser.add_argument(
        "--model",
        type=str,
        default="openai",
        help=f"Model to use for generating questions (e.g., {models_string}). Default: openai",
    )

    # --- Validate Command ---
    validate_parser = subparsers.add_parser("validate", help="Validate exam questions using an LLM.")
    validate_parser.add_argument(
        "--question-file",
        type=str,
        required=True,
        help="Path to the TOML question file to validate.",
    )
    validate_parser.add_argument(
        "--model",
        type=str,
        default="openai",
        help=f"Model to use for validation, e.g. {models_string} (default: openai).",
    )

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
    research_parser.add_argument(
        "--model",
        type=str,
        default="openai",
        help=f"Model to use for generating the study guide (e.g., {models_string}). Default: openai",
    )

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
    study_plan_parser.add_argument(
        "--model",
        type=str,
        default="openai",
        help=f"Model to use for generating the study plan (e.g., {models_string}). Default: openai",
    )

    # --- Customize Command (New) ---
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

    argcomplete.autocomplete(parser)

    args = parser.parse_args(args=argv)

    if args.verbose:
        config = logging_config.generate_config()
        logging.config.dictConfig(config)
    else:
        # Configure a basic logger for user-facing messages
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.command == "take":
        if hasattr(args, "question_file") and args.question_file:
            take_exam_now(question_file=args.question_file)
        else:
            take_exam_now()
    elif args.command == "generate":
        toc_file = args.toc_file
        if not toc_file.endswith(".txt"):
            toc_file_base = toc_file + ".txt"
        else:
            toc_file_base = toc_file
        generate_questions_now(
            questions_per_toc_topic=args.n,
            file_name=args.output_file or toc_file_base.replace(".txt", ".toml"),
            toc_file=args.toc_file,
            model=args.model,
            system_prompt="You are a test maker.",
        )
    elif args.command == "validate":
        validate_questions_now(file_name=args.question_file, model=args.model)
    elif args.command == "research":
        generate_topic_research_now(topic=args.topic, model=args.model)
    elif args.command == "study-plan":
        generate_study_plan_now(toc_file=args.toc_file, model=args.model)
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
