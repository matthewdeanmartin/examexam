from __future__ import annotations

import logging
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter, sleep

import dotenv
import rtoml as toml

# --- Rich UI for users ---
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

from examexam.apis.conversation_and_router import Conversation, Router
from examexam.jinja_management import jinja_env

# Load environment variables (e.g., OPENAI_API_KEY)
dotenv.load_dotenv()

# ---- Logging setup (for developers) ----
# Keep logger.info/debug; print user-facing stuff with Rich Console.
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[RichHandler(rich_tracebacks=True, markup=True, show_time=False, show_level=True)],
    )

console = Console()


# ---------- Helpers ----------
@dataclass
class GenStats:
    calls: int = 0
    tokens_prompt: int | None = None
    tokens_completion: int | None = None
    tokens_total: int | None = None
    last_call_seconds: float | None = None


def create_new_conversation(system_prompt: str) -> Conversation:
    logger.debug("Creating new Conversation with system prompt length=%d", len(system_prompt))
    conversation = Conversation(system=system_prompt)
    return conversation


class FatalLLMError(Exception):
    """Errors that should not be retried (e.g., missing API key)."""


def _fatal_if_misconfigured(model: str) -> None:
    """Raise FatalLLMError for obviously fatal misconfigurations before calling LLM."""
    # Allow a stub model name used for tests.
    if model.lower() not in {"fakebot", "none", "noop"} and not os.getenv("OPENAI_API_KEY"):
        raise FatalLLMError("OPENAI_API_KEY is not set. Set the environment variable or pass api_key to the client.")


def _is_fatal_message(msg: str) -> bool:
    msg_lower = msg.lower()
    fatal_markers = [
        "unknown model" "api_key client option must be set",
        "no api key",
        "invalid api key",
        "unauthorized",
        "model not found",
        "does not exist or you do not have access",
        "access denied",
    ]
    return any(m in msg_lower for m in fatal_markers)


def extract_toml(markdown_content: str) -> str | None:
    """Extract TOML fenced block from markdown content."""
    if not markdown_content:
        logger.debug("No content returned from router.call; skipping TOML extract")
        return None
    match = re.search(r"```toml\n(.*?)\n```", markdown_content, re.DOTALL)
    if match:
        logger.info("TOML content found in response.")
        return match.group(1)
    logger.debug("TOML fenced block not found in content (len=%d)", len(markdown_content))
    return None


def generate_questions(
    prompt: str,
    n: int,
    conversation: Conversation,
    service: str,
    model: str,
    *,
    max_retries: int = 2,
    retry_delay_seconds: float = 1.5,
    stats: GenStats | None = None,
) -> dict[str, list[dict[str, str]]] | None:
    """Request questions from an LLM and return parsed TOML as a dict.

    - Avoids looping on fatal errors (API key missing, auth, model not found).
    - Retries a couple times on transient failures.
    """
    logger.info("Generating %d questions with prompt: %s", n, prompt)
    _fatal_if_misconfigured(model)

    # Render the prompt from the Jinja2 template
    try:
        template = jinja_env.get_template("generate.md.j2")
        user_prompt = template.render(n=n, prompt=prompt)
    except Exception as e:
        logger.error("Failed to load or render Jinja2 template 'generate.md.j2': %s", e)
        raise

    router = Router(conversation)

    attempts = 0
    first_started = perf_counter()

    while True:
        attempts += 1
        try:
            started = perf_counter()
            content = router.call(user_prompt, model)
            duration = perf_counter() - started
            if stats is not None:
                stats.calls += 1
                stats.last_call_seconds = duration
            logger.debug("router.call returned content length=%d in %.2fs", len(content or ""), duration)
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
            continue

        toml_content = extract_toml(content)
        if toml_content is None:
            logger.debug(
                "Attempt %d: TOML not found; %s", attempts, "retrying" if attempts <= max_retries else "giving up"
            )
            if attempts > max_retries:
                return None
            sleep(retry_delay_seconds)
            continue

        # Try to parse TOML
        try:
            parsed = toml.loads(toml_content)
            logger.debug("Parsed TOML successfully with %d questions", len(parsed.get("questions", [])))
            logger.info(
                "Time taken to generate questions: %s",
                datetime.now() - datetime.fromtimestamp(first_started),
            )
            return parsed
        except Exception as e:  # noqa: BLE001
            logger.error("Error loading TOML content: %s", e)
            if attempts > max_retries:
                return None
            sleep(retry_delay_seconds)


def save_toml_to_file(toml_content: str, file_name: str) -> None:
    """Save TOML to file, appending to existing [[questions]]."""
    path = Path(file_name)
    logger.debug("Saving TOML to %s (exists=%s)", path, path.exists())
    if path.exists():
        with path.open(encoding="utf-8") as file:
            existing_content = toml.load(file)
        existing_content.setdefault("questions", [])
        new_questions = toml.loads(toml_content).get("questions", [])
        logger.debug(
            "Extending existing %d questions with %d new questions",
            len(existing_content["questions"]),
            len(new_questions),
        )
        existing_content["questions"].extend(new_questions)
        with path.open("w", encoding="utf-8") as file:
            toml.dump(existing_content, file)
    else:
        with path.open("w", encoding="utf-8") as file:
            file.write(toml_content)
    console.print(f"[bold green]TOML content saved to[/] {file_name}")


def generate_questions_now(
    questions_per_toc_topic: int,
    file_name: str,
    toc_file: str,
    system_prompt: str,
    model: str = "fakebot",
) -> int:
    """Main execution with Rich progress UI."""
    toc_path = Path(toc_file)
    if not toc_path.exists():
        console.print(Panel.fit(f"[red]TOC file not found:[/] {toc_file}", title="Error"))
        return 0

    with toc_path.open(encoding="utf-8") as f:
        services = [line.strip() for line in f if line.strip()]

    total_topics = len(services)
    if total_topics == 0:
        console.print(Panel.fit("[yellow]TOC file is empty.[/]", title="Nothing to do"))
        return 0

    console.rule("[bold]Exam Question Generation")
    console.print(
        f"Generating [bold]{questions_per_toc_topic}[/] per topic across [bold]{total_topics}[/] topics with model [italic]{model}[/]…\n"
    )

    # Overall and per-topic progress bars.
    total_so_far = 0

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

    stats = GenStats()

    absolute_failures = 0
    with progress:
        overall_task = progress.add_task("Overall", total=total_topics)

        for idx, service in enumerate(services, start=1):
            topic_task = progress.add_task(f"{idx}/{total_topics} {service}", total=1)
            prompt = f"They must all be '{service}' questions."
            conversation = create_new_conversation(system_prompt)

            t0 = perf_counter()
            questions = generate_questions(
                prompt,
                questions_per_toc_topic,
                conversation,
                service,
                model,
                stats=stats,
            )
            dt = perf_counter() - t0

            if not questions:
                absolute_failures += 1
                progress.update(topic_task, description=f"{idx}/{total_topics} {service} [red](failed)[/]")
                progress.advance(topic_task)
                progress.advance(overall_task)
                if absolute_failures > 3:
                    print("Three times bot absolutely failed to generate any proper questions")
                    return 0
                continue

            for question in questions.get("questions", []):
                question["id"] = str(uuid.uuid4())

            total_so_far += len(questions.get("questions", []))
            logger.info("Total questions so far: %d", total_so_far)

            toml_content = toml.dumps(questions)
            save_toml_to_file(toml_content, file_name)

            progress.update(
                topic_task,
                description=f"{idx}/{total_topics} {service} [green](ok in {dt:.2f}s)[/]",
            )
            progress.advance(topic_task)
            progress.advance(overall_task)

    console.rule()
    console.print(
        Panel.fit(
            f"[bold green]Done[/]: generated [bold]{total_so_far}[/] questions across [bold]{total_topics}[/] topics.",
            title="Summary",
        )
    )
    return total_so_far


if __name__ == "__main__":
    # Example direct run; tweak as needed.
    generate_questions_now(
        questions_per_toc_topic=10,
        file_name="personal_multiple_choice_tests.toml",
        toc_file="../example_inputs/personally_important.txt",
        model="openai",
        system_prompt="We are writing multiple choice tests.",
    )
