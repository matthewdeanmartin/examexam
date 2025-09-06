"""
Call a bot to create study guide

    router = Router(conversation)
    content = router.call(user_prompt, model)

Use jinja templating strategy as seen elsewhere.

Create a study guide in pwd folder pwd/study_guide/(test_name).md

- Original question, answers, explanations
- Searches
    - Google plain search
    - google searches with operators
    - bing plain search
    - bing with operators
    - Kagi plain
    - Kagi with operators

Add more md text to study guide

Display in terminal

Prompt to continue
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm

from examexam.apis.conversation_and_router import Conversation, Router
from examexam.jinja_management import jinja_env

# Load environment variables (e.g., OPENAI_API_KEY)
dotenv.load_dotenv()

# ---- Logging setup (for developers) ----
# Keep logger.info/debug; print user-facing stuff with Rich Console.
logger = logging.getLogger(__name__)
if not logger.handlers:
    from rich.logging import RichHandler

    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[RichHandler(rich_tracebacks=True, markup=True, show_time=False, show_level=True)],
    )

console = Console()


# ---------- Core Logic ----------
def generate_study_guide(topic: str, model: str) -> str | None:
    """
    Calls an LLM to generate a study guide for a given topic.

    Args:
        topic: The topic for the study guide.
        model: The LLM to use.

    Returns:
        The generated study guide in Markdown format, or None on failure.
    """
    console.print(f"Generating study guide for topic: [bold cyan]{topic}[/] using model [italic]{model}[/]...")

    system_prompt = (
        "You are an expert tutor and research assistant. Your goal is to create a concise, "
        "well-structured study guide on a given topic. The guide should be in Markdown format. "
        "It must include a section with suggested search engine queries to help the user learn more."
    )
    conversation = Conversation(system=system_prompt)
    router = Router(conversation)

    try:
        template = jinja_env.get_template("study_guide.md.j2")
        user_prompt = template.render(topic=topic)
    except Exception as e:
        logger.error("Failed to load or render Jinja2 template 'study_guide.md.j2': %s", e)
        return None

    content = router.call(user_prompt, model)
    if not content:
        console.print("[bold red]Failed to generate study guide. The model returned no content.[/bold red]")
        return None

    return content


def save_and_display_guide(guide_content: str, topic: str) -> None:
    """
    Saves the study guide to a file and displays it in the terminal.
    """
    # Sanitize topic for filename
    safe_topic = "".join(c for c in topic if c.isalnum() or c in (" ", "_", "-")).rstrip()
    safe_topic = safe_topic.replace(" ", "_").lower()
    filename = f"{safe_topic}.md"

    # Create study_guide directory
    output_dir = Path("study_guide")
    output_dir.mkdir(exist_ok=True)
    file_path = output_dir / filename

    # Write to file
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(guide_content)
        console.print(Panel(f"Study guide saved to [bold green]{file_path}[/]", title="File Saved"))
    except OSError as e:
        console.print(Panel(f"[bold red]Error saving file: {e}[/]", title="Save Error"))
        return

    # Display in terminal
    console.rule(f"[bold]Study Guide: {topic}[/bold]")
    console.print(Markdown(guide_content))
    console.rule()


def generate_topic_research_now(topic: str, model: str = "openai") -> None:
    """
    Main execution function to generate, save, and display a study guide.
    """
    guide_content = generate_study_guide(topic, model)

    if guide_content:
        save_and_display_guide(guide_content, topic)
        if not os.environ.get("EXAMEXAM_NONINTERACTIVE"):
            Confirm.ask("Press Enter to exit", default=True, show_default=False)
    else:
        console.print("[bold red]Could not generate the study guide.[/bold red]")


if __name__ == "__main__":
    # Example direct run
    example_topic = "Python decorators"
    generate_topic_research_now(topic=example_topic, model="openai")
