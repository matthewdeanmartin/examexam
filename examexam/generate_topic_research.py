"""Call a bot to create study guide.

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
from typing import TYPE_CHECKING

import dotenv

from examexam.apis.conversation_and_router import Conversation, Router
from examexam.jinja_management import jinja_env

if TYPE_CHECKING:
    from examexam.ui_protocol import FrontendUI

# Load environment variables (e.g., OPENAI_API_KEY)
dotenv.load_dotenv()

# ---- Logging setup (for developers) ----
# Keep logger.info/debug; print user-facing stuff with FrontendUI.
logger = logging.getLogger(__name__)
if not logger.handlers:
    from rich.logging import RichHandler

    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[RichHandler(rich_tracebacks=True, markup=True, show_time=False, show_level=True)],
    )


# ---------- Core Logic ----------
def generate_study_guide(topic: str, model: str, ui: FrontendUI | None = None) -> str | None:
    """Calls an LLM to generate a study guide for a given topic.

    Args:
        topic: The topic for the study guide.
        model: The LLM to use.
        ui: Optional frontend UI for user-facing output.

    Returns:
        The generated study guide in Markdown format, or None on failure.
    """
    if ui:
        ui.show_message(f"Generating study guide for topic: {topic} using model {model}...")

    system_prompt = "You are an expert tutor and research assistant. Your goal is to create a concise, well-structured study guide on a given topic. The guide should be in Markdown format. It must include a section with suggested search engine queries to help the user learn more."
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
        if ui:
            ui.show_error("Failed to generate study guide. The model returned no content.")
        return None

    return content


def save_and_display_guide(guide_content: str, topic: str, ui: FrontendUI) -> None:
    """Saves the study guide to a file and displays it."""
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
        ui.show_panel(f"Study guide saved to {file_path}", title="File Saved")
    except OSError as e:
        ui.show_panel(f"Error saving file: {e}", title="Save Error", style="red")
        return

    # Display in terminal
    ui.show_rule(f"Study Guide: {topic}")
    ui.show_markdown(guide_content)
    ui.show_rule()


def generate_topic_research_now(topic: str, model: str = "openai", ui: FrontendUI | None = None) -> None:
    """Main execution function to generate, save, and display a study guide."""
    # Default to Rich CLI if no UI provided
    if ui is None:
        from examexam.frontends.rich_ui import RichUI

        ui = RichUI()

    guide_content = generate_study_guide(topic, model, ui)

    if guide_content:
        save_and_display_guide(guide_content, topic, ui)
        if not os.environ.get("EXAMEXAM_NONINTERACTIVE"):
            ui.confirm("Press Enter to exit", default=True)
    else:
        ui.show_error("Could not generate the study guide.")


if __name__ == "__main__":
    # Example direct run
    example_topic = "Python decorators"
    generate_topic_research_now(topic=example_topic, model="openai")
