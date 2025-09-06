# utilities.py
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any

from examexam.apis.types import Conversation, FailureToHaltError

LOGGER = logging.getLogger(__name__)


# --- Decorators ---


def call_limit(limit: int) -> Callable:
    """
    Decorator factory to limit the number of times a function can be called.
    """

    def decorator(func: Callable) -> Callable:
        func.call_count = 0  # type: ignore

        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            func.call_count += 1  # type: ignore
            name = func.__name__
            LOGGER.info(f"{name} called {func.call_count} times")
            if func.call_count > limit:
                raise FailureToHaltError(f"{name} has been called more than {limit} times")
            return func(*args, **kwargs)

        return wrapper

    return decorator


def log_duration(func: Callable) -> Callable:
    """Decorator to log the execution time of a function."""

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        LOGGER.info(f"Function {func.__name__} executed in {end_time - start_time:.4f} seconds.")
        return result

    return wrapper


# --- Helper Functions ---


def load_env() -> None:
    """
    Loads environment variables from a .env file.
    Placeholder for dotenv.load_dotenv().
    """
    # In a real app, you would use:
    # from dotenv import load_dotenv
    # load_dotenv()


def format_conversation_to_markdown(
    conversation: list[dict[str, str]], user_label: str = "User", assistant_label: str = "Assistant"
) -> str:
    """
    Formats a conversation history into a Markdown string.
    """
    markdown_lines = []
    for message in conversation:
        role = message.get("role", "").capitalize()
        content = message.get("content", "")

        label_map = {
            "User": user_label,
            "Assistant": assistant_label,
            "Examexam": "LLM Build Error Message",
            "System": "System",
        }
        label = label_map.get(role, role)

        if content is None:
            content = f"**** {label} returned None, maybe API failed ****"
        elif not content.strip():
            content = f"**** {label} returned whitespace ****"

        markdown_lines.append(f"## {label}")
        markdown_lines.append("```")
        markdown_lines.append(str(content))
        markdown_lines.append("```")
    return "\n".join(markdown_lines)


def log_conversation_to_file(conversation: Conversation, model_name: str, request: str) -> None:
    """Logs the full conversation history to a timestamped markdown file."""
    log_dir = Path("conversations")
    try:
        log_dir.mkdir(exist_ok=True)
    except OSError as e:
        LOGGER.error(f"Could not create conversations directory: {e}")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    # Sanitize model name for filename
    safe_model_name = model_name.replace(":", "_").replace("/", "_")
    filename = log_dir / f"{timestamp}_{safe_model_name}.md"

    # Create header for the markdown file
    header = "# Conversation Log\n\n"
    header += f"**Model:** `{model_name}`\n"
    header += f"**Timestamp:** `{datetime.now().isoformat()}`\n\n---\n\n"

    # Format the conversation
    markdown_content = format_conversation_to_markdown(conversation.conversation)

    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(header)
            f.write(markdown_content)
        LOGGER.debug(f"Conversation logged to {filename}")
    except OSError as e:
        LOGGER.error(f"Could not write to conversation log file {filename}: {e}")
