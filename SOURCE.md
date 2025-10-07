## Tree for examexam
```
├── apis/
│   ├── conversation_and_router.py
│   ├── third_party_apis.py
│   ├── types.py
│   └── utilities.py
├── config.py
├── constants.py
├── convert_to_pretty.py
├── find_the_toml.py
├── generate_questions.py
├── generate_study_plan.py
├── generate_topic_research.py
├── jinja_management.py
├── prompts/
│   ├── answer_question.md.j2
│   ├── evaluate_question.md.j2
│   ├── generate.md.j2
│   └── study_guide.md.j2
├── take_exam.py
├── validate_questions.py
└── __main__.py
```

## File: config.py
```python
"""
Manages configuration for the Examexam application.

This module handles loading settings from a TOML file, allowing environment
variables to override those settings, and provides a centralized, test-friendly
way to access configuration values.

Configuration Precedence:
1. Environment Variables (e.g., EXAMEXAM_GENERAL_DEFAULT_N)
2. Values from the TOML configuration file (e.g., examexam.toml)
3. Hardcoded default values in this module.

The configuration can be reset in memory, which is particularly useful for testing.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import rtoml as toml

# --- Constants ---
DEFAULT_CONFIG_FILENAME = "examexam.toml"
ENV_PREFIX = "EXAMEXAM_"

logger = logging.getLogger(__name__)

# --- Default Configuration ---
# This dictionary represents the structure and default values for the config.
DEFAULT_CONFIG = {
    "general": {
        "default_n": 5,
        "preferred_cheap_model": "openai",  # Corresponds to a key in FRONTIER_MODELS
        "preferred_frontier_model": "anthropic",
        "use_frontier_model": False,
        "override_model": "",  # If set, this model is used for all commands
    },
    "generate": {
        # "toc_file": "path/to/your/toc.txt",
        # "output_file": "path/to/your/output.toml",
        # "n": 5,
        # "model": "openai"
    },
    "validate": {
        # "question_file": "path/to/your/questions.toml",
        # "model": "openai"
    },
    "convert": {
        # "input_file": "path/to/your/questions.toml",
        # "output_base_name": "my-exam"
    },
    "research": {
        # "topic": "your-topic",
        # "model": "openai"
    },
    "study-plan": {
        # "toc_file": "path/to/your/toc.txt",
        # "model": "openai"
    },
}

# --- Default TOML content for initialization ---
DEFAULT_TOML_CONTENT = """
# Examexam Configuration File
# Settings in this file can be overridden by environment variables
# (e.g., EXAMEXAM_GENERAL_DEFAULT_N=10).

[general]
# Default number of questions to generate per topic.
# default_n = 5

# Preferred models for generation and validation.
# These keys should correspond to models available in the application.
# preferred_cheap_model = "openai"
# preferred_frontier_model = "anthropic"

# Set to true to default to using the 'preferred_frontier_model'.
# use_frontier_model = false

# If set, this model will be used for all commands, ignoring other model settings.
# override_model = ""


# --- Command-Specific Overrides ---
# You can provide default arguments for each command here.
# When running a command, these values will be used if the
# corresponding command-line argument is not provided.

[generate]
# toc_file = "path/to/your/toc.txt"
# output_file = "path/to/your/output.toml"
# n = 5
# model = "openai"

[validate]
# question_file = "path/to/your/questions.toml"
# model = "openai"

[convert]
# input_file = "path/to/your/questions.toml"
# output_base_name = "my-exam"

[research]
# topic = "your-topic"
# model = "openai"

[study-plan]
# toc_file = "path/to/your/toc.txt"
# model = "openai"
"""


class Config:
    """A singleton class to manage application configuration."""

    _instance: Config | None = None
    _config_data: dict[str, Any]
    _config_path: Path

    def __new__(cls) -> Config:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.reset()  # Initialize on first creation
        return cls._instance

    def load(self, config_path: str | Path | None = None) -> None:
        """
        Loads configuration from a TOML file and environment variables.

        Args:
            config_path: Path to the TOML configuration file. Defaults to
                         'examexam.toml' in the current directory.
        """
        self._config_path = Path(config_path or DEFAULT_CONFIG_FILENAME)
        self._config_data = self._deep_copy(DEFAULT_CONFIG)

        # Load from TOML file if it exists
        if self._config_path.exists():
            try:
                with self._config_path.open("r", encoding="utf-8") as f:
                    toml_config = toml.load(f)
                self._merge_configs(self._config_data, toml_config)
                logger.debug(f"Loaded config from {self._config_path}")
            except Exception as e:
                logger.error(f"Error reading config file {self._config_path}: {e}")
        else:
            logger.debug(
                f"Config file not found at {self._config_path}. Using defaults."
            )

        # Override with environment variables
        self._load_from_env()

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Retrieves a configuration value using a dot-separated key path.

        Example: config.get("general.default_n")
        """
        keys = key_path.split(".")
        value = self._config_data
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def reset(self) -> None:
        """Resets the configuration to its initial state. Useful for tests."""
        self._config_data = {}
        self._config_path = Path(DEFAULT_CONFIG_FILENAME)
        # We don't load here automatically to allow tests to point to a new file.
        # A manual `load()` is required after `reset()`.
        logger.debug("Configuration has been reset.")

    def _load_from_env(self) -> None:
        """Overrides config values with environment variables."""
        for section, settings in self._config_data.items():
            if isinstance(settings, dict):
                for key, value in settings.items():
                    env_var_name = f"{ENV_PREFIX}{section.upper()}_{key.upper()}"
                    env_value = os.environ.get(env_var_name)
                    if env_value is not None:
                        # Attempt to cast env var to the original type
                        original_type = type(value)
                        try:
                            if isinstance(original_type, bool):
                                casted_value = env_value.lower() in ("true", "1", "yes")
                            else:
                                casted_value = original_type(env_value)
                            self._config_data[section][key] = casted_value
                            logger.debug(
                                f"Overrode '{section}.{key}' with env var '{env_var_name}'."
                            )
                        except (ValueError, TypeError) as e:
                            logger.warning(
                                f"Could not cast env var {env_var_name}='{env_value}' to type {original_type}: {e}"
                            )

    def _merge_configs(self, base: dict[str, Any], new: dict[str, Any]) -> None:
        """Recursively merges the 'new' config dict into the 'base' dict."""
        for key, value in new.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_configs(base[key], value)
            else:
                if (
                    key in base
                    and isinstance(base[key], dict)
                    and not isinstance(value, dict)
                ):
                    logger.warning(
                        f"Config conflict: Section '{key}' cannot be overridden by a non-section value."
                    )
                else:
                    base[key] = value

    def _deep_copy(self, d: dict[str, Any]) -> dict[str, Any]:
        """Performs a simple deep copy for nested dictionaries."""
        new_dict = {}
        for key, value in d.items():
            if isinstance(value, dict):
                new_dict[key] = self._deep_copy(value)
            else:
                new_dict[key] = value
        return new_dict


def create_default_config_if_not_exists(
    filename: str = DEFAULT_CONFIG_FILENAME,
) -> bool:
    """
    Creates a default 'examexam.toml' file if it doesn't already exist.

    Args:
        filename: The name of the config file to create.

    Returns:
        True if the file was created, False if it already existed.
    """
    config_path = Path(filename)
    if config_path.exists():
        return False

    try:
        with config_path.open("w", encoding="utf-8") as f:
            f.write(DEFAULT_TOML_CONTENT)
        logger.info(f"Created default configuration file at: ./{filename}")
        return True
    except OSError as e:
        logger.error(f"Failed to create default configuration file: {e}")
        return False


# --- Global Instance ---
# Import this instance throughout the application to access config.
config = Config()
# Load config immediately on module import.
config.load()


def reset_for_testing(config_path_override: Path | None = None) -> Config:
    """
    Resets the singleton config instance. For testing purposes only.
    Allows specifying a direct path to a config file.
    """
    config.load(config_path_override)
    return config
```
## File: constants.py
```python
BAD_QUESTION_TEXT = "This is a bad question and is not answerable as posed."
```
## File: convert_to_pretty.py
```python
from __future__ import annotations

from typing import Any

import rtoml as toml
from markdown import markdown


def read_toml_file(file_path: str) -> list[dict[str, Any]]:
    """Reads a TOML file and returns the list of questions."""
    with open(file_path, encoding="utf-8") as file:
        data = toml.load(file)
    return data.get("questions", [])


def generate_markdown(questions: list[dict[str, Any]]) -> str:
    """
    Generates a Markdown string from a list of questions using Schema A.

    Schema A structure:
    [[questions]]
    id = "..."
    question = "..."
    [[questions.options]]
    text = "Some answer"
    explanation = "Why it is right/wrong"
    is_correct = true
    """
    markdown_content = ""
    for question in questions:
        markdown_content += f"### Question {question['id']}: {question['question']}\n\n"

        # Display all options
        markdown_content += "#### Options:\n"
        for option in question.get("options", []):
            markdown_content += f"- {option.get('text', 'N/A')}\n"

        # Find and display the correct answers by checking the 'is_correct' flag
        markdown_content += "\n#### Correct Answers:\n"
        correct_answers = [
            opt.get("text")
            for opt in question.get("options", [])
            if opt.get("is_correct")
        ]
        if not correct_answers:
            markdown_content += "- *No correct answer marked in source file.*\n"
        else:
            for answer in correct_answers:
                markdown_content += f"- {answer}\n"

        # Display the explanation for each option
        markdown_content += "\n#### Explanation:\n"
        for option in question.get("options", []):
            status = "Correct" if option.get("is_correct") else "Incorrect"
            explanation = option.get("explanation", "No explanation provided.")
            option_text = option.get("text", "N/A")
            markdown_content += f"- **{option_text}**: {explanation} *({status})*\n"

        markdown_content += "\n---\n\n"
    return markdown_content


def convert_markdown_to_html(markdown_content: str) -> str:
    """Converts a Markdown string to an HTML string."""
    html_content = markdown(markdown_content)
    return html_content


def write_to_file(content: str, file_path: str) -> None:
    """Writes content to a specified file."""
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)


def run(toml_file_path: str, markdown_file_path: str, html_file_path: str) -> None:
    """
    Main function to run the conversion process.
    Reads a TOML file and writes Markdown and HTML outputs.
    """
    questions = read_toml_file(toml_file_path)

    markdown_content = generate_markdown(questions)
    write_to_file(markdown_content, markdown_file_path)

    html_content = convert_markdown_to_html(markdown_content)
    write_to_file(html_content, html_file_path)

    print(f"Successfully created '{markdown_file_path}' and '{html_file_path}'.")
```
## File: find_the_toml.py
```python
from __future__ import annotations

import logging
import re
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

try:
    import rtoml as _toml  # fast, lenient
except Exception:  # pragma: no cover
    import tomllib as _toml  # stdlib (py3.11+), stricter


def _try_parse_toml(text: str) -> Optional[dict]:
    """Parse TOML; return dict or None."""
    try:
        return _toml.loads(text)
    except Exception as e:
        logger.debug("TOML parse failed: %s", e)
        return None


def _maybe_fix_quotes(s: str) -> str:
    """Replace common smart quotes with ASCII quotes for a second-chance parse."""
    return (
        s.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )


def _valid_schema(data: dict) -> bool:
    """
    Validate expected schema:

    [[questions]]
    question = "..."
    [[questions.options]]
    text = "..."
    explanation = "..."
    is_correct = true/false
    """
    if not isinstance(data, dict):
        return False
    questions = data.get("questions")
    if not isinstance(questions, list) or not questions:
        return False

    for q in questions:
        if not isinstance(q, dict):
            return False
        if "question" not in q or not isinstance(q["question"], str):
            return False

        options = q.get("options")
        if not isinstance(options, list) or not options:
            return False

        saw_true = False
        for opt in options:
            if not isinstance(opt, dict):
                return False
            if not isinstance(opt.get("text"), str):
                return False
            if not isinstance(opt.get("explanation"), str):
                return False
            if not isinstance(opt.get("is_correct"), bool):
                return False
            if opt["is_correct"]:
                saw_true = True
        # At least one correct per question is a reasonable sanity check
        if not saw_true:
            return False

    return True


_FENCE_RE = re.compile(
    r"(?ms)^(?:`{3,}|~{3,})\s*([A-Za-z0-9_-]+)?\s*\n(.*?)\n(?:`{3,}|~{3,})\s*$"
)

_TOML_FENCE_RE = re.compile(
    r"(?ms)^(?:`{3,}|~{3,})\s*(toml)\s*\n(.*?)\n(?:`{3,}|~{3,})\s*$",
    re.IGNORECASE,
)

# Any fenced block (not just TOML) – we will attempt parsing if content looks plausible
_ANY_FENCE_FINDALL_RE = re.compile(
    r"(?ms)^(?:`{3,}|~{3,})\s*([A-Za-z0-9_-]+)?\s*\n(.*?)\n(?:`{3,}|~{3,})\s*"
)


# Heuristic for TOML-ish line
def _looks_tomlish(line: str) -> bool:
    line = line.strip()
    if not line:
        return True
    if line.startswith("#"):
        return True
    if line.startswith("[[") and line.endswith("]]"):
        return True
    if line.startswith("[") and line.endswith("]"):
        return True
    # key = value (very loose)
    if "=" in line:
        left, _, right = line.partition("=")
        left = left.strip()
        right = right.strip()
        if left and re.match(r"^[A-Za-z0-9_.-]+$", left):
            # value starts like TOML string/bool/number/array/object
            if (
                right.startswith(('"', "'", "[", "{"))
                or right.lower() in {"true", "false"}
                or re.match(r"^[0-9\.\-]+", right)
            ):
                return True
    return False


def _gather_toml_candidates(markdown: str) -> list[str]:
    """
    Collect possible TOML snippets:
      1) ```toml fenced``` blocks
      2) any fenced blocks whose contents parse as TOML
      3) unfenced regions starting at '[[questions]]' and extending while lines look TOML-ish
    """
    candidates: list[str] = []

    # 1) Explicit TOML fences
    for m in _TOML_FENCE_RE.finditer(markdown):
        content = m.group(2)
        candidates.append(content)

    # 2) Any fence content that *might* be TOML
    for m in _ANY_FENCE_FINDALL_RE.finditer(markdown):
        lang = (m.group(1) or "").lower()
        content = m.group(2)
        # Skip if already captured as explicit TOML
        if lang == "toml":
            continue
        # Try as-is
        candidates.append(content)

    # 3) Unfenced heuristic extraction from first [[questions]] occurrence(s)
    lines = markdown.splitlines()
    indices = [
        i for i, ln in enumerate(lines) if "[[questions]]" in ln.replace(" ", "")
    ]
    for start in indices:
        # Expand upwards slightly if we started in the middle of a TOML block
        s = start
        while s > 0 and _looks_tomlish(lines[s - 1]):
            # stop if we would cross a fence marker (we don't want to merge across markdown code fences)
            if lines[s - 1].lstrip().startswith("```") or lines[
                s - 1
            ].lstrip().startswith("~~~"):
                break
            s -= 1
        # Expand downwards while TOML-ish and not hitting a fence close/open
        e = start
        fence_depth = 0
        while e < len(lines):
            ln = lines[e]
            stripped = ln.lstrip()
            if stripped.startswith("```") or stripped.startswith("~~~"):
                # Don't cross markdown fences in unfenced scan
                break
            if not _looks_tomlish(ln):
                # allow one non-tomlish line as a gap; stop after a second
                # (helps include trailing blank/comment)
                if e + 1 < len(lines) and _looks_tomlish(lines[e + 1]):
                    e += 1
                    continue
                else:
                    break
            e += 1
        block = "\n".join(lines[s:e]).strip()
        if block and "[[questions]]" in block:
            candidates.append(block)

    # De-dup while preserving order
    seen = set()
    uniq: list[str] = []
    for c in candidates:
        key = c.strip()
        if key and key not in seen:
            uniq.append(key)
            seen.add(key)
    return uniq


def _first_valid_toml(candidates: Iterable[str]) -> Optional[str]:
    """
    Try to parse & validate candidates in order, with a second-chance smart-quote fix.
    Return the original candidate text (not re-serialized) on success.
    """
    for cand in candidates:
        data = _try_parse_toml(cand)
        if data is None:
            fixed = _maybe_fix_quotes(cand)
            if fixed != cand:
                data = _try_parse_toml(fixed)
                if data is not None and _valid_schema(data):
                    logger.info("Recovered TOML by normalizing quotes.")
                    return fixed
        if data is not None and _valid_schema(data):
            return cand
    return None


def extract_questions_toml(markdown_content: str) -> str | None:
    """
    Aggressively extract SMBC-quiz TOML from arbitrary markdown/plaintext.

    Rules:
      - If it can load as TOML and matches the schema, it's TOML.
      - If fenced with a TOML language tag, strip only the outer fence.
      - If fenced with some other/no language, still try to parse.
      - If unfenced, scan from [[questions]] and include TOML-ish lines.
      - Never touch backticks *inside* TOML strings.

    Returns:
      The TOML snippet (without outer code fences), or None if not found/valid.
    """
    if not markdown_content:
        logger.debug("No content; skipping TOML extract.")
        return None

    candidates = _gather_toml_candidates(markdown_content)
    if not candidates:
        logger.debug("No TOML candidates found.")
        return None

    result = _first_valid_toml(candidates)
    if result is None:
        logger.debug(
            "No valid TOML matched expected schema among %d candidates.",
            len(candidates),
        )
    else:
        logger.info("TOML content found and validated.")
    return result
```
## File: generate_questions.py
```python
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

from examexam.find_the_toml import extract_toml
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
        handlers=[
            RichHandler(
                rich_tracebacks=True, markup=True, show_time=False, show_level=True
            )
        ],
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
    logger.debug(
        "Creating new Conversation with system prompt length=%d", len(system_prompt)
    )
    conversation = Conversation(system=system_prompt)
    return conversation


class FatalLLMError(Exception):
    """Errors that should not be retried (e.g., missing API key)."""


def _fatal_if_misconfigured(model: str) -> None:
    """Raise FatalLLMError for obviously fatal misconfigurations before calling LLM."""
    # Allow a stub model name used for tests.
    if model.lower() not in {"fakebot", "none", "noop"} and not os.getenv(
        "OPENAI_API_KEY"
    ):
        raise FatalLLMError(
            "OPENAI_API_KEY is not set. Set the environment variable or pass api_key to the client."
        )


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


# def extract_toml(markdown_content: str) -> str | None:
#     """Extract TOML fenced block from markdown content."""
#     if not markdown_content:
#         logger.debug("No content returned from router.call; skipping TOML extract")
#         return None
#     match = re.search(r"```toml\n(.*?)\n```", markdown_content, re.DOTALL)
#     if match:
#         logger.info("TOML content found in response.")
#         return match.group(1)
#     logger.debug("TOML fenced block not found in content (len=%d)", len(markdown_content))
#     return None


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
            logger.debug(
                "router.call returned content length=%d in %.2fs",
                len(content or ""),
                duration,
            )
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
                "Attempt %d: TOML not found; %s",
                attempts,
                "retrying" if attempts <= max_retries else "giving up",
            )
            if attempts > max_retries:
                return None
            sleep(retry_delay_seconds)
            continue

        # Try to parse TOML
        try:
            parsed = toml.loads(toml_content)
            logger.debug(
                "Parsed TOML successfully with %d questions",
                len(parsed.get("questions", [])),
            )
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
        console.print(
            Panel.fit(f"[red]TOC file not found:[/] {toc_file}", title="Error")
        )
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
                progress.update(
                    topic_task,
                    description=f"{idx}/{total_topics} {service} [red](failed)[/]",
                )
                progress.advance(topic_task)
                progress.advance(overall_task)
                if absolute_failures > 3:
                    print(
                        "Three times bot absolutely failed to generate any proper questions"
                    )
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
```
## File: generate_study_plan.py
```python
from __future__ import annotations

import logging
import os
from pathlib import Path
from time import perf_counter

import dotenv
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

from examexam.generate_topic_research import generate_study_guide

# Load environment variables
dotenv.load_dotenv()

# ---- Logging setup ----
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            RichHandler(
                rich_tracebacks=True, markup=True, show_time=False, show_level=True
            )
        ],
    )

console = Console()


def generate_study_plan_now(toc_file: str, model: str = "openai") -> None:
    """
    Generates a consolidated study guide for all topics in a TOC file.
    """
    toc_path = Path(toc_file)
    if not toc_path.exists():
        console.print(
            Panel.fit(f"[red]TOC file not found:[/] {toc_file}", title="Error")
        )
        return

    with toc_path.open(encoding="utf-8") as f:
        topics = [line.strip() for line in f if line.strip()]

    total_topics = len(topics)
    if total_topics == 0:
        console.print(Panel.fit("[yellow]TOC file is empty.[/]", title="Nothing to do"))
        return

    console.rule("[bold]Study Plan Generation[/bold]")
    console.print(
        f"Generating study guides for [bold]{total_topics}[/] topics using model [italic]{model}[/]…\n"
    )

    all_guides_content = [f"# Study Plan for {toc_path.stem}\n\n"]
    failures = 0

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

    with progress:
        overall_task = progress.add_task("Overall", total=total_topics)

        for idx, topic in enumerate(topics, start=1):
            topic_task_desc = (
                f"{idx}/{total_topics} {topic[:40]}{'...' if len(topic) > 40 else ''}"
            )
            topic_task = progress.add_task(topic_task_desc, total=1)

            t0 = perf_counter()
            guide_content = generate_study_guide(topic, model)
            dt = perf_counter() - t0

            if guide_content:
                all_guides_content.append(
                    f"## Topic: {topic}\n\n{guide_content}\n\n---\n\n"
                )
                progress.update(
                    topic_task,
                    description=f"{topic_task_desc} [green](ok in {dt:.2f}s)[/]",
                )
            else:
                failures += 1
                progress.update(
                    topic_task, description=f"{topic_task_desc} [red](failed)[/]"
                )

            progress.advance(topic_task)
            progress.advance(overall_task)

    console.rule()

    if not all_guides_content or len(all_guides_content) == 1:
        console.print(
            Panel.fit(
                "[bold red]Failed to generate any study guides.[/]",
                title="Complete Failure",
            )
        )
        return

    # Save the consolidated file
    output_dir = Path("study_guide")
    output_dir.mkdir(exist_ok=True)
    output_filename = f"{toc_path.stem}_study_plan.md"
    output_path = output_dir / output_filename

    try:
        with output_path.open("w", encoding="utf-8") as f:
            f.write("".join(all_guides_content))

        summary_message = (
            f"Successfully generated study guides for [bold green]{total_topics - failures}[/] topics. "
            f"Saved to:\n[bold cyan]{output_path}[/]"
        )
        if failures > 0:
            summary_message += f"\n[bold red]{failures}[/] topics failed."

        console.print(Panel.fit(summary_message, title="Summary"))

    except OSError as e:
        console.print(Panel(f"[bold red]Error saving file: {e}[/]", title="Save Error"))


if __name__ == "__main__":
    # Example of how to run this directly
    # You would need an 'example_toc.txt' file for this to work
    generate_study_plan_now(toc_file="example_toc.txt", model="openai")
```
## File: generate_topic_research.py
```python
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
        handlers=[
            RichHandler(
                rich_tracebacks=True, markup=True, show_time=False, show_level=True
            )
        ],
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
    console.print(
        f"Generating study guide for topic: [bold cyan]{topic}[/] using model [italic]{model}[/]..."
    )

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
        logger.error(
            "Failed to load or render Jinja2 template 'study_guide.md.j2': %s", e
        )
        return None

    content = router.call(user_prompt, model)
    if not content:
        console.print(
            "[bold red]Failed to generate study guide. The model returned no content.[/bold red]"
        )
        return None

    return content


def save_and_display_guide(guide_content: str, topic: str) -> None:
    """
    Saves the study guide to a file and displays it in the terminal.
    """
    # Sanitize topic for filename
    safe_topic = "".join(
        c for c in topic if c.isalnum() or c in (" ", "_", "-")
    ).rstrip()
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
        console.print(
            Panel(
                f"Study guide saved to [bold green]{file_path}[/]", title="File Saved"
            )
        )
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
```
## File: jinja_management.py
```python
from __future__ import annotations

import hashlib
import importlib.resources
import logging
import os
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, PackageLoader
from rich.logging import RichHandler

# ---- Logging setup (for developers) ----
# Keep logger.info/debug; print user-facing stuff with Rich Console.
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            RichHandler(
                rich_tracebacks=True, markup=True, show_time=False, show_level=True
            )
        ],
    )

# --- Constants ---
CUSTOM_PROMPTS_DIR = Path("./prompts")
HASHES_FILENAME = "hashes.txt"


# --- Hashing Utilities ---
def _calculate_hash(content: bytes) -> str:
    """Calculates the SHA256 hash of file content."""
    return hashlib.sha256(content).hexdigest()


def _read_hashes_file(path: Path) -> dict[str, str]:
    """Reads a hashes.txt file and returns a dictionary of filename:hash."""
    hashes = {}
    if not path.is_file():
        return hashes
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    name, _, hash_val = line.strip().partition(":")
                    hashes[name.strip()] = hash_val.strip()
    except OSError as e:
        logger.warning("Could not read hashes file at %s: %s", path, e)
    return hashes


def _write_hashes_file(path: Path, hashes: dict[str, str]) -> None:
    """Writes a dictionary of hashes to a hashes.txt file."""
    try:
        with path.open("w", encoding="utf-8") as f:
            for name, hash_val in sorted(hashes.items()):
                f.write(f"{name}:{hash_val}\n")
    except OSError as e:
        logger.error("Could not write hashes file to %s: %s", path, e)


# --- Template Deployment ---
def deploy_for_customization(target_dir: Path, force: bool = False) -> None:
    """
    Copies the package's built-in templates to a target directory for user customization.

    This function creates a 'prompts' folder inside the specified `target_dir`.
    It also creates a `hashes.txt` file to track the original state of the
    deployed templates. On subsequent runs, it will safely update unmodified
    templates and add new ones, but will not overwrite user-modified templates
    unless the `force` flag is set.

    Args:
        target_dir: The directory where the 'prompts' folder will be created.
        force: If True, overwrite all existing files, including those modified by the user.
    """
    dest_prompts_path = target_dir / "prompts"
    dest_hashes_path = dest_prompts_path / HASHES_FILENAME
    new_hashes = {}

    try:
        dest_prompts_path.mkdir(parents=True, exist_ok=True)
        logger.info("Ensured prompts directory exists at: %s", dest_prompts_path)
    except OSError as e:
        logger.error("Fatal: Could not create target directory for prompts: %s", e)
        return

    original_hashes = _read_hashes_file(dest_hashes_path)
    source_files = importlib.resources.files("examexam") / "prompts"

    for src_file in source_files.iterdir():
        if not src_file.is_file() or not src_file.name.endswith(".j2"):
            continue

        dest_file_path = dest_prompts_path / src_file.name
        try:
            src_content = src_file.read_bytes()
            src_hash = _calculate_hash(src_content)
            new_hashes[src_file.name] = src_hash

            should_write = True
            if dest_file_path.exists() and not force:
                original_hash = original_hashes.get(src_file.name)
                # If we have a record of the original hash, check if the user has modified the file.
                if original_hash:
                    current_dest_hash = _calculate_hash(dest_file_path.read_bytes())
                    if current_dest_hash != original_hash:
                        logger.warning(
                            "Skipping modified template '%s'. Use --force to overwrite.",
                            src_file.name,
                        )
                        should_write = False
                # If no original hash, we assume it's user-managed and don't touch it without force.
                else:
                    logger.warning(
                        "Skipping template '%s' (no hash record). Use --force to overwrite.",
                        src_file.name,
                    )
                    should_write = False

            if should_write:
                dest_file_path.write_bytes(src_content)
                logger.debug("Deployed template: %s", dest_file_path)

        except Exception as e:
            logger.error(
                "Failed to process or deploy template '%s': %s", src_file.name, e
            )

    _write_hashes_file(dest_hashes_path, new_hashes)
    logger.info("Template deployment complete. Hashes updated in %s", dest_hashes_path)


# ---------- Jinja2 Template Loading (Updated) ----------
def get_jinja_env() -> Environment:
    """
    Initializes and returns a Jinja2 Environment with a prioritized loading strategy.

    The search order for the 'prompts' directory is:
    1. User-customized: A 'prompts' directory in the current working directory (`./prompts`).
    2. Development mode: A 'prompts' directory relative to the project's source root.
    3. Installed package: The 'prompts' directory bundled with the installed package.
    """
    # 1. Check for user-customized prompts in the current directory
    if CUSTOM_PROMPTS_DIR.is_dir():
        logger.debug(
            "Loading Jinja2 templates from user-customized directory: %s",
            CUSTOM_PROMPTS_DIR.resolve(),
        )
        loader = FileSystemLoader(CUSTOM_PROMPTS_DIR)
        return Environment(loader=loader, autoescape=False)  # nosec

    # 2. Check for development mode prompts
    dev_prompts_path = Path(__file__).parent.parent / "prompts"
    if dev_prompts_path.is_dir():
        logger.debug(
            "Loading Jinja2 templates from development directory: %s", dev_prompts_path
        )
        loader = FileSystemLoader(dev_prompts_path)
        return Environment(loader=loader, autoescape=False)  # nosec

    # 3. Fallback to installed package prompts
    logger.debug("Loading Jinja2 templates from installed package 'examexam.prompts'")
    try:
        loader = PackageLoader("examexam", "prompts")
        return Environment(loader=loader, autoescape=False)  # nosec
    except ModuleNotFoundError:
        logger.error("Could not find the 'examexam' package to load templates.")
        raise


# Create a single environment instance to be used by the module
jinja_env = get_jinja_env()
```
## File: take_exam.py
```python
"""
Example toml


[[questions]]
question = "What is the primary purpose of Amazon Athena?"
id = "10fc5083-5528-4be1-a3cf-f377ae963dfc"

[[questions.options]]
text = "To perform ad-hoc querying on data stored in S3 using SQL."
explanation = "Amazon Athena allows users to run SQL queries directly on data in S3 without needing to manage any infrastructure. Correct."
is_correct = true

[[questions.options]]
text = "To manage relational databases on EC2."
explanation = "Amazon Athena is a serverless query service, and it does not manage databases on EC2. Incorrect."
is_correct = false
"""

from __future__ import annotations

import math
import os
import random
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import dotenv
import rtoml as toml
from rich.align import Align
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from scipy import stats

from examexam.constants import BAD_QUESTION_TEXT
from examexam.utils.secure_random import SecureRandom
from examexam.utils.toml_normalize import normalize_exam_for_toml

# Load environment variables (e.g., OPENAI_API_KEY)
dotenv.load_dotenv()

console = Console()

# ----------------- NEW: answer provider protocol & strategies -----------------


class AnswerProvider(Protocol):
    def __call__(
        self, question: dict[str, Any], options_list: list[dict[str, Any]]
    ) -> list[dict[str, Any]]: ...


MachineStrategy = Literal["oracle", "random", "first", "none"]


def build_machine_answer_provider(
    strategy: MachineStrategy = "oracle", *, seed: int | None = 42
) -> AnswerProvider:
    """Return a function that selects answers without user input.

    Strategies:
      - 'oracle': choose exactly the options with is_correct=True
      - 'random': choose a random valid set of size 'answer_count'
      - 'first': choose the first 'answer_count' options
      - 'none': choose an incorrect set on purpose, if possible
    """
    rng = SecureRandom(seed)  # nosec

    def provider(
        question: dict[str, Any], options_list: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        answer_count = sum(1 for o in question["options"] if o.get("is_correct"))
        if strategy == "oracle":
            return [o for o in options_list if o.get("is_correct")]
        if strategy == "first":
            # Skip the "bad question" sentinel; we never include it in machine mode
            return options_list[:answer_count]
        if strategy == "random":
            # sample from actual options (exclude bad-question sentinel)
            population = list(options_list)
            if answer_count <= 0:
                return []
            if answer_count >= len(population):
                return population
            picks = rng.sample(range(len(population)), k=answer_count)
            return [population[i] for i in picks]
        if strategy == "none":
            # Try to pick a *different* set than the correct one
            correct = {id(o) for o in options_list if o.get("is_correct")}
            population = list(range(len(options_list)))
            if not correct:
                # If there is no correct answer, pick one anyway (e.g., trick Q)
                return [options_list[0]] if options_list else []
            # Greedy: start from first 'answer_count' indices; ensure it differs
            attempt = population[:answer_count]
            if {id(options_list[i]) for i in attempt} == correct and len(
                population
            ) > answer_count:
                attempt[-1] = population[-1]
            return [options_list[i] for i in attempt]
        raise ValueError(f"Unknown strategy: {strategy!r}")

    return provider


# ----------------- existing helpers unchanged above this line -----------------


def load_questions(file_path: str) -> list[dict[str, Any]]:
    """Load questions from a file"""
    with open(file_path, encoding="utf-8") as file:
        data = toml.load(file)["questions"]
        return cast(list[dict[str, Any]], data)


def get_session_path(test_name: str) -> Path:
    """Get the session file path for a given test"""
    session_dir = Path(".session")
    session_dir.mkdir(exist_ok=True)
    return session_dir / f"{test_name}.toml"


def get_available_tests() -> list[str]:
    """Get list of available test files from /data/ folder"""
    data_dir = Path("data")
    if not data_dir.exists():
        console.print("[bold red]Error: /data/ folder not found![/bold red]")
        data_dir = Path(".")

    test_files = list(data_dir.glob("*.toml"))
    return [f.stem for f in test_files]


def select_test() -> str | None:
    """Let user select a test to take"""
    tests = get_available_tests()
    if not tests:
        console.print("[bold red]No test files found in /data/ folder![/bold red]")
        return None

    console.print("[bold blue]Available Tests:[/bold blue]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Number", style="dim", width=6)
    table.add_column("Test Name")

    for idx, test in enumerate(tests, 1):
        table.add_row(str(idx), test)

    console.print(table)

    while True:
        try:
            choice = Prompt.ask("Enter the test number", default="1")
            test_idx = int(choice) - 1
            if 0 <= test_idx < len(tests):
                return tests[test_idx]
            console.print("[bold red]Invalid choice. Please try again.[/bold red]")
        except ValueError:
            console.print("[bold red]Please enter a valid number.[/bold red]")


def check_resume_session(
    test_name: str,
) -> tuple[bool, list[dict[str, Any]] | None, datetime | None]:
    """Check if a session exists and ask if user wants to resume"""
    session_path = get_session_path(test_name)
    if not session_path.exists():
        return False, None, None

    try:
        with open(session_path, encoding="utf-8") as file:
            data = toml.load(file)
            session_data = data.get("questions", [])
            start_time = data.get("start_time")

        # Check if there's any progress
        completed = sum(1 for q in session_data if q.get("user_score") is not None)
        total = len(session_data)

        if completed == 0:
            return False, None, None

        console.print(
            f"[bold yellow]Found existing session for '{test_name}'[/bold yellow]"
        )
        console.print(f"Progress: {completed}/{total} questions completed")

        start_dt = None
        if start_time:
            try:
                start_dt = datetime.fromisoformat(start_time)
                elapsed = datetime.now() - start_dt
                console.print(f"Started: {humanize_timedelta(elapsed)} ago")
            except (ValueError, TypeError):
                # Invalid start_time format, will use current time as fallback
                console.print("Started: Unknown time ago")

        resume = Confirm.ask("Do you want to resume this session?")
        if resume:
            return True, session_data, start_dt
        # User wants to start fresh
        session_path.unlink()  # Delete old session
        return False, None, None

    except Exception as e:
        console.print(f"[bold red]Error reading session file: {e}[/bold red]")
        return False, None, None


def humanize_timedelta(td: timedelta) -> str:
    """Convert timedelta to human readable format"""
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes > 0:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if seconds > 0 or not parts:
        parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")

    return " ".join(parts)


def calculate_time_estimates(
    session: list[dict[str, Any]], start_time: datetime
) -> tuple[timedelta, timedelta | None]:
    """Calculate average time per question and estimated completion time, removing outliers"""
    completed_times = []

    for question in session:
        if "completion_time" in question and question.get("user_score") is not None:
            completion_dt = datetime.fromisoformat(question["completion_time"])
            question_start = datetime.fromisoformat(
                question.get("start_time", start_time.isoformat())
            )
            question_duration = completion_dt - question_start
            completed_times.append(question_duration.total_seconds())

    if len(completed_times) < 2:
        return timedelta(), None

    # Remove outliers (questions that took more than 3x the median)
    median_time = sorted(completed_times)[len(completed_times) // 2]
    filtered_times = [t for t in completed_times if t <= 3 * median_time]

    if not filtered_times:
        return timedelta(), None

    avg_seconds = sum(filtered_times) / len(filtered_times)
    avg_time_per_question = timedelta(seconds=avg_seconds)

    # Calculate remaining questions
    remaining = sum(1 for q in session if q.get("user_score") is None)
    estimated_time_left = (
        timedelta(seconds=avg_seconds * remaining) if remaining > 0 else None
    )

    return avg_time_per_question, estimated_time_left


def clear_screen() -> None:
    """Function to clear the terminal"""
    os.system("cls" if os.name == "nt" else "clear")  # nosec


def play_sound(_file: str) -> None:
    """Function to play sound effects"""
    # playsound(_file)


def find_select_pattern(input_string: str) -> str:
    """
    Finds the first occurrence of "(Select n)" in the input string where n is a number from 1 to 5.
    """
    match = re.search(r"\(Select [1-5]\)", input_string)
    return match.group(0) if match else ""


def is_valid(
    answer: str,
    option_count: int,
    answer_count: int,
    last_is_bad_question_flag: bool = True,
) -> tuple[bool, str]:
    if not answer:
        return False, "Please enter an answer."

    answers = answer.split(",")

    # Check if all answers are valid numbers
    for number in answers:
        try:
            int(number)
        except ValueError:
            return False, f"'{number}' is not a valid number."

    # Special case for bad question flag
    if (
        answer_count == 1
        and last_is_bad_question_flag
        and len(answers) == 1
        and int(answers[0]) == option_count
    ):
        return True, ""

    # Check bounds
    for number in answers:
        num = int(number)
        if num < 1 or num > option_count:
            return False, f"Answer {num} is out of range (1-{option_count})."

    # Check answer count
    if len(answers) != answer_count:
        return (
            False,
            f"Please select exactly {answer_count} answer{'s' if answer_count != 1 else ''}, you selected {len(answers)}.",
        )

    return True, ""


def ask_question(
    question: dict[str, Any], options_list: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    clear_screen()
    question_text = question["question"]

    pattern = find_select_pattern(question_text)
    answer_count = len(
        list(option for option in question["options"] if option.get("is_correct"))
    )

    if pattern:
        correct_select = f"(Select {answer_count})"
        if correct_select not in question_text:
            question_text = question_text.replace(pattern, correct_select)

    if "(Select" not in question_text:
        question_text = f"{question_text} (Select {answer_count})"

    if "(Select n)" in question_text:
        question_text = question_text.replace("(Select n)", f"(Select {answer_count})")

    question_panel = Align.center(Panel(Markdown(question_text)), vertical="middle")
    console.print(question_panel)

    table = Table(title="Options", style="green")
    table.add_column("Option Number", justify="center")
    table.add_column("Option Text", justify="left")

    for idx, option in enumerate(options_list, 1):
        table.add_row(str(idx), option["text"])

    table.add_row(str(len(options_list) + 1), BAD_QUESTION_TEXT)
    console.print(Align.center(table))

    answer = ""
    option_count = len(options_list) + 1
    while True:
        answer = console.input(
            "[bold yellow]Enter your answer(s) as a comma-separated list (e.g., 1,2): [/bold yellow]"
        )
        is_valid_answer, error_msg = is_valid(answer, option_count, answer_count)
        if is_valid_answer:
            break
        console.print(f"[bold red]{error_msg}[/bold red]")

    selected = [
        options_list[int(idx) - 1]
        for idx in answer.split(",")
        if idx.isdigit() and 1 <= int(idx) <= len(options_list)
    ]
    return selected


def calculate_confidence_interval(
    score: int, total: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Calculate confidence interval for population proportion"""
    if total == 0:
        return 0.0, 0.0

    p = score / total  # sample proportion
    z = stats.norm.ppf((1 + confidence) / 2)  # z-score for confidence level

    # Standard error
    se = math.sqrt(p * (1 - p) / total)

    # Margin of error
    me = z * se

    # Confidence interval
    lower = max(0, p - me)
    upper = min(1, p + me)

    return lower, upper


def display_results(
    score: float,
    total: float,
    start_time: datetime,
    session: list[dict[str, Any]] = None,
    withhold_judgement: bool = False,
) -> None:
    percent = (score / total) * 100
    passed = "Passed" if percent >= 70 else "Failed"

    # Calculate timing
    elapsed = datetime.now() - start_time

    # Calculate confidence interval
    lower, upper = calculate_confidence_interval(int(score), int(total))

    # Format timing info
    total_time_str = humanize_timedelta(elapsed)

    # Calculate time estimates with outlier removal
    if session:
        avg_time_per_question, estimated_time_left = calculate_time_estimates(
            session, start_time
        )
        avg_time_str = humanize_timedelta(avg_time_per_question)

        time_info = (
            f"Total Time: {total_time_str}\nAvg Time per Question: {avg_time_str}"
        )
        if estimated_time_left and not withhold_judgement:
            time_info += f"\nEstimated Time to Complete: {humanize_timedelta(estimated_time_left)}"
    else:
        # Fallback to simple calculation
        time_per_question = elapsed / total if total > 0 else timedelta()
        avg_time_str = humanize_timedelta(time_per_question)
        time_info = f"Total Time: {total_time_str}\nTime per Question: {avg_time_str}"

    # Format confidence interval
    confidence_str = f"{lower * 100:.1f}%-{upper * 100:.1f}%, 95% confidence interval"

    if withhold_judgement:
        judgement = ""
    else:
        judgement = f"\n[green]{passed}[/green]"

    result_text = (
        f"[bold yellow]Your Score: {score}/{total} ({percent:.2f}%){judgement}\n"
        f"{time_info}\n"
        f"Population Estimate: {confidence_str}[/bold yellow]"
    )

    console.print(
        Panel(
            result_text,
            title="Results",
            style="magenta",
        )
    )


def save_session_file(
    session_file: Path, state: list[dict[str, Any]], start_time: datetime
) -> None:
    with open(session_file, "w", encoding="utf-8") as file:
        data = {
            "questions": state,
            "start_time": start_time.isoformat(),
            "last_updated": datetime.now().isoformat(),
        }
        toml.dump(normalize_exam_for_toml(data), file)


def take_exam_now(
    question_file: str = None,
    *,
    machine: bool = False,
    strategy: MachineStrategy = "oracle",
    seed: int | None = 42,
    quiet: bool = False,
) -> None:
    """Main function to run the quiz (interactive by default, or machine mode if requested)."""
    if (machine and question_file) or (os.environ.get("EXAMEXAM_MACHINE_TAKES_EXAM")):
        _ = take_exam_machine(
            question_file,
            strategy=strategy,
            seed=seed,
            quiet=quiet,
            persist_session=True,
        )
        return

    if question_file:
        # Legacy API - use provided file path
        test_path = Path(question_file)
        test_name = test_path.stem
        session_path = get_session_path(test_name)

        # Check for existing session
        resume_session, session_data, session_start_time = check_resume_session(
            test_name
        )

        if resume_session and session_data:
            session = session_data
            questions = load_questions(question_file)
            start_time = (
                session_start_time or datetime.now()
            )  # Fallback to current time
        else:
            questions = load_questions(question_file)
            session = questions.copy()
            start_time = datetime.now()
            save_session_file(session_path, session, start_time)
    else:
        # New interactive API
        test_name = select_test()
        if not test_name:
            return

        if (Path("data") / f"{test_name}.toml").exists():
            test_file = Path("data") / f"{test_name}.toml"
        else:
            test_file = f"{test_name}.toml"

        session_path = get_session_path(test_name)

        # Check for existing session
        resume_session, session_data, session_start_time = check_resume_session(
            test_name
        )

        if resume_session and session_data:
            session = session_data
            questions = load_questions(str(test_file))
            start_time = (
                session_start_time or datetime.now()
            )  # Fallback to current time
        else:
            questions = load_questions(str(test_file))
            session = questions.copy()
            start_time = datetime.now()
            save_session_file(session_path, session, start_time)
    try:
        interactive_question_and_answer(questions, session, session_path, start_time)
        save_session_file(session_path, session, start_time)
    except KeyboardInterrupt:
        save_session_file(session_path, session, start_time)
        console.print("[bold red]Exiting the exam...[/bold red]")


# def interactive_question_and_answer(questions, session, session_path: Path, start_time: datetime):
#     score = 0
#     so_far = 0
#
#     # Count already completed questions
#     for question in session:
#         if question.get("user_score") == 1:
#             score += 1
#             so_far += 1
#
#     random.shuffle(questions)
#     for question in questions:
#         session_question = find_question(question, session)
#
#         if session_question.get("user_score") == 1:
#             continue
#
#         # Record when this question started
#         question_start_time = datetime.now()
#         session_question["start_time"] = question_start_time.isoformat()
#
#         options_list = list(question["options"])
#         random.shuffle(options_list)
#         try:
#             selected = ask_question(question, options_list)
#         except KeyboardInterrupt:
#             display_results(score, len(questions), start_time, session)
#             raise
#
#         # Record completion time
#         session_question["completion_time"] = datetime.now().isoformat()
#
#         correct = {option["text"] for option in options_list if option.get("is_correct", False)}
#         user_answers = {option["text"] for option in selected}
#
#         # Only show comparison if answers differ
#         if user_answers == correct:
#             console.print(
#                 Panel(
#                     "[bold green]✓ Correct![/bold green]",
#                     title="Answer Review",
#                     style="green",
#                 )
#             )
#         else:
#             console.print(
#                 Panel(
#                     f"[bold cyan]Correct Answer(s): {', '.join(correct)}\nYour Answer(s): {', '.join(user_answers)}[/bold cyan]",
#                     title="Answer Review",
#                     style="blue",
#                 )
#             )
#
#         # Create numbered explanations matching the original option order
#         colored_explanations = []
#         for idx, option in enumerate(options_list, 1):
#             if option.get("is_correct", False):
#                 colored_explanations.append(f"{idx}. [bold green]{option['explanation']}[/bold green]")
#             else:
#                 colored_explanations.append(f"{idx}. [bold red]{option['explanation']}[/bold red]")
#
#         console.print(Panel("\n".join(colored_explanations), title="Explanation"))
#
#         session_question["user_answers"] = list(user_answers)
#         if user_answers == correct:
#             play_sound("correct.mp3")
#             score += 1
#             session_question["user_score"] = 1
#         else:
#             console.print("[bold red]Incorrect.[/bold red]", style="bold red")
#             play_sound("incorrect.mp3")
#             session_question["user_score"] = 0
#
#         so_far += 1
#         display_results(score, so_far, start_time, session, withhold_judgement=True)
#
#         go_on = None
#         while go_on not in ("", "bad"):
#             go_on = console.input("[bold yellow]Press Enter to continue to the next question...[/bold yellow]")
#
#         if go_on == "bad":
#             session_question["defective"] = True
#             save_session_file(session_path, session, start_time)
#
#     clear_screen()
#     display_results(score, len(questions), start_time, session)
#     save_session_file(session_path, session, start_time)
#     return score


def find_question(
    question: dict[str, Any], session: list[dict[str, Any]]
) -> dict[str, Any]:
    session_question = {}
    for q in session:
        if q["id"] == question["id"]:
            session_question = q
            break
    return session_question


def ask_question_interactive(
    question: dict[str, Any], options_list: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    clear_screen()
    question_text = question["question"]

    pattern = find_select_pattern(question_text)
    answer_count = len([o for o in question["options"] if o.get("is_correct")])

    if pattern:
        correct_select = f"(Select {answer_count})"
        if correct_select not in question_text:
            question_text = question_text.replace(pattern, correct_select)

    if "(Select" not in question_text:
        question_text = f"{question_text} (Select {answer_count})"

    if "(Select n)" in question_text:
        question_text = question_text.replace("(Select n)", f"(Select {answer_count})")

    question_panel = Align.center(Panel(Markdown(question_text)), vertical="middle")
    console.print(question_panel)

    table = Table(title="Options", style="green")
    table.add_column("Option Number", justify="center")
    table.add_column("Option Text", justify="left")

    for idx, option in enumerate(options_list, 1):
        table.add_row(str(idx), option["text"])

    table.add_row(str(len(options_list) + 1), BAD_QUESTION_TEXT)
    console.print(Align.center(table))

    option_count = len(options_list) + 1
    while True:
        answer = console.input(
            "[bold yellow]Enter your answer(s) as a comma-separated list (e.g., 1,2): [/bold yellow]"
        )
        is_valid_answer, error_msg = is_valid(answer, option_count, answer_count)
        if is_valid_answer:
            break
        console.print(f"[bold red]{error_msg}[/bold red]")

    selected = [
        options_list[int(idx) - 1]
        for idx in answer.split(",")
        if idx.isdigit() and 1 <= int(idx) <= len(options_list)
    ]
    return selected


def ask_question_machine(
    provider: AnswerProvider,
    question: dict[str, Any],
    options_list: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    # No terminal I/O, no BAD_QUESTION sentinel ever chosen by the machine
    return provider(question, options_list)


def interactive_question_and_answer(
    questions: list[dict[str, Any]],
    session: list[dict[str, Any]],
    session_path: Path,
    start_time: datetime,
    *,
    answer_provider: AnswerProvider | None = None,
    quiet: bool = False,
) -> int:
    """Run through questions using either interactive or machine answer provider."""
    score = 0
    so_far = 0

    # Count already completed questions
    for question in session:
        if question.get("user_score") == 1:
            score += 1
            so_far += 1

    random.shuffle(questions)
    for question in questions:
        session_question = find_question(question, session)
        if session_question.get("user_score") == 1:
            continue

        # Record start time
        question_start_time = datetime.now()
        session_question["start_time"] = question_start_time.isoformat()

        options_list = list(question["options"])
        random.shuffle(options_list)

        try:
            if answer_provider is None:
                selected = ask_question_interactive(question, options_list)
            else:
                selected = ask_question_machine(answer_provider, question, options_list)
        except KeyboardInterrupt:
            if not quiet:
                display_results(score, len(questions), start_time, session)
            raise

        # Record completion time
        session_question["completion_time"] = datetime.now().isoformat()

        correct = {o["text"] for o in options_list if o.get("is_correct", False)}
        user_answers = {o["text"] for o in selected}

        # Feedback (skip in quiet mode)
        if not quiet:
            if user_answers == correct:
                console.print(
                    Panel(
                        "[bold green]✓ Correct![/bold green]",
                        title="Answer Review",
                        style="green",
                    )
                )
            else:
                console.print(
                    Panel(
                        f"[bold cyan]Correct Answer(s): {', '.join(correct)}\nYour Answer(s): {', '.join(user_answers)}[/bold cyan]",
                        title="Answer Review",
                        style="blue",
                    )
                )
            colored_explanations = []
            for idx, option in enumerate(options_list, 1):
                if option.get("is_correct", False):
                    colored_explanations.append(
                        f"{idx}. [bold green]{option['explanation']}[/bold green]"
                    )
                else:
                    colored_explanations.append(
                        f"{idx}. [bold red]{option['explanation']}[/bold red]"
                    )
            console.print(Panel("\n".join(colored_explanations), title="Explanation"))

        session_question["user_answers"] = list(user_answers)
        if user_answers == correct:
            if not quiet:
                play_sound("correct.mp3")
            score += 1
            session_question["user_score"] = 1
        else:
            if not quiet:
                console.print("[bold red]Incorrect.[/bold red]", style="bold red")
                play_sound("incorrect.mp3")
            session_question["user_score"] = 0

        so_far += 1
        if not quiet:
            display_results(score, so_far, start_time, session, withhold_judgement=True)

        if answer_provider is None:
            go_on = None
            while go_on not in ("", "bad"):
                go_on = console.input(
                    "[bold yellow]Press Enter to continue to the next question...[/bold yellow]"
                )
            if go_on == "bad":
                session_question["defective"] = True
                save_session_file(session_path, session, start_time)

    if not quiet:
        clear_screen()
        display_results(score, len(questions), start_time, session)
    save_session_file(session_path, session, start_time)
    return score


def take_exam_machine(
    question_file: str,
    *,
    strategy: MachineStrategy = "oracle",
    seed: int | None = 42,
    quiet: bool = True,
    persist_session: bool = False,
) -> dict[str, Any]:
    """Non-interactive exam runner for integration tests.

    Returns:
      dict with keys: score, total, percent, session_path, session, start_time
    """
    test_path = Path(question_file)
    test_name = test_path.stem
    questions = load_questions(str(test_path))

    # Fresh session each time unless you deliberately persist
    session_path = get_session_path(test_name)
    if not persist_session and session_path.exists():
        session_path.unlink(missing_ok=True)

    session = questions.copy()
    start_time = datetime.now()
    save_session_file(session_path, session, start_time)

    score = interactive_question_and_answer(
        questions,
        session,
        session_path,
        start_time,
        answer_provider=build_machine_answer_provider(strategy=strategy, seed=seed),
        quiet=quiet,
    )

    total = len(questions)
    percent = (score / total * 100) if total else 0.0

    return {
        "score": score,
        "total": total,
        "percent": percent,
        "session_path": session_path,
        "session": session,
        "start_time": start_time,
    }


if __name__ == "__main__":
    take_exam_now()
```
## File: validate_questions.py
```python
"""
Example toml


[[questions]]
question = "What is the primary purpose of Amazon Athena?"
id = "10fc5083-5528-4be1-a3cf-f377ae963dfc"

[[questions.options]]
text = "To perform ad-hoc querying on data stored in S3 using SQL."
explanation = "Amazon Athena allows users to run SQL queries directly on data in S3 without needing to manage any infrastructure. Correct."
is_correct = true

[[questions.options]]
text = "To manage relational databases on EC2."
explanation = "Amazon Athena is a serverless query service, and it does not manage databases on EC2. Incorrect."
is_correct = false

Rich-enhanced validator for multiple-choice TOML question sets.
- User-facing output uses Rich (progress bars, panels, tables).
- Developer logs go to logger.info / logger.debug via RichHandler.
- Avoids retry loops on fatal API errors (missing keys, auth, bad model).
"""

from __future__ import annotations

import csv
import logging
import os
from io import StringIO
from pathlib import Path
from time import perf_counter, sleep
from typing import Any

import dotenv
import rtoml as toml
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
from rich.table import Table

from examexam.apis.conversation_and_router import Conversation, Router
from examexam.jinja_management import jinja_env
from examexam.utils.custom_exceptions import ExamExamTypeError

# ----------------------------------------------------------------------------
# Env & logging setup
# ----------------------------------------------------------------------------
# Load environment variables (e.g., OPENAI_API_KEY / ANTHROPIC_API_KEY)
dotenv.load_dotenv()

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(message)s",
        handlers=[
            RichHandler(
                rich_tracebacks=True, markup=True, show_time=False, show_level=True
            )
        ],
    )

console = Console()


# ----------------------------------------------------------------------------
# Fatal error detection & LLM helpers
# ----------------------------------------------------------------------------
class FatalLLMError(Exception):
    """Raised for obviously fatal misconfigurations (missing API keys, etc.)."""


def _fatal_precheck(model: str) -> None:
    """Detect common fatal cases before calling the LLM so we don't loop."""
    m = model.lower()
    if m in {"fakebot", "none", "noop"}:
        return
    if "gpt" in m and not os.getenv("OPENAI_API_KEY"):
        raise FatalLLMError("OPENAI_API_KEY is not set for OpenAI model.")
    if "claude" in m and not os.getenv("ANTHROPIC_API_KEY"):
        raise FatalLLMError("ANTHROPIC_API_KEY is not set for Claude model.")


def _is_fatal_message(msg: str) -> bool:
    msg = msg.lower()
    markers = [
        "api_key client option must be set",
        "no api key",
        "invalid api key",
        "unauthorized",
        "model not found",
        "does not exist or you do not have access",
        "access denied",
    ]
    return any(k in msg for k in markers)


def _llm_call(
    prompt: str,
    model: str,
    system: str,
    *,
    max_retries: int = 2,
    retry_delay_seconds: float = 1.25,
) -> str | None:
    """Make a guarded LLM call with minimal retries and fatal detection."""
    _fatal_precheck(model)

    conversation = Conversation(system=system)
    router = Router(conversation)

    attempts = 0
    while True:
        attempts += 1
        try:
            t0 = perf_counter()
            content = router.call(prompt, model)
            dt = perf_counter() - t0
            logger.debug("router.call returned len=%d in %.2fs", len(content or ""), dt)
            return content
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


# ----------------------------------------------------------------------------
# Core utilities (unchanged logic, richer logs)
# ----------------------------------------------------------------------------


def read_questions(file_path: Path) -> list[dict[str, Any]]:
    """Reads a TOML file and returns the list of questions."""
    logger.debug("Reading TOML questions from %s", file_path)
    with open(file_path, encoding="utf-8") as file:
        data = toml.load(file)
    questions = data.get("questions", [])
    logger.info("Loaded %d questions", len(questions))
    return questions


def parse_answer(answer: str) -> list[str]:
    """Parses the string response from the LLM to extract the answers."""
    if answer.startswith("Answers:"):
        answer = answer[8:]
        if (
            "','" in answer or "', '" in answer or '","' in answer or '", "' in answer
        ) and "|" not in answer:
            return parse_quote_lists(answer)

        if "[" in answer and "]" in answer:
            after_square_bracket = answer.split("[")[1]
            answer_part = after_square_bracket.split("]")[0]

            answer_part = answer_part.replace('", "', "|").strip('"')
            answers = answer_part.strip().strip("[]").split("|")
            return [ans.strip("'\" ").strip("'\" ") for ans in answers]
    return []


def parse_quote_lists(answer: str) -> list[str]:
    """Helper function to parse comma-separated, quoted lists."""
    if "[" in answer and "]" in answer:
        after_square_bracket = answer.split("[")[1]
        answer_part = after_square_bracket.split("]")[0]

        if "', '" in answer_part or '","' in answer_part:
            answer_part_io = StringIO(answer_part)
            reader = csv.reader(answer_part_io, delimiter=",")
            answers = next(reader)
            return answers

        # Clean odd quotes sometimes returned by models
        answer_part = answer_part.replace("â€˜", "").replace("â€™", "")
        answer_part = answer_part.replace('", "', "|").strip('"')
        answers = answer_part.strip("[] ").split("|")
        return [ans.strip("'\" ").strip("'\" ") for ans in answers]
    return []


def ask_llm(
    question: str, options: list[str], answers: list[str], model: str, system: str
) -> list[str]:
    """Asks the LLM to answer a given question."""
    if "(Select" not in question:
        question = f"{question} (Select {len(answers)})"

    try:
        template = jinja_env.get_template("answer_question.md.j2")
        prompt = template.render(question=question, options=options)
    except Exception as e:
        logger.error(
            "Failed to load or render Jinja2 template 'answer_question.md.j2': %s", e
        )
        raise

    content = _llm_call(prompt, model=model, system=system)
    if content is None:
        logger.debug("ask_llm returned None content; treating as no answer")
        return []

    content = content.strip()
    logger.debug("ask_llm raw content: %r", content[:200])
    if content.startswith("Answers:"):
        parsed = parse_answer(content)
        logger.debug("ask_llm parsed answers: %s", parsed)
        return parsed
    raise ExamExamTypeError(
        f"Unexpected response format, didn't start with Answers:, got {content[:120]!r}"
    )


def ask_if_bad_question(
    question: str, options: list[str], answers: list[str], model: str
) -> tuple[str, str]:
    """Asks the LLM to evaluate if a question is Good or Bad."""
    try:
        template = jinja_env.get_template("evaluate_question.md.j2")
        prompt = template.render(question=question, options=options, answers=answers)
    except Exception as e:
        logger.error(
            "Failed to load or render Jinja2 template 'evaluate_question.md.j2': %s", e
        )
        raise

    system = "You are a test reviewer and are validating questions."
    content = _llm_call(prompt, model=model, system=system)
    if content is None:
        return "bad", "**** Bot returned None, maybe API failed ****"

    content = content.strip()
    logger.debug("ask_if_bad_question raw content: %r", content[:200])
    if "---" in content:
        return parse_good_bad(content)
    raise ExamExamTypeError(
        f"Unexpected response format, didn't contain '---'. got {content[:120]!r}"
    )


def parse_good_bad(answer: str) -> tuple[str, str]:
    """Parses the good/bad response from the LLM."""
    parts = answer.split("---")
    why = parts[0]
    good_bad = parts[1].strip(" \n").lower()
    if "good" in good_bad:
        return "good", why
    return "bad", why


# You will need this helper function inside grade_test or at the module level


def _is_array_of_tables(val: Any) -> bool:
    return isinstance(val, list) and (len(val) == 0 or isinstance(val[0], dict))


# ----------------------------------------------------------------------------
# Grading & Orchestration with Rich progress
# ----------------------------------------------------------------------------


def grade_test(
    questions: list[dict[str, Any]],
    responses: list[list[str]],
    good_bad: list[tuple[str, str]],
    file_path: Path,
    model: str,
) -> float:
    """Grades the LLM's performance and writes results to a TOML file."""
    score = 0
    total = len(questions)
    questions_to_write: list[dict[str, Any]] = []
    failures: list[tuple[str, str, set[str], set[str]]] = (
        []
    )  # (id, question, correct, given)

    for question, response, opinion in zip(questions, responses, good_bad, strict=True):
        correct_answers = {
            opt["text"] for opt in question.get("options", []) if opt.get("is_correct")
        }
        given_answers = set(response)

        if correct_answers == given_answers:
            score += 1
        else:
            failures.append(
                (
                    question.get("id", "<no-id>"),
                    question.get("question", "<no-question>"),
                    correct_answers,
                    given_answers,
                )
            )

        # Build new question dict without mutating original, scalars first.
        new_question_data = {
            k: v for k, v in question.items() if not _is_array_of_tables(v)
        }
        new_question_data[f"{model}_answers"] = sorted(list(given_answers))
        new_question_data["good_bad"], new_question_data["good_bad_why"] = opinion

        for k, v in question.items():
            if _is_array_of_tables(v):
                new_question_data[k] = v

        questions_to_write.append(new_question_data)

    # Write results next to the original file
    out_path = file_path
    with open(out_path, "w", encoding="utf-8") as file:
        toml.dump({"questions": questions_to_write}, file)

    # Pretty print summary
    console.print(
        Panel.fit(
            f"Final Score: [bold]{score}[/] / [bold]{total}[/]", title="Grading Summary"
        )
    )

    if failures:
        table = Table(title=f"Incorrect ({len(failures)})", show_lines=False)
        table.add_column("ID", no_wrap=True, overflow="fold")
        table.add_column("Question", overflow="fold")
        table.add_column("Correct", overflow="fold")
        table.add_column("Given", overflow="fold")
        for qid, qtext, correct, given in failures[
            :25
        ]:  # show top 25 to keep output readable
            table.add_row(
                str(qid), qtext, " | ".join(sorted(correct)), " | ".join(sorted(given))
            )
        if len(failures) > 25:
            table.caption = f"(+{len(failures) - 25} more not shown)"
        console.print(table)

    return 0 if total == 0 else score / total


def validate_questions_now(
    file_name: str,
    model: str = "claude",
) -> float:
    """Main function to orchestrate the validation process with Rich progress."""
    file_path = Path(file_name)
    if not file_path.exists():
        console.print(
            Panel.fit(f"[red]TOML file not found:[/] {file_name}", title="Error")
        )
        return 0.0

    questions = read_questions(file_path)
    total = len(questions)
    if total == 0:
        console.print(
            Panel.fit(
                "[yellow]No questions found in TOML.[/]", title="Nothing to validate"
            )
        )
        return 0.0

    console.rule("[bold]Exam Question Validation")
    console.print(
        f"Validating [bold]{total}[/] questions using model [italic]{model}[/]…\n"
    )

    responses: list[list[str]] = []
    opinions: list[tuple[str, str]] = []

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

    with progress:
        overall_task = progress.add_task("Overall", total=total)

        for idx, question_data in enumerate(questions, start=1):
            q = question_data.get("question", "<no-question>")
            opts = question_data.get("options", [])
            option_texts = [opt.get("text", "") for opt in opts]
            correct_answer_texts = [
                opt.get("text", "") for opt in opts if opt.get("is_correct")
            ]

            # Show per-question task
            desc = f"{idx}/{total} answering"
            q_task = progress.add_task(desc, total=2)  # step 1: answer, step 2: review

            try:
                resp = ask_llm(
                    q,
                    option_texts,
                    correct_answer_texts,
                    model,
                    system="You are test evaluator.",
                )
                responses.append(resp)
            except ExamExamTypeError as e:
                logger.error("ask_llm parse error: %s", e)
                responses.append([])
            finally:
                progress.advance(q_task)

            try:
                op = ask_if_bad_question(q, option_texts, correct_answer_texts, model)
                opinions.append(op)
            except ExamExamTypeError as e:
                logger.error("ask_if_bad_question parse error: %s", e)
                opinions.append(("bad", "**** parse error ****"))
            finally:
                progress.update(q_task, description=f"{idx}/{total} reviewed")
                progress.advance(q_task)
                progress.advance(overall_task)

    score = grade_test(questions, responses, opinions, file_path, model)
    console.rule()
    return score


if __name__ == "__main__":
    # Example usage; update paths as needed.
    validate_questions_now(
        file_name="personal_multiple_choice_tests.toml",
        model="claude",
    )
```
## File: __main__.py
```python
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


def main(argv: Sequence[str] | None = None) -> int:
    """Main function for the command-line interface."""
    start_background_update_check("examexam", __about__.__version__)
    parser = SmartParser(
        prog=__about__.__title__,
        description="A CLI for generating, taking, and managing exams.",
        formatter_class=argparse.RawTextHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__about__.__version__}"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        required=False,
        help="Enable detailed logging.",
    )
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands", required=True
    )

    # --- Take Command ---
    take_parser = subparsers.add_parser("take", help="Take an exam from a TOML file.")
    take_parser.add_argument(
        "--question-file",
        type=str,
        default="",
        required=False,
        help="Path to the TOML question file.",
    )

    # --- Generate Command ---
    generate_parser = subparsers.add_parser(
        "generate", help="Generate new exam questions using an LLM."
    )

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
    validate_parser = subparsers.add_parser(
        "validate", help="Validate exam questions using an LLM."
    )
    validate_parser.add_argument(
        "--question-file",
        type=str,
        required=True,
        help="Path to the TOML question file to validate.",
    )
    add_model_args(validate_parser)

    # --- Convert Command ---
    convert_parser = subparsers.add_parser(
        "convert", help="Convert a TOML question file to Markdown and HTML formats."
    )
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
    research_parser = subparsers.add_parser(
        "research", help="Generate a study guide for a specific topic."
    )
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

    # --- Customize Command (New) ---
    customize_parser = subparsers.add_parser(
        "customize",
        help="Deploy Jinja2 templates to a local directory for customization.",
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
        model = pick_model(args.model, args.model_provider, args.model_class)

        generate_questions_now(
            questions_per_toc_topic=args.n,
            file_name=args.output_file or toc_file_base.replace(".txt", ".toml"),
            toc_file=args.toc_file,
            model=model,
            system_prompt="You are a test maker.",
        )
    elif args.command == "validate":
        model = pick_model(args.model, args.model_provider, args.model_class)
        validate_questions_now(file_name=args.question_file, model=model)
    elif args.command == "research":
        model = pick_model(args.model, args.model_provider, args.model_class)
        generate_topic_research_now(topic=args.topic, model=model)
    elif args.command == "study-plan":
        model = pick_model(args.model, args.model_provider, args.model_class)
        generate_study_plan_now(toc_file=args.toc_file, model=model)
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
```
## File: apis\conversation_and_router.py
```python
# conversation_and_router.py
from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from typing import Any

from examexam.apis.third_party_apis import (
    AnthropicCaller,
    BedrockCaller,
    FakeBotCaller,
    FakeBotException,
    GoogleCaller,
    OpenAICaller,
)
from examexam.apis.types import Conversation, ExamExamValueError, FatalConversationError
from examexam.apis.utilities import log_conversation_to_file, log_duration

LOGGER = logging.getLogger(__name__)

# Map bot class to specific bot model
FRONTIER_MODELS = {
    "fakebot": "fakebot",
    "openai": "gpt-5",  # Aug 2025, current flagship
    "anthropic": "claude-opus-4-1-20250805",  # Aug 2025, strongest Claude
    "google": "gemini-2.5-pro",  # June 2025, top Gemini
    "meta": "llama-3.1-405b-instruct",  # July 2025, Meta’s largest model
    "mistral": "mixtral-8x22b-instruct-v0.1",  # 2025 frontier release
    "cohere": "command-r-plus-08-2025",  # Cohere’s reasoning-tuned flagship
    "ai21": "jamba-1.5-large",  # AI21’s strongest hybrid model
    "amazon": "amazon.nova-pro-v1",  # Amazon’s own top Bedrock model
}
GOOD_FAST_CHEAP_MODELS = {
    "openai": "gpt-4.1-mini",  # lightweight, fast, inexpensive
    "anthropic": "claude-3.7-sonnet",  # Feb 2025, balance of speed/cost
    "google": "gemini-2.5-flash",  # optimized for speed/cheap inference
    "meta": "llama-3.1-8b-instruct",  # small open-source Llama
    "mistral": "mistral-7b-instruct-v0.3",  # efficient small model
    "cohere": "command-r-08-2025",  # cheaper sibling to “plus”
    "ai21": "jamba-1.5-mini",  # fast, smaller Jamba
    "amazon": "amazon.nova-lite-v1",  # Amazon’s cost-optimized Bedrock model
}


def pick_model(model: str, provider: str, model_class: str):
    if model:
        return model
    if model_class == "frontier":
        return FRONTIER_MODELS[provider]
    return GOOD_FAST_CHEAP_MODELS[provider]


class Router:
    """
    Routes requests to various LLM APIs, maintaining conversation state and handling errors.
    """

    def __init__(self, conversation: Conversation):
        self.standard_conversation: Conversation = conversation
        self.callers: dict[str, Any] = {}
        self.errors_so_far = 0
        self.conversation_cannot_continue = False

        self.most_recent_python: str | None = None
        self.most_recent_answer: str | None = None
        self.most_recent_json: dict[str, Any] | list[Any] | None = None
        self.most_recent_just_code: list[str] | None = None
        self.most_recent_exception: Exception | None = None

        self._caller_map = {
            "openai": OpenAICaller,
            "anthropic": AnthropicCaller,
            "google": GoogleCaller,
            "fakebot": FakeBotCaller,
            "mistral": BedrockCaller,
            "cohere": BedrockCaller,
            "meta": BedrockCaller,
            "ai21": BedrockCaller,
            "amazon": BedrockCaller,
        }

    def reset(self) -> None:
        """Resets the state of the most recent call."""
        self.most_recent_python = None
        self.most_recent_answer = None
        self.most_recent_json = None
        self.most_recent_just_code = None
        self.most_recent_exception = None

    def _get_caller(self, model_provider: str, model_id: str) -> Any:
        """Lazily initializes and returns the appropriate API caller."""
        caller_class = self._caller_map.get(model_provider)
        if not caller_class:
            print(f"unkown model provider {model_provider}")
            sys.exit(-100)
            # raise FatalConversationError(f"Unknown model {model_key}")

        # Use the class name as the key to store only one instance per caller type
        caller_key = caller_class.__name__
        if caller_key not in self.callers:
            # model_id = FRONTIER_MODELS[model_key]
            if caller_class == AnthropicCaller:
                self.callers[caller_key] = AnthropicCaller(
                    model=model_id, conversation=self.standard_conversation, tokens=4096
                )
            else:
                self.callers[caller_key] = caller_class(
                    model=model_id, conversation=self.standard_conversation
                )

        # For callers like Bedrock that handle multiple models, update the model ID
        caller_instance = self.callers[caller_key]
        caller_instance.model = model_id  #  FRONTIER_MODELS[model_key]

        return caller_instance

    @log_duration
    def call(self, request: str, model: str, essential: bool = False) -> str | None:
        """
        Routes a request to the specified model and returns the response.

        Args:
            request: The user's prompt.
            model: The key for the model to use (e.g., 'gpt4', 'claude').
            essential: If True, an error during this call will halt future conversation.

        Returns:
            The model's string response, or None if an error occurred.
        """
        if self.conversation_cannot_continue:
            raise ExamExamValueError(
                "Conversation cannot continue, an essential exchange previously failed."
            )
        if not request:
            raise ExamExamValueError("Request cannot be empty")
        if len(request) < 5:
            LOGGER.warning(
                f"Request ('{request}') is short, which may be inappropriate for non-interactive use."
            )

        self.reset()
        LOGGER.info(f"Calling {model} with request of length {len(request)}")

        # deal with legacy behavior
        model_provider = ""
        for key, value in FRONTIER_MODELS.items():
            if value == model:
                model_provider = key

        for key, value in GOOD_FAST_CHEAP_MODELS.items():
            if value == model:
                model_provider = key

        if not model_provider:
            raise TypeError(f"Can't identify model provider for model {model}")

        caller = None
        try:
            caller = self._get_caller(model_provider, model)
            answer = caller.completion(request)
        except (FatalConversationError, FakeBotException) as e:
            self.most_recent_exception = e
            if self.standard_conversation:
                self.standard_conversation.error(e)
            if essential:
                self.conversation_cannot_continue = True
            self.errors_so_far += 1
            LOGGER.error(f"Error calling {model}: {e}")
            self.most_recent_answer = ""
            if isinstance(e, FatalConversationError):
                sys.exit(100)
            return None
        except Exception as e:
            self.most_recent_exception = e
            if self.standard_conversation:
                self.standard_conversation.error(e)
            if essential:
                self.conversation_cannot_continue = True
            if "pytest" in sys.modules:
                raise
            self.errors_so_far += 1
            LOGGER.error(f"Error calling {model} with request '{request[:15]}...': {e}")
            self.most_recent_answer = ""
            return None
        finally:
            if caller:
                log_conversation_to_file(
                    self.standard_conversation, caller.model, request
                )

        self.most_recent_answer = answer
        return answer

    def call_until(self, request: str, model: str, stop_check: Callable) -> str | None:
        """
        Calls a model repeatedly with the same request until the stop_check function returns True.

        Args:
            request: The request to send to the model.
            model: The model to call.
            stop_check: A function that takes the model's answer and returns True to stop.

        Returns:
            The final answer from the model that satisfied the stop_check.
        """
        answer = self.call(request, model)
        while not stop_check(answer):
            answer = self.call(request, model)
        return answer
```
## File: apis\third_party_apis.py
```python
# third_party_apis.py
from __future__ import annotations

import logging
import os
import random
from abc import ABC, abstractmethod

import anthropic
import google.generativeai as genai
import openai
from google.generativeai import ChatSession
from retry import retry

# Local application imports
from examexam.apis.types import Conversation, ExamExamTypeError
from examexam.apis.utilities import call_limit, load_env

LOGGER = logging.getLogger(__name__)
load_env()


class FakeBotException(ValueError):
    """Contrived simulation of an API error."""


class BaseLLMCaller(ABC):
    """Abstract base class for all LLM API callers."""

    def __init__(self, model: str, conversation: Conversation):
        self.model = model
        self.conversation = conversation

    @abstractmethod
    def completion(self, prompt: str) -> str:
        """Sends a prompt and returns the completion."""
        raise NotImplementedError


class OpenAICaller(BaseLLMCaller):
    """Handles API calls to OpenAI models."""

    _client = None

    def __init__(self, model: str, conversation: Conversation):
        super().__init__(model, conversation)
        if OpenAICaller._client is None:
            OpenAICaller._client = openai.OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY")
            )
        self.client = OpenAICaller._client
        self.supported_models = ["gpt-5", "gpt-4o-mini"]

    @call_limit(500)
    def completion(self, prompt: str) -> str:
        self.conversation.prompt(prompt)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.conversation.conversation,
        )
        if response.usage:
            LOGGER.info(
                f"Tokens used (prompt/completion/total): {response.usage.prompt_tokens}/"
                f"{response.usage.completion_tokens}/{response.usage.total_tokens}"
            )
        core_response = response.choices[0].message.content or ""
        role = response.choices[0].message.role or ""
        self.conversation.response(core_response, role)
        return core_response


class AnthropicCaller(BaseLLMCaller):
    """Handles API calls to Anthropic models."""

    def __init__(self, model: str, conversation: Conversation, tokens: int):
        super().__init__(model, conversation)
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.tokens = tokens
        self.supported_models = [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-2.1",
            "claude-2.0",
            "claude-instant-1.2",
        ]

    @retry(
        exceptions=anthropic.RateLimitError,
        tries=3,
        delay=5,
        jitter=(0.15, 0.23),
        backoff=1.5,
        logger=LOGGER,
    )
    def completion(self, prompt: str) -> str:
        self.conversation.prompt(prompt)
        try:
            message = self.client.messages.create(
                max_tokens=self.tokens,
                messages=self.conversation.without_system(),
                model=self.model,
                system=self.conversation.system,
            )
            LOGGER.info(f"Actual Anthropic token count {message.usage}")
            core_response = message.content[0].text
            self.conversation.response(core_response)
            return core_response
        except anthropic.RateLimitError as e:
            self.conversation.pop()
            LOGGER.warning(f"Anthropic rate limit hit: {e}. Backing off.")
            raise
        except Exception:
            self.conversation.pop()
            raise


class GoogleCaller(BaseLLMCaller):
    """Handles API calls to Google's Gemini models."""

    _initialized = False

    def __init__(self, model: str, conversation: Conversation):
        super().__init__(model, conversation)
        self._initialize_google()
        self.client = genai.GenerativeModel(
            model_name=self.model, system_instruction=conversation.system
        )
        self.chat: ChatSession | None = None
        self.supported_models = ["gemini-1.0-pro-001"]

    def _initialize_google(self):
        if GoogleCaller._initialized:
            return
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            try:
                from google.colab import userdata

                api_key = userdata.get("GOOGLE_API_KEY")
            except ImportError:
                pass  # No key found
        if api_key:
            genai.configure(api_key=api_key)
            GoogleCaller._initialized = True

    def completion(self, prompt: str) -> str:
        self.conversation.prompt(prompt)
        if not self.chat:
            self.chat = self.client.start_chat()

        message = (self.conversation.system or "") + "\n" + prompt
        response = self.chat.send_message(message)
        core_response = response.text
        self.conversation.response(core_response)
        return core_response


class BedrockCaller(BaseLLMCaller):
    """Handles API calls to AWS Bedrock models (Placeholder)."""

    def __init__(self, model: str, conversation: Conversation):
        super().__init__(model, conversation)
        # In a real implementation, the boto3 client would be initialized here.
        # self.client = boto3.client(service_name='bedrock-runtime')
        LOGGER.warning(
            "BedrockCaller is a placeholder and does not make real API calls."
        )

    def completion(self, prompt: str) -> str:
        self.conversation.prompt(prompt)
        # This would contain logic to invoke the correct model on Bedrock
        # using self.model (e.g., 'amazon.titan-text-express-v1')
        # For example: body = json.dumps({"inputText": prompt})
        # response = self.client.invoke_model(body=body, modelId=self.model)
        LOGGER.info(f"Pretending to call Bedrock model: {self.model}")
        response_text = f"This is a mocked response from Bedrock model {self.model}."
        self.conversation.response(response_text)
        return response_text


class FakeBotCaller(BaseLLMCaller):
    """A fake bot for integration tests and dry runs."""

    def __init__(
        self,
        model: str,
        conversation: Conversation,
        data: list[str] | None = None,
        reliable: bool = False,
    ):
        super().__init__(model, conversation)
        self.data = data or ["Answers: [1,2]\n---Blah blah. Bad."]
        self.reliable = reliable
        self.invocation_count = 0
        if self.model not in ["fakebot"]:
            raise ExamExamTypeError(
                f"FakeBotCaller doesn't support model: {self.model}"
            )

    def completion(self, prompt: str) -> str:
        self.invocation_count += 1
        self.conversation.prompt(prompt)

        if not self.reliable and random.random() < 0.1:  # nosec
            raise FakeBotException(
                "Fakebot has failed to return an answer, just like a real API."
            )

        core_response = random.choice(self.data)  # nosec
        LOGGER.info(f"FakeBot Response: {core_response.replace(chr(10), r' ')}")
        self.conversation.response(core_response)
        return core_response
```
## File: apis\types.py
```python
# --- Custom Exceptions ---
from __future__ import annotations


class ExamExamValueError(ValueError):
    """Custom value error for the application."""


class ExamExamTypeError(TypeError):
    """Custom type error for the application."""


class FatalConversationError(Exception):
    """Raised for unrecoverable errors in conversation flow."""


class FailureToHaltError(Exception):
    """Raised when a function is called more than its allowed limit."""


class Conversation:
    """Manages the state and flow of a conversation with an LLM."""

    def __init__(self, system: str) -> None:
        self.system = system
        self.conversation: list[dict[str, str]] = [
            {
                "role": "system",
                "content": system,
            },
        ]

    def prompt(self, prompt: str, role: str = "user") -> dict[str, str]:
        if self.conversation and self.conversation[-1]["role"] == role:
            raise FatalConversationError("Prompting the same role twice in a row")
        self.conversation.append(
            {
                "role": role,
                "content": prompt,
            },
        )
        return self.conversation[-1]

    def error(self, error: Exception) -> dict[str, str]:
        self.conversation.append(
            {"role": "examexam", "content": str(error)},
        )
        return self.conversation[-1]

    def response(self, response: str, role: str = "assistant") -> dict[str, str]:
        if self.conversation and self.conversation[-1]["role"] == role:
            raise FatalConversationError("Responding with the same role twice in a row")
        self.conversation.append(
            {
                "role": role,
                "content": response,
            },
        )
        return self.conversation[-1]

    def pop(self) -> None:
        """Removes the last message from the conversation."""
        if self.conversation:
            self.conversation.pop()

    def without_system(self) -> list[dict[str, str]]:
        """Returns the conversation history without the system message."""
        return [_ for _ in self.conversation if _["role"] != "system"]
```
## File: apis\utilities.py
```python
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
                raise FailureToHaltError(
                    f"{name} has been called more than {limit} times"
                )
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
        LOGGER.info(
            f"Function {func.__name__} executed in {end_time - start_time:.4f} seconds."
        )
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
    conversation: list[dict[str, str]],
    user_label: str = "User",
    assistant_label: str = "Assistant",
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


def log_conversation_to_file(
    conversation: Conversation, model_name: str, request: str
) -> None:
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
```
## File: prompts\answer_question.md.j2
```
Answer the following question in the format 'Answers: [option1 | option2 | ...]'.
Question: {{ question }}
Options: {{ options }}
```
## File: prompts\evaluate_question.md.j2
```
Tell me if the following question is Good or Bad, e.g. would it be unfair to ask this on a test.
It is good if it has an answer, if it not every single option is an answer, if it is not opinion based, if it does not have weasel words such as best, optimal, primary which would make many of the answers arguably true on some continuum of truth or opinion, or if the question is about numerical ephemeral truths, such as system limitations (max GB, etc) and UI defaults.

Question: {{ question }}
Options: {{ options }}
Answers: {{ answers }}

Think about the answer then write `---
Good` or `---
Bad`
```
## File: prompts\generate.md.j2
```
Generate {{ n }} medium difficulty certification exam questions. {{ prompt }}.
Follow the following TOML format:

```toml
[[questions]]
question = "Question for user here"

[[questions.options]]
text = "Some Correct answer. Must be first."
explanation = "Explanation. Must be before is_correct. Correct."
is_correct = true

[[questions.options]]
text = "Wrong Answer. Must be first."
explanation = "Explanation. Must be before is_correct. Incorrect."
is_correct = false
```
One or more can be correct!
Five options.
Each explanation must end in  "Correct" or "Incorrect", e.g. "Instance storage is ephemeral. Correct".
Do not use numbers or letters to represent the answers.
   [[questions.options]]
   text = "A. Answer"  # never do this.
   [[questions.options]]
   text = "1. Answer"  # never do this.
Do not use "All of the above" or the like as an answer.
```
## File: prompts\study_guide.md.j2
```
Create a comprehensive study guide for the following topic: {{ topic }}.

The study guide should be in Markdown format and include the following sections:

---
## Core Concepts:
A clear explanation of the fundamental ideas.

## Key Terminology:
Definitions of important terms.

## Code Examples:
Practical, well-commented code snippets (if applicable).

## Common Pitfalls:
Mistakes or misunderstandings to avoid.

## Further Research:
A list of suggested search engine queries to deepen understanding.

---
Structure the "Further Research" section like this, providing varied and useful queries (with hyperlinks!):

## Further Research
Google:
- {{ topic }} tutorial for beginners
- advanced {{ topic }} techniques
- {{ topic }} real-world examples

Google (with operators):
- site:stackoverflow.com {{ topic }} "common error"
- filetype:pdf {{ topic }} cheat sheet

Bing:
- {{ topic }} practical applications
- compare {{ topic }} with [related concept]

Bing (with operators):
- {{ topic }} inurl:blog
- {{ topic }} -"getting started"

Kagi:
- How does {{ topic }} work internally?
- Best practices for using {{ topic }}

Kagi (with operators):
- {{ topic }} !pinterest !shopping
- {{ topic }} discussion lang:en

---
Use additional operators as appropriate

Please generate the complete study guide now.
```
