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
        handlers=[RichHandler(rich_tracebacks=True, markup=True, show_time=False, show_level=True)],
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
                        logger.warning("Skipping modified template '%s'. Use --force to overwrite.", src_file.name)
                        should_write = False
                # If no original hash, we assume it's user-managed and don't touch it without force.
                else:
                    logger.warning("Skipping template '%s' (no hash record). Use --force to overwrite.", src_file.name)
                    should_write = False

            if should_write:
                dest_file_path.write_bytes(src_content)
                logger.debug("Deployed template: %s", dest_file_path)

        except Exception as e:
            logger.error("Failed to process or deploy template '%s': %s", src_file.name, e)

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
        logger.debug("Loading Jinja2 templates from user-customized directory: %s", CUSTOM_PROMPTS_DIR.resolve())
        loader = FileSystemLoader(CUSTOM_PROMPTS_DIR)
        return Environment(loader=loader, autoescape=False)  # nosec

    # 2. Check for development mode prompts
    dev_prompts_path = Path(__file__).parent.parent / "prompts"
    if dev_prompts_path.is_dir():
        logger.debug("Loading Jinja2 templates from development directory: %s", dev_prompts_path)
        loader = FileSystemLoader(dev_prompts_path)
        return Environment(loader=loader, autoescape=False)  # nosec

    # 3. Fallback to installed package prompts
    logger.debug("Loading Jinja2 templates from installed package 'examexam.prompts'")
    try:
        alt_loader = PackageLoader("examexam", "prompts")
        return Environment(loader=alt_loader, autoescape=False)  # nosec
    except ModuleNotFoundError:
        logger.error("Could not find the 'examexam' package to load templates.")
        raise


# Create a single environment instance to be used by the module
jinja_env = get_jinja_env()
