"""Doctor / diagnostics for examexam.

Collects system information useful for tech support and troubleshooting.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import platform
import sys
from pathlib import Path


def _safe_import_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _env_key_status(key: str) -> str:
    """Return masked value if set, else 'not set'."""
    val = os.environ.get(key)
    if val is None:
        return "not set"
    if len(val) <= 8:
        return f"{'*' * len(val)} (set, {len(val)} chars)"
    return f"{val[:4]}...{val[-4:]} (set, {len(val)} chars)"


def collect_diagnostics() -> dict:
    """Collect diagnostic information and return as a dict."""
    # Python / platform
    diag: dict = {}

    diag["python_version"] = sys.version
    diag["platform"] = platform.platform()
    diag["machine"] = platform.machine()
    diag["processor"] = platform.processor()
    diag["cwd"] = str(Path.cwd())
    diag["executable"] = sys.executable

    # Package versions
    packages = [
        "examexam",
        "openai",
        "anthropic",
        "google-generativeai",
        "mistralai",
        "boto3",
        "rich",
        "rtoml",
        "toml",
        "Jinja2",
        "python-dotenv",
        "argcomplete",
        "textual",
        "fastapi",
        "uvicorn",
    ]
    diag["packages"] = {pkg: _safe_import_version(pkg) for pkg in packages}

    # API key presence (masked)
    api_keys = [
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
    diag["api_keys"] = {key: _env_key_status(key) for key in api_keys}

    # examexam config file
    config_path = Path("examexam.toml")
    diag["config_file"] = str(config_path.resolve())
    diag["config_exists"] = config_path.exists()

    # .env file
    env_path = Path(".env")
    diag["dotenv_file"] = str(env_path.resolve())
    diag["dotenv_exists"] = env_path.exists()

    # TOML question files in current dir
    toml_files = list(Path.cwd().glob("*.toml"))
    diag["toml_files_cwd"] = [str(p) for p in toml_files]

    # tkinter availability
    try:
        import tkinter  # noqa: F401

        diag["tkinter"] = "available"
    except ImportError:
        diag["tkinter"] = "not available"

    return diag


def format_diagnostics(diag: dict) -> str:
    """Format diagnostics dict into a human-readable string."""
    lines = []
    lines.append("=" * 60)
    lines.append("ExamExam Doctor Report")
    lines.append("=" * 60)

    lines.append("\n[System]")
    lines.append(f"  Python:     {diag['python_version']}")
    lines.append(f"  Platform:   {diag['platform']}")
    lines.append(f"  Machine:    {diag['machine']}")
    lines.append(f"  Executable: {diag['executable']}")
    lines.append(f"  CWD:        {diag['cwd']}")

    lines.append("\n[Package Versions]")
    for pkg, ver in diag["packages"].items():
        lines.append(f"  {pkg:<30} {ver}")

    lines.append("\n[API Keys]")
    for key, status in diag["api_keys"].items():
        lines.append(f"  {key:<30} {status}")

    lines.append("\n[Config]")
    lines.append(f"  examexam.toml: {diag['config_file']} ({'exists' if diag['config_exists'] else 'MISSING'})")
    lines.append(f"  .env file:     {diag['dotenv_file']} ({'exists' if diag['dotenv_exists'] else 'not found'})")

    lines.append("\n[TOML Question Files in CWD]")
    if diag["toml_files_cwd"]:
        for f in diag["toml_files_cwd"]:
            lines.append(f"  {f}")
    else:
        lines.append("  (none found)")

    lines.append("\n[GUI]")
    lines.append(f"  tkinter: {diag['tkinter']}")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


def run_doctor() -> str:
    """Run diagnostics and return formatted report."""
    diag = collect_diagnostics()
    return format_diagnostics(diag)
