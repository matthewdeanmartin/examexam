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
            logger.debug(f"Config file not found at {self._config_path}. Using defaults.")

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
                            logger.debug(f"Overrode '{section}.{key}' with env var '{env_var_name}'.")
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
                if key in base and isinstance(base[key], dict) and not isinstance(value, dict):
                    logger.warning(f"Config conflict: Section '{key}' cannot be overridden by a non-section value.")
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


def create_default_config_if_not_exists(filename: str = DEFAULT_CONFIG_FILENAME) -> bool:
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
