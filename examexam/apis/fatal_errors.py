"""Shared fatal-error detection for LLM call sites.

Used by both generate_questions.py and validate_questions.py to avoid retrying
calls that are doomed to fail the same way every time (missing/bad API keys,
unauthorized, unknown model, etc).
"""

from __future__ import annotations

import os


class FatalLLMError(Exception):
    """Errors that should not be retried (e.g., missing API key)."""


def fatal_if_misconfigured(model: str) -> None:
    """Raise FatalLLMError for obviously fatal misconfigurations before calling an LLM."""
    m = model.lower()
    if m in {"fakebot", "none", "noop"}:
        return
    if "gpt" in m and not os.getenv("OPENAI_API_KEY"):
        raise FatalLLMError("OPENAI_API_KEY is not set for OpenAI model.")
    if "claude" in m and not os.getenv("ANTHROPIC_API_KEY"):
        raise FatalLLMError("ANTHROPIC_API_KEY is not set for Claude model.")


_FATAL_MESSAGE_MARKERS = (
    "api_key client option must be set",
    "no api key",
    "invalid api key",
    "unauthorized",
    "model not found",
    "does not exist or you do not have access",
    "access denied",
)


def is_fatal_message(msg: str) -> bool:
    """Detects error messages that indicate a misconfiguration, not a transient failure."""
    msg_lower = msg.lower()
    return any(marker in msg_lower for marker in _FATAL_MESSAGE_MARKERS)
