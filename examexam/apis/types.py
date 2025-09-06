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
