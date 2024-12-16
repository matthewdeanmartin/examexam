import logging
from typing import Literal

from examexam.utils.custom_exceptions import ExamExamTypeError

LOGGER = logging.getLogger(__name__)


class FatalConversationError(Exception):
    """Missing code or the like"""


ConversationMode = Literal["no_context", "minimal_context", "full"]


class Conversation:
    def __init__(self, system: str, mode: ConversationMode, model: str | None = None) -> None:
        self.system = system
        self.model = model
        self.conversation: list[dict[str, str]] = [
            {
                "role": "system",
                "content": system,
            },
        ]
        self.mode: ConversationMode = mode
        self.stub_conversation = []

    def reinitialize_stub(self) -> None:
        self.stub_conversation = [
            {
                "role": "system",
                "content": self.system,
            },
        ]

    def prompt(self, prompt: str, role: str = "user") -> dict[str, str]:
        if self.conversation[-1]["role"] == role:
            raise FatalConversationError("Prompting the same role twice in a row")
        self.conversation.append(
            {
                "role": role,
                "content": prompt,
            },
        )
        return self.conversation[-1]

    def convert_to(self) -> list[dict[str, str]]:
        # mode: ConversationMode
        # if mode == "no_context":
        #     # initial prompt just repeats, 1st prompt has code, so we're okay
        #     return [self.conversation[0], self.conversation[1]]
        # if mode == "minimal_context":
        #     # latest exchange.
        #     if len(self.conversation) < 3:
        #         minimal = self.conversation
        #     else:
        #         minimal = [self.conversation[0]] + self.conversation[-3:]
        #     # make sure code not lost
        #     if "````" not in minimal[1]["content"]:
        #         minimal[1]["content"] = "\n".join(_ for _ in self.code if _) + minimal[1]["content"]
        #     if minimal[0]["role"] != "system":
        #         raise FatalConversationError("System prompt not found")
        #     if minimal[1]["role"] != "user":
        #         raise FatalConversationError("User prompt must be 2nd")
        #
        #     return minimal
        # if mode == "full":
        # mostly useless middle convo context, but no code is lost.
        return self.conversation

    def error(self, error: Exception) -> dict[str, str]:
        self.conversation.append(
            {"role": "examexam", "content": str(error)},
        )
        return self.conversation[-1]

    def response(self, response: str, role: str = "assistant") -> dict[str, str]:
        if self.conversation[-1]["role"] == role:
            raise FatalConversationError("Prompting the same role twice in a row")
        # This only applies to completion models that sometimes prefix answers with a role.
        if self.model not in ("gpt3.5", "gpt4", "claude", "gemini-pro"):
            response = clean_text(response)
        self.conversation.append(
            {
                "role": role,
                "content": response,
            },
        )
        return self.conversation[-1]

    def pop(self) -> None:
        self.conversation.pop()

    def without_system(self) -> list[dict[str, str]]:
        return [_ for _ in self.conversation if _["role"] != "system"]

    def templatize_conversation(self) -> str:

        conversation = self.conversation

        entire_conversation = ""
        for exchange in conversation:
            exchange_content = exchange["content"]
            # Make invisible characters visible
            if exchange_content is None:
                exchange_content = "**** Bot returned None, maybe API failed ****"
            elif exchange_content.strip() == "":
                exchange_content = "**** Bot returned whitespace ****"
            elif not exchange_content:
                exchange_content = f"**** Bot returned falsy value {exchange_content} ****"

            if exchange["role"] == "user":
                entire_conversation += f"User: {exchange_content}\n"
            elif exchange["role"] == "assistant":
                entire_conversation += f"Assistant: {exchange_content}\n"
            elif exchange["role"] == "system" and exchange["content"]:
                entire_conversation += f"User: {exchange_content}\n"
            elif exchange["role"] == "examexam":
                entire_conversation += f"ERROR: {exchange_content}\n"
            else:
                raise ExamExamTypeError(f"Unknown role {exchange['role']}")
        return entire_conversation


def clean_text(text: str) -> str:
    if not text:
        return ""
    # Define prefixes to remove
    prefixes = [
        "Assistant:",
        "Assistant: ",
        "Assistant:\n",
        "Assistant: \n",
        "Assistant: \n\n",
    ]

    # Remove any of the prefixes from the start of the text

    loops = 0
    while any(text.startswith(prefix) for prefix in prefixes):
        loops += 1
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix) :]
                text = text.strip()
        if loops > 10:
            # This only happens when the text is actually a mock.
            break
    return text.strip()  # Remove leading and trailing whitespaces
