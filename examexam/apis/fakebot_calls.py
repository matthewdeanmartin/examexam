"""
A fake bot for integration tests, dry runs, etc..

Always returns 'I don't know'
"""

import logging
import random
from typing import Literal

from examexam.apis.conversation_model import Conversation
from examexam.utils.custom_exceptions import ExamExamTypeError
from examexam.utils.env_loader import load_env

load_env()

LOGGER = logging.getLogger(__name__)

DATA = ["Answers: [1,2]\n---Blah blah. Bad."]

FakeBotModels = Literal["fakebot"]


class FakeBotException(ValueError):
    """Contrived simulation of API error"""


class FakeBotCaller:
    def __init__(
        self,
        model: str,
        conversation: Conversation,
        data: list[str] | None = None,
        reliable: bool = False,
    ):
        self.conversation = conversation
        self.model = model
        self.supported_models = ["fakebot"]
        if self.model not in self.supported_models:
            raise ExamExamTypeError(f"Caller doesn't support that model : {self.model}")
        if data is None:
            self.data = DATA
        else:
            self.data = data
        self.reliable = reliable
        self.invocation_count = 0

    def completion(self, prompt: str) -> str:
        self.invocation_count += 1
        self.conversation.prompt(prompt)

        self.conversation.templatize_conversation()

        if not self.reliable:
            if random.random() < 0.1:  # nosec
                raise FakeBotException("Fakebot has failed to return an answer, just like a real API.")

        core_response = random.choice(self.data)  # nosec
        LOGGER.info(core_response.replace("\n", "\\n"))

        self.conversation.response(core_response)

        return core_response
