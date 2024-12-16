import logging
import os
from typing import Literal

import openai

from examexam.apis.conversation_model import Conversation
from examexam.apis.halting_checker import call_limit
from examexam.utils.env_loader import load_env

CLIENT = None


load_env()

LOGGER = logging.getLogger(__name__)


def get_client(force_new_client: bool = False) -> openai.OpenAI:
    # pylint: disable=global-statement
    global CLIENT
    if CLIENT is None or force_new_client:
        # base_url = "localhost://"
        client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),  # this is also the default, it can be omitted
        )
        CLIENT = client
    return CLIENT


OpenAIModels = Literal["gpt-4o-mini"]
OPENAI_SUPPORTED_MODELS = ["gpt-4o-mini"]


class OpenAICaller:
    def __init__(
        self,
        model: str,
        conversation: Conversation,
    ):
        self.conversation = conversation

        self.model = model
        self.client = get_client()

        self.supported_models = OPENAI_SUPPORTED_MODELS

    @call_limit(500)
    def completion(self, prompt: str) -> str:
        self.conversation.prompt(prompt)

        completion = self.client.chat.completions.create(
            # model="gpt-4-0125-preview", # expensive, but smart. 20x more expensive
            # "gpt-3.5-turbo-1106"
            model=self.model,
            # stub because gpt4 is so expensive.
            messages=self.conversation.conversation,
            # temperature
            # top_p
            # max_tokens =
        )

        if completion.usage:
            LOGGER.info(
                f"Prompt/completion/total tokens: {completion.usage.prompt_tokens}"
                f"/{completion.usage.completion_tokens}/{completion.usage.total_tokens}"
            )

        core_response = completion.choices[0].message.content or ""
        role = completion.choices[0].message.role or ""
        self.conversation.response(core_response, role)

        return core_response


if __name__ == "__main__":

    def example():
        conversation = Conversation("You are a python 3.11 developer.", "full", model="test-model")
        caller = OpenAICaller(model="gpt-3.5-turbo-1106", conversation=conversation)

        result = caller.completion("How do I display a character in a console at a specific location?")
        print(result)
        result = caller.completion("I see, now how to unit test that?")
        print(result)

    example()
