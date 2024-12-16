"""
Route calls to a variety of sort of compatible models.

Also handles pulling a structured data type out of the semi-structured response.
"""

import logging
import sys
from collections.abc import Callable
from typing import Any, Union

import examexam.apis.openai_calls as ai
from examexam.apis import anthropic_calls
from examexam.apis.anthropic_calls import AnthropicCaller
from examexam.apis.bedrock_calls import BedrockCaller
from examexam.apis.conversation_model import Conversation, FatalConversationError
from examexam.apis.fakebot_calls import FakeBotCaller, FakeBotException
from examexam.apis.google_calls import GoogleCaller
from examexam.utils.benchmark_utils import log_duration
from examexam.utils.custom_exceptions import ExamExamTypeError, ExamExamValueError

LOGGER = logging.getLogger(__name__)

# Map bot class to specific bot
DEFAULT_BOT = {
    # "titan": "", Waste of money. Nova or go home.
    "jurassic": "",
    "cohere": "",
    # "llama": "", Not worth money unless 3.3
    "mixtral": "",
    # "gpt3.5": "gpt-3.5-turbo-0125",  # 16k context, "gpt-3.5-turbo-1106",
    "gpt4": "gpt-4o-mini",
    # gpt-4-1106-preview and gpt-4-1106-vision-preview <- cheap
    # "gpt4": "gpt-4-0125-preview",  # 128000 tokens <-- expensive
    # "claude-3-opus-20240229",  #  too expensive!!!
    "claude": "claude-3-haiku-20240307",  # "claude-3-sonnet-20240229",  # model="claude-3-haiku-20240307",
    # "gemini-pro": "gemini-1.0-pro-001",
    "fakebot": "fakebot",
}


class Router:

    def __init__(self, conversation: Conversation):
        # Most recent successful

        self.most_recent_python: str | None = None
        self.most_recent_answer: str | None = None
        self.most_recent_json: Union[dict[str, Any], list[Any], None] = None
        self.most_recent_just_code: list[str] | None = None
        self.most_recent_exception: Exception | None = None

        # persistent callers
        self.openai_caller: ai.OpenAICaller | None = None
        self.fakebot_caller: FakeBotCaller | None = None
        self.bedrock_caller: BedrockCaller | None = None
        self.anthropic_caller: anthropic_calls.AnthropicCaller | None = None
        self.standard_conversation: Conversation = conversation
        self.errors_so_far = 0
        """If there are too many errors, stop."""
        self.conversation_cannot_continue = False

    def reset(self) -> None:
        self.most_recent_python = None
        self.most_recent_answer = None
        self.most_recent_json = None
        self.most_recent_just_code = None
        self.most_recent_exception = None

    def call_until(self, request: str, model: str, stop_check: Callable) -> str | None:
        """
        Call the appropriate model based on the model name.
        Args:
            request: The request is the newest user request.
            model: The model to call.
            stop_check: A function that takes the current answer and returns True if the conversation should stop.

        Returns:
            The single string answer from the model.
        """
        answer = self.call(request, model)
        while not stop_check(answer):
            answer = self.call(request, model)
        return answer

    @log_duration
    def call(self, request: str, model: str, essential: bool = False) -> str | None:
        """
        Call the appropriate model based on the model name.
        Args:
            request: The request is the newest user request.
            model: The model to call.
            essential: Is it okay if this exchange fails

        Returns:
            The single string answer from the model.
        """
        if self.conversation_cannot_continue:
            raise ExamExamValueError("Conversation cannot continue, an essential exchange previously failed.")
        if not request:
            raise ExamExamValueError("Request cannot be empty")
        if len(request) < 5:
            LOGGER.warning(
                f"Request ('{request}') is less than 5 characters, unless this is a interactive chat, that is probably wrong."
            )
        self.reset()
        LOGGER.info(f"Calling {model} with request of length {len(request)}")

        # pylint: disable=broad-exception-caught
        try:
            if model == "titan":
                if self.bedrock_caller is None:
                    self.bedrock_caller = BedrockCaller(self.standard_conversation)
                self.standard_conversation = self.bedrock_caller.conversation
                answer = self.bedrock_caller.titan_call(request)
            elif model == "llama":
                if not self.bedrock_caller:
                    self.bedrock_caller = BedrockCaller(conversation=self.standard_conversation)
                self.standard_conversation = self.bedrock_caller.conversation
                answer = self.bedrock_caller.llama2(request)
            elif model == "jurassic":
                prompt = request
                if not self.bedrock_caller:
                    self.bedrock_caller = BedrockCaller(conversation=self.standard_conversation)
                self.standard_conversation = self.bedrock_caller.conversation
                answer = self.bedrock_caller.jurassic_call(prompt)
            elif model == "cohere":
                prompt = request
                if not self.bedrock_caller:
                    self.bedrock_caller = BedrockCaller(conversation=self.standard_conversation)
                self.standard_conversation = self.bedrock_caller.conversation
                answer = self.bedrock_caller.cohere_call(prompt)
            elif model in ("gpt4",):
                # property injection for stateful caller
                if not self.openai_caller:
                    self.openai_caller = ai.OpenAICaller(
                        model=DEFAULT_BOT[model],
                        conversation=self.standard_conversation,
                    )
                self.standard_conversation = self.openai_caller.conversation
                answer = self.openai_caller.completion(request)

            elif model == "claude":
                if not self.anthropic_caller:
                    self.anthropic_caller = anthropic_calls.AnthropicCaller(
                        model=DEFAULT_BOT[model],
                        tokens=4096,  # TODO: Make this a lookup
                        conversation=self.standard_conversation,
                    )
                # this is a chat model
                self.standard_conversation = self.anthropic_caller.conversation
                answer = self.anthropic_caller.single_completion(request)

            elif model == "mixtral":
                if not self.bedrock_caller:
                    self.bedrock_caller = BedrockCaller(conversation=self.standard_conversation)
                self.standard_conversation = self.bedrock_caller.conversation
                answer = self.bedrock_caller.mixtral_8x7b_call(request)
            elif model == "gemini-pro":
                prompt = request
                google_caller = GoogleCaller(
                    model=DEFAULT_BOT[model],
                    conversation=self.standard_conversation,
                )
                answer = google_caller.single_completion(prompt)
            elif model == "fakebot":
                if not self.fakebot_caller:
                    self.fakebot_caller = FakeBotCaller(
                        model=DEFAULT_BOT[model],
                        conversation=self.standard_conversation,
                    )

                self.standard_conversation = self.fakebot_caller.conversation
                answer = self.fakebot_caller.completion(request)

            else:
                raise ExamExamValueError(f"Unknown model {model}")
        except FatalConversationError:
            raise
        except FakeBotException as e:
            if self.standard_conversation:
                self.standard_conversation.error(e)
            # this gets cleared
            self.most_recent_exception = e

            if essential:
                self.conversation_cannot_continue = True
            # If truncated... maybe don't keep going.
            self.errors_so_far += 1
            LOGGER.error(e)
            LOGGER.error(f"Error calling {model} with...{request[:15]}")
            self.most_recent_answer = ""
            return None
        except Exception as e:
            if self.standard_conversation:
                self.standard_conversation.error(e)
            # this gets cleared
            self.most_recent_exception = e

            if essential:
                # "Write test for this code..." Can't skip.
                # "Keep going" Could warrant trying again with different text.
                self.conversation_cannot_continue = True
            # Catch all for all possible API errors, which could be HTTPX, openai client etc.
            if "pytest" in sys.modules:
                # Any error during testing is a code bug, not an API
                raise
            # If truncated... maybe don't keep going.
            self.errors_so_far += 1
            LOGGER.error(e)
            LOGGER.error(f"Error calling {model} with...{request[:15]}")
            self.most_recent_answer = ""
            return None

        self.most_recent_answer = answer
        return answer

    def persist_caller(self, model: str) -> None:
        if model in ("gpt3.5", "gpt4"):
            self.openai_caller = ai.OpenAICaller(
                model=DEFAULT_BOT[model],
                conversation=self.standard_conversation,
            )
        elif model == "fakebot":
            self.fakebot_caller = FakeBotCaller(model=model, conversation=self.standard_conversation)
        elif model in ("mixtral", "titan"):
            self.bedrock_caller = BedrockCaller(conversation=self.standard_conversation)
        elif model == "claude":
            self.anthropic_caller = AnthropicCaller(
                model=DEFAULT_BOT[model],
                tokens=4096,  # TODO: Make this a lookup
                conversation=self.standard_conversation,
            )
        else:
            raise ExamExamTypeError("Unit testing needs a stateful bot, currently just openapi and fakebot")
