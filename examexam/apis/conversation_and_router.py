# conversation_and_router.py

import logging
import sys
from collections.abc import Callable
from typing import Any, Union

from examexam.apis.third_party_apis import (
    AnthropicCaller,
    BedrockCaller,
    FakeBotCaller,
    FakeBotException,
    GoogleCaller,
    OpenAICaller,
)
from examexam.apis.types import Conversation, ExamExamValueError, FatalConversationError
from examexam.apis.utilities import log_duration

LOGGER = logging.getLogger(__name__)

# Map bot class to specific bot model
DEFAULT_BOT_MODELS = {
    "gpt4": "gpt-4o-mini",
    "claude": "claude-3-haiku-20240307",
    "gemini-pro": "gemini-1.0-pro-001",
    "fakebot": "fakebot",
    # Bedrock models are mapped directly
    "mixtral": "mistral.mixtral-8x7b-instruct-v0:1",
    "titan": "amazon.titan-text-express-v1",
    "llama": "meta.llama2-70b-chat-v1",
    "jurassic": "ai21.j2-ultra-v1",
    "cohere": "cohere.command-text-v14",
}


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
        self.most_recent_json: Union[dict[str, Any], list[Any], None] = None
        self.most_recent_just_code: list[str] | None = None
        self.most_recent_exception: Exception | None = None

        self._caller_map = {
            "gpt4": OpenAICaller,
            "claude": AnthropicCaller,
            "gemini-pro": GoogleCaller,
            "fakebot": FakeBotCaller,
            "mixtral": BedrockCaller,
            "titan": BedrockCaller,
            "llama": BedrockCaller,
            "jurassic": BedrockCaller,
            "cohere": BedrockCaller,
        }

    def reset(self) -> None:
        """Resets the state of the most recent call."""
        self.most_recent_python = None
        self.most_recent_answer = None
        self.most_recent_json = None
        self.most_recent_just_code = None
        self.most_recent_exception = None

    def _get_caller(self, model_key: str) -> Any:
        """Lazily initializes and returns the appropriate API caller."""
        caller_class = self._caller_map.get(model_key)
        if not caller_class:
            raise ExamExamValueError(f"Unknown model {model_key}")

        # Use the class name as the key to store only one instance per caller type
        caller_key = caller_class.__name__
        if caller_key not in self.callers:
            model_id = DEFAULT_BOT_MODELS[model_key]
            if caller_class == AnthropicCaller:
                self.callers[caller_key] = AnthropicCaller(
                    model=model_id, conversation=self.standard_conversation, tokens=4096
                )
            else:
                self.callers[caller_key] = caller_class(model=model_id, conversation=self.standard_conversation)

        # For callers like Bedrock that handle multiple models, update the model ID
        caller_instance = self.callers[caller_key]
        caller_instance.model = DEFAULT_BOT_MODELS[model_key]

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
            raise ExamExamValueError("Conversation cannot continue, an essential exchange previously failed.")
        if not request:
            raise ExamExamValueError("Request cannot be empty")
        if len(request) < 5:
            LOGGER.warning(f"Request ('{request}') is short, which may be inappropriate for non-interactive use.")

        self.reset()
        LOGGER.info(f"Calling {model} with request of length {len(request)}")

        try:
            caller = self._get_caller(model)
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
                raise
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
