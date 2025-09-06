# conversation_and_router.py
from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from typing import Any

from examexam.apis.third_party_apis import (
    AnthropicCaller,
    BedrockCaller,
    FakeBotCaller,
    FakeBotException,
    GoogleCaller,
    OpenAICaller,
)
from examexam.apis.types import Conversation, ExamExamValueError, FatalConversationError
from examexam.apis.utilities import log_conversation_to_file, log_duration

LOGGER = logging.getLogger(__name__)

# Map bot class to specific bot model
# DEFAULT_BOT_MODELS = {
#     "gpt4": "gpt-5",  # "gpt-4o-mini",
#     "claude": "claude-3-haiku-20240307",
#     "gemini-pro": "gemini-1.0-pro-001",
#     "fakebot": "fakebot",
#     # Bedrock models are mapped directly
#     "mixtral": "mistral.mixtral-8x7b-instruct-v0:1",
#     "titan": "amazon.titan-text-express-v1",
#     "llama": "meta.llama2-70b-chat-v1",
#     "jurassic": "ai21.j2-ultra-v1",
#     "cohere": "cohere.command-text-v14",
# }
# FRONTIER_MODELS = {
#     "fakebot":"fakebot",
#     "openai": "gpt-5",                        # Aug 2025, current flagship
#     "anthropic": "claude-opus-4-1-20250805",  # Aug 2025, strongest Claude
#     "google": "gemini-2.5-pro",               # June 2025, top Gemini
#     "meta": "llama-3.1-405b-instruct",        # July 2025, Meta’s largest model
#     "mistral": "mixtral-8x22b-instruct-v0.1", # 2025 frontier release
#     "cohere": "command-r-plus-08-2025",       # Cohere’s reasoning-tuned flagship
#     "ai21": "jamba-1.5-large",                # AI21’s strongest hybrid model
#     "amazon": "amazon.nova-pro-v1",           # Amazon’s own top Bedrock model
# }
# GOOD_FAST_CHEAP_MODELS
FRONTIER_MODELS = {
    "openai": "gpt-4.1-mini",  # lightweight, fast, inexpensive
    "anthropic": "claude-3.7-sonnet",  # Feb 2025, balance of speed/cost
    "google": "gemini-2.5-flash",  # optimized for speed/cheap inference
    "meta": "llama-3.1-8b-instruct",  # small open-source Llama
    "mistral": "mistral-7b-instruct-v0.3",  # efficient small model
    "cohere": "command-r-08-2025",  # cheaper sibling to “plus”
    "ai21": "jamba-1.5-mini",  # fast, smaller Jamba
    "amazon": "amazon.nova-lite-v1",  # Amazon’s cost-optimized Bedrock model
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
        self.most_recent_json: dict[str, Any] | list[Any] | None = None
        self.most_recent_just_code: list[str] | None = None
        self.most_recent_exception: Exception | None = None

        self._caller_map = {
            "openai": OpenAICaller,
            "anthropic": AnthropicCaller,
            "google": GoogleCaller,
            "fakebot": FakeBotCaller,
            "mistral": BedrockCaller,
            "cohere": BedrockCaller,
            "meta": BedrockCaller,
            "ai21": BedrockCaller,
            "amazon": BedrockCaller,
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
            print(f"unkown model {model_key}")
            sys.exit(-100)
            # raise FatalConversationError(f"Unknown model {model_key}")

        # Use the class name as the key to store only one instance per caller type
        caller_key = caller_class.__name__
        if caller_key not in self.callers:
            model_id = FRONTIER_MODELS[model_key]
            if caller_class == AnthropicCaller:
                self.callers[caller_key] = AnthropicCaller(
                    model=model_id, conversation=self.standard_conversation, tokens=4096
                )
            else:
                self.callers[caller_key] = caller_class(model=model_id, conversation=self.standard_conversation)

        # For callers like Bedrock that handle multiple models, update the model ID
        caller_instance = self.callers[caller_key]
        caller_instance.model = FRONTIER_MODELS[model_key]

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

        caller = None
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
                sys.exit(100)
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
        finally:
            if caller:
                log_conversation_to_file(self.standard_conversation, caller.model, request)

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
