# third_party_apis.py
from __future__ import annotations

import logging
import os
import random
from abc import ABC, abstractmethod

import anthropic
import google.generativeai as genai
import openai
from google.generativeai import ChatSession
from retry import retry

# Local application imports
from examexam.apis.types import Conversation, ExamExamTypeError
from examexam.apis.utilities import call_limit, load_env

LOGGER = logging.getLogger(__name__)
load_env()


class FakeBotException(ValueError):
    """Contrived simulation of an API error."""


class BaseLLMCaller(ABC):
    """Abstract base class for all LLM API callers."""

    def __init__(self, model: str, conversation: Conversation):
        self.model = model
        self.conversation = conversation

    @abstractmethod
    def completion(self, prompt: str) -> str:
        """Sends a prompt and returns the completion."""
        raise NotImplementedError


class OpenAICaller(BaseLLMCaller):
    """Handles API calls to OpenAI models."""

    _client = None

    def __init__(self, model: str, conversation: Conversation):
        super().__init__(model, conversation)
        if OpenAICaller._client is None:
            OpenAICaller._client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.client = OpenAICaller._client
        self.supported_models = ["gpt-5", "gpt-4o-mini"]

    @call_limit(500)
    def completion(self, prompt: str) -> str:
        self.conversation.prompt(prompt)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.conversation.conversation,
        )
        if response.usage:
            LOGGER.info(
                f"Tokens used (prompt/completion/total): {response.usage.prompt_tokens}/{response.usage.completion_tokens}/{response.usage.total_tokens}"
            )
        core_response = response.choices[0].message.content or ""
        role = response.choices[0].message.role or ""
        self.conversation.response(core_response, role)
        return core_response


class AnthropicCaller(BaseLLMCaller):
    """Handles API calls to Anthropic models."""

    def __init__(self, model: str, conversation: Conversation, tokens: int):
        super().__init__(model, conversation)
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.tokens = tokens
        self.supported_models = [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-2.1",
            "claude-2.0",
            "claude-instant-1.2",
        ]

    @retry(exceptions=anthropic.RateLimitError, tries=3, delay=5, jitter=(0.15, 0.23), backoff=1.5, logger=LOGGER)
    def completion(self, prompt: str) -> str:
        self.conversation.prompt(prompt)
        try:
            message = self.client.messages.create(
                max_tokens=self.tokens,
                messages=self.conversation.without_system(),
                model=self.model,
                system=self.conversation.system,
            )
            LOGGER.info(f"Actual Anthropic token count {message.usage}")
            core_response = message.content[0].text
            self.conversation.response(core_response)
            return core_response
        except anthropic.RateLimitError as e:
            self.conversation.pop()
            LOGGER.warning(f"Anthropic rate limit hit: {e}. Backing off.")
            raise
        except Exception:
            self.conversation.pop()
            raise


class GoogleCaller(BaseLLMCaller):
    """Handles API calls to Google's Gemini models."""

    _initialized = False

    def __init__(self, model: str, conversation: Conversation):
        super().__init__(model, conversation)
        self._initialize_google()
        self.client = genai.GenerativeModel(model_name=self.model, system_instruction=conversation.system)
        self.chat: ChatSession | None = None
        self.supported_models = ["gemini-1.0-pro-001"]

    def _initialize_google(self):
        if GoogleCaller._initialized:
            return
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            try:
                from google.colab import userdata

                api_key = userdata.get("GOOGLE_API_KEY")
            except ImportError:
                pass  # No key found
        if api_key:
            genai.configure(api_key=api_key)
            GoogleCaller._initialized = True

    def completion(self, prompt: str) -> str:
        self.conversation.prompt(prompt)
        if not self.chat:
            self.chat = self.client.start_chat()

        message = (self.conversation.system or "") + "\n" + prompt
        response = self.chat.send_message(message)
        core_response = response.text
        self.conversation.response(core_response)
        return core_response


class BedrockCaller(BaseLLMCaller):
    """Handles API calls to AWS Bedrock models (Placeholder)."""

    def __init__(self, model: str, conversation: Conversation):
        super().__init__(model, conversation)
        # In a real implementation, the boto3 client would be initialized here.
        # self.client = boto3.client(service_name='bedrock-runtime')
        LOGGER.warning("BedrockCaller is a placeholder and does not make real API calls.")

    def completion(self, prompt: str) -> str:
        self.conversation.prompt(prompt)
        # This would contain logic to invoke the correct model on Bedrock
        # using self.model (e.g., 'amazon.titan-text-express-v1')
        # For example: body = json.dumps({"inputText": prompt})
        # response = self.client.invoke_model(body=body, modelId=self.model)
        LOGGER.info(f"Pretending to call Bedrock model: {self.model}")
        response_text = f"This is a mocked response from Bedrock model {self.model}."
        self.conversation.response(response_text)
        return response_text


class FakeBotCaller(BaseLLMCaller):
    """A fake bot for integration tests and dry runs."""

    def __init__(self, model: str, conversation: Conversation, data: list[str] | None = None, reliable: bool = False):
        super().__init__(model, conversation)
        self.data = data or ["Answers: [1,2]\n---Blah blah. Bad."]
        self.reliable = reliable
        self.invocation_count = 0
        if self.model not in ["fakebot"]:
            raise ExamExamTypeError(f"FakeBotCaller doesn't support model: {self.model}")

    def completion(self, prompt: str) -> str:
        self.invocation_count += 1
        self.conversation.prompt(prompt)

        if not self.reliable and random.random() < 0.1:  # nosec
            raise FakeBotException("Fakebot has failed to return an answer, just like a real API.")

        core_response = random.choice(self.data)  # nosec
        LOGGER.info(f"FakeBot Response: {core_response.replace(chr(10), r' ')}")
        self.conversation.response(core_response)
        return core_response
