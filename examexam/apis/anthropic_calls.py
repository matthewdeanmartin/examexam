import logging
import os
import sys
import time

import anthropic
from retry import retry

from examexam.apis.conversation_model import Conversation
from examexam.utils.custom_exceptions import ExamExamTypeError
from examexam.utils.env_loader import load_env

LOGGER = logging.getLogger(__name__)


load_env()

ANTHROPIC_SUPPORTED_MODELS = [
    "claude-3-opus-20240229",  # smartest
    "claude-3-sonnet-20240229",  # 2nd smartest
    "claude-3-haiku-20240307",  # new fast/cheap model
    "claude-2.1",  # smarter last get
    "claude-2.0",  # lat gen
    "claude-instant-1.2",  # least smart/fast/cheap?
]

# Rate limiting
FREE_PLAN = int(60 / 5) + 1
TIER_ONE = int(60 / 50) + 1
TIER_TWO = 0


class AnthropicCaller:
    def __init__(
        self,
        model: str,
        tokens: int,
        conversation: Conversation,
    ):
        if not model:
            raise ExamExamTypeError("Model required.")
        self.client = anthropic.Anthropic(
            # This is the default and can be omitted
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
        )
        self.model = model
        self.supported_models = ANTHROPIC_SUPPORTED_MODELS
        self.tokens = tokens  # e.g. 1024

        self.conversation = conversation

    @retry(exceptions=anthropic.RateLimitError, tries=3, delay=5, jitter=(0.15, 0.23), backoff=1.5, logger=LOGGER)
    def single_completion(self, prompt: str) -> str:
        if not prompt:
            raise ValueError("Prompt cannot be empty")
        self.conversation.prompt(prompt)
        try:
            # https://docs.anthropic.com/claude/reference/rate-limits
            # On the most limited plan, calls are 1x every 12 seconds!
            if "pytest" not in sys.modules:
                # TODO: needs better way to configure this
                time.sleep(TIER_TWO)
            message = self.client.messages.create(
                max_tokens=self.tokens,
                messages=self.conversation.without_system(),  # type: ignore
                model=self.model,
                system=self.conversation.system,
                # temperature: float | NotGiven = NOT_GIVEN,
                # top_k: int | NotGiven = NOT_GIVEN,
                # top_p: float | NotGiven = NOT_GIVEN,
            )
            # `self.client.messages.create()` mock return can be done like this:
            # MagicMock(content=[MagicMock(text="Generated Response")], usage=5))
            LOGGER.info(f"Actual Anthropic token count {message.usage}")
            core_response = message.content[0].text
            self.conversation.response(core_response)
            # ContentBlock has text and type
            return core_response
        except anthropic.RateLimitError as e:
            self.conversation.pop()
            # This should old run in unit tests when an error is simulated.
            if "pytest" not in sys.modules:

                time.sleep(TIER_ONE)
            LOGGER.info(f"Anthropic rate limit {e}")
            LOGGER.warning("A 429 status code was received; we should back off a bit.")
            raise
        except:
            self.conversation.pop()
            raise
