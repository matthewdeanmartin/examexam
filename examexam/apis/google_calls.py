import google.generativeai as genai
from google.generativeai import ChatSession

from examexam.apis.conversation_model import Conversation
from examexam.utils.env_loader import load_env

load_env()

INITIALIZED = False


def initialize_google(force_initialization: bool = False):
    # pylint: disable=global-statement
    global INITIALIZED
    if INITIALIZED and not force_initialization:
        return
    try:
        # pylint: disable=import-outside-toplevel
        # Used to securely store your API key
        from google.colab import userdata

        # Or use `os.getenv('API_KEY')` to fetch an environment variable.
        GOOGLE_API_KEY = userdata.get("GOOGLE_API_KEY")
    except ImportError:
        import os

        GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    genai.configure(api_key=GOOGLE_API_KEY)
    INITIALIZED = True


class GoogleCaller:
    def __init__(
        self,
        model: str,
        conversation: Conversation,
    ):
        initialize_google()
        self.model = model
        self.client = genai.GenerativeModel(model_name=self.model, system_instruction=conversation.system)
        self.chat: ChatSession | None = None

        self.supported_models = ["gemini-1.0-pro-001"]
        self.conversation = conversation

    def converse(self, prompt: str) -> str:
        self.conversation.prompt(prompt)
        if not self.chat:
            self.chat = self.client.start_chat()
        response = self.chat.send_message(prompt)
        core_response = response.text

        return core_response

    def single_completion(self, prompt: str) -> str:
        """Suitable for things like classification or simple answer, e.g. what lanuage is this text
        Not suitable for multi-turn conversation.
        """
        self.conversation.prompt(prompt)

        self.chat = self.client.start_chat()
        # enable_automatic_function_calling=True

        message = self.conversation.system + "\n" + prompt
        # TODO: Use conversation object here...
        # message can be list of parts.
        response = self.chat.send_message(message)
        core_response = response.text

        return core_response
