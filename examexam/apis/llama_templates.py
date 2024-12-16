# TODO: https://github.com/flabat/llm-bedrock-meta/tree/8af6d5c170b1ca3f907f0fc79e5cfc2f3ebc4bf6
from examexam.utils.custom_exceptions import ExamExamTypeError

# """<s>[INST] <<SYS>>
# {your_system_message}
# <</SYS>>
#
# {user_message_1} [/INST] {model_reply_1}</s><s>[INST] {user_message_2} [/INST]"""


class LlamaConvo:
    template = """<s>[INST] <<SYS>>
{your_system_message}
<</SYS>>

{user_message_1} [/INST]"""

    def __init__(self, your_system_message: str, user_message: str) -> None:
        if your_system_message is None or user_message is None:
            raise ExamExamTypeError("No args can be None")
        self.your_system_message = your_system_message
        self.user_message = user_message

    def render(self) -> str:
        return self.template.format(your_system_message=self.your_system_message, user_message_1=self.user_message)

    def next(self, history: str, user_reply: str) -> str:
        if not history.endswith("</s><s>[INST]"):
            history += "</s><s>[INST]"
        return history + f" {user_reply} [/INST]"

    def __str__(self) -> str:
        return self.render()
