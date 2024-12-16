"""
Facade over invocation for each model.
"""

import json
import logging
import os
from typing import Any

import boto3

import examexam.apis.llama_templates as llama_templates
from examexam.apis.bedrock_models import TitanAnswers, TitanResponse
from examexam.apis.conversation_model import Conversation
from examexam.utils.custom_exceptions import ExamExamTypeError
from examexam.utils.env_loader import load_env

# Configure logging
LOGGER = logging.getLogger(__name__)

load_env()


class BedrockCaller:
    def __init__(self, conversation: Conversation) -> None:

        self.bedrock_runtime_client = boto3.client(
            "bedrock-runtime",
            region_name="us-east-1",
            aws_access_key_id=os.environ["ACCESS_KEY"],
            aws_secret_access_key=os.environ["SECRET_KEY"],
        )
        self.bedrock_client = boto3.client(
            "bedrock",
            region_name="us-east-1",
            aws_access_key_id=os.environ["ACCESS_KEY"],
            aws_secret_access_key=os.environ["SECRET_KEY"],
        )
        self.supported_models = [
            "amazon.titan-text-express-v1",
            "meta.llama2-70b-chat-v1",
            "ai21.j2-ultra-v1",
            "cohere.command-text-v14",
            "mistral.mixtral-8x7b-instruct-v0:1",
        ]
        # we're setting this even if we don't need it.
        self.llama_convo: llama_templates.LlamaConvo | None = None
        # maybe set this later when we know the model
        self.conversation = conversation

    def list_foundation_models(self) -> list[str]:
        response = self.bedrock_client.list_foundation_models()
        models = response["modelSummaries"]
        return models

    def model_info(self, model_id: str) -> Any:
        return self.bedrock_client.get_foundation_model(modelIdentifier=model_id)

    def titan_call(self, prompt: str) -> str:
        self.conversation.prompt(prompt)
        model = "amazon.titan-text-express-v1"
        final_prompt = self.conversation.templatize_conversation()
        response = self.titan(final_prompt, model, max_token_count=7999)
        core_response = response.results[0].output_text.strip("\n ")
        self.conversation.response(core_response)
        return core_response

    def titan(
        self,
        prompt: str,
        model: str,
        max_token_count: int = 512,
        deterministic: bool = False,
    ) -> TitanResponse:

        if max_token_count > 8000:
            raise ExamExamTypeError("max_token_count must be less than 8000")

        if deterministic:
            temperature = 0.0
        else:
            temperature = 0.5

        initial_length = len(prompt)
        max_tokens = 8000
        tokens_in_characters = int((5 * max_tokens) * 0.95)
        if tokens_in_characters < initial_length:
            LOGGER.warning(f"Input text is too long. Truncating to {tokens_in_characters} characters")

        final_prompt = prompt[:tokens_in_characters]
        payload = {
            "inputText": final_prompt,
            "textGenerationConfig": {
                "maxTokenCount": 8000,  # max_token_count,  # up to 8000
                # only 2 stop sequences supported
                "stopSequences": [
                    # "|",
                    # "User:"
                ],
                "temperature": temperature,  # 0 to 1
                "topP": 1.0,  # 0 to 1
            },
        }
        body = json.dumps(payload)

        response = self.bedrock_runtime_client.invoke_model(
            body=body, modelId=model, accept="application/json", contentType="application/json"
        )
        raw = json.loads(response.get("body").read())
        tokens = raw["inputTextTokenCount"]
        response = TitanResponse(
            tokens,
            [
                TitanAnswers(answer.get("tokenCount"), answer.get("outputText"), answer.get("completionReason"))
                for answer in raw["results"]
            ],
        )
        return response

    # # modelId="meta.llama2-70b-chat-v1",

    def llama2(self, prompt: str, deterministic: bool = False) -> str:
        self.conversation.prompt(prompt)
        if self.llama_convo is None:
            self.llama_convo = llama_templates.LlamaConvo(self.conversation.system, "")
        context = self.llama_convo.render()
        prompt = self.llama_convo.next(context, prompt)

        model = "meta.llama2-70b-chat-v1"
        # ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-meta.html

        if deterministic:
            temperature = 0.0
        else:
            temperature = 0.5

        max_tokens = 4096
        initial_length = len(prompt)
        tokens_in_characters = int((3.9 * max_tokens) * 0.90)

        if tokens_in_characters < initial_length:
            LOGGER.warning(f"Input text is too long. Truncating to {tokens_in_characters} characters")

        body = json.dumps(
            {
                "prompt": prompt[:tokens_in_characters],
                "max_gen_len": 2048,  # int, Range: 1-2048
                "temperature": temperature,  # float
                "top_p": 0.9,  # float, like temperature, 0-1, 0.9 default
            }
        )

        response = self.bedrock_runtime_client.invoke_model(
            body=body, modelId=model, accept="application/json", contentType="application/json"
        )
        response_body = json.loads(response.get("body").read())
        # {
        #     "generation": "\n\n<response>",
        #     "prompt_token_count": int,
        #     "generation_token_count": int,
        #     "stop_reason" : string
        # }
        core_response = response_body.get("generation")
        self.conversation.response(core_response)

        return core_response

    def jurassic_call(self, prompt: str, deterministic: bool = False) -> str:
        self.conversation.prompt(prompt)
        model = "ai21.j2-ultra-v1"

        # wrapper for boto for jurassic models:
        # https://pypi.org/project/ai21/

        if deterministic:
            # Response will be identical for identical prompts
            temperature = 0.0
            # set topP too?
        else:
            # default is 0.5
            temperature = 0.5
            # set topP too?

        max_tokens = 5000
        initial_length = len(prompt)
        tokens_in_characters = int((5 * max_tokens) * 0.90)

        if tokens_in_characters < initial_length:
            LOGGER.warning(f"Input text is too long. Truncating to {tokens_in_characters} characters")

        body = json.dumps(
            {
                "prompt": prompt[:tokens_in_characters],
                "maxTokens": max_tokens,  # 8,191 for largest model otherwise 2,048
                "temperature": temperature,  # 0 to 1. 0 is most deterministic
                "topP": 1,  # 0 to 1. 1 is most likely
            }
            # presencePenalty
            # frequencyPenalty
            # "countPenalty": {
            #     "scale": float,
            #     "applyToWhitespaces": boolean,
            #     "applyToPunctuations": boolean,
            #     "applyToNumbers": boolean,
            #     "applyToStopwords": boolean,
            #     "applyToEmojis": boolean
            # }
        )

        response = self.bedrock_runtime_client.invoke_model(
            body=body, modelId=model, accept="application/json", contentType="application/json"
        )

        response_body = json.loads(response.get("body").read())
        core_response = response_body["completions"][0]["data"]["text"]
        self.conversation.response(
            core_response,
        )

        return core_response

    def cohere_call(self, prompt: str, stop_sequences: list[str] | None = None, deterministic: bool = False) -> str:
        self.conversation.prompt(prompt)
        model = "cohere.command-text-v14"

        if deterministic:
            # Will return exact same result for identical prompts
            temperature = 0.0
        else:
            # default is 0.9
            temperature = 1.0

        max_tokens = 4096
        initial_length = len(prompt)
        tokens_in_characters = int((5 * max_tokens) * 0.90)

        if tokens_in_characters < initial_length:
            LOGGER.warning(f"Input text is too long. Truncating to {tokens_in_characters} characters")

        parameters = {
            "prompt": prompt,
            "max_tokens": max_tokens,  # max 4096
            "temperature": temperature,  # 0-5
            "p": 1,  # lower value ignores less likely
            "k": 0,  # 0 to 500, options considered?
            "num_generations": 1,  # shot count. Alternative answers returned
            "return_likelihoods": "NONE",  # "GENERATION|ALL|NONE",
            # "prompt": string,
            # "temperature": float,  # 0 -5
            # "p": float,
            # "k": float,
            # "max_tokens": int,
            # "stop_sequences": [string],
            # "return_likelihoods": "GENERATION|ALL|NONE",
            # "stream": boolean,
            # "num_generations": int,
            # "logit_bias": {token_id: bias}, # avoid/prefer certain words
            # "truncate": "NONE|START|END"
        }
        if stop_sequences:
            parameters["stop_sequences"] = stop_sequences
        body = json.dumps(parameters)

        response = self.bedrock_runtime_client.invoke_model(
            body=body, modelId=model, accept="application/json", contentType="application/json"
        )
        json_text = json.loads(response["body"].read())
        generations = json_text.get("generations")
        core_response = generations[0]["text"]
        self.conversation.response(core_response)
        return core_response

    def mixtral_8x7b_call(self, prompt: str) -> str:
        """
        Invokes the Mixtral 8c7B model to run an inference using the input
        provided in the request body.

        :param prompt: The prompt that you want Mixtral to complete.
        :return: List of inference responses from the model.
        """
        self.conversation.prompt(prompt)
        model = "mistral.mixtral-8x7b-instruct-v0:1"

        # <s>[INST] What is your favorite condiment? [/INST]
        # "Well, I'm quite partial to a good squeeze of fresh lemon juice.
        # It adds just the right amount of zesty flavour to whatever
        # I'm cooking up in the kitchen!"</s> [INST] The right amount of what? [/INST]
        mixtral_prompt = "<s>"
        for exchange in self.conversation.conversation[:-1]:
            if exchange["role"] in ("user", "system"):
                mixtral_prompt += exchange["content"] + " [/INST]\n"
            elif exchange["role"] == "assistant":
                mixtral_prompt += exchange["content"]
        mixtral_prompt += "</s>\n"
        mixtral_prompt += f"[INST] {prompt} [INST]\n"

        # prompt = f"<s>[INST] {prompt} [/INST]"

        # Mistral 7B Instruct – 8,192
        # Mixtral 8X7B Instruct – 4,096
        # TODO: put truncation logic in a function
        max_tokens = 4096
        initial_length = len(prompt)
        tokens_in_characters = int((5 * max_tokens) * 0.90)

        if tokens_in_characters < initial_length:
            LOGGER.warning(f"Input text is too long. Truncating to {tokens_in_characters} characters")
        final_prompt = mixtral_prompt[:tokens_in_characters]

        body = {
            "prompt": final_prompt,
            "max_tokens": max_tokens,
            "temperature": 0.5,  # default 0.5, range 0,1
            # top_p: 0.9,  # default 0.9, range 0,1
            # top_k: 50,  # default 50, range 1,200
            # stop: [], # list of stop sequences, up to 10 (?)
        }

        response = self.bedrock_runtime_client.invoke_model(modelId=model, body=json.dumps(body))

        response_body = json.loads(response["body"].read())
        outputs = response_body.get("outputs")

        completions = [output["text"] for output in outputs]
        core_response = completions[0]
        self.conversation.response(core_response)

        return core_response
