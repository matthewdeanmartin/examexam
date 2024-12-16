from __future__ import annotations

import logging
import os
import re
import uuid
from datetime import datetime

import dotenv
import rtoml as toml

from examexam.apis import Conversation
from examexam.apis.model_router import Router

dotenv.load_dotenv()


logger = logging.getLogger(__name__)


def create_new_conversation(system_prompt: str) -> Conversation:
    conversation = Conversation(system=system_prompt, mode="full", model="gpt4")
    return conversation


def generate_questions(
    prompt: str, n: int, conversation: Conversation, service: str, model: str
) -> dict[str, list[dict[str, str]]] | None:
    """Function to request questions from OpenAI"""
    toml_content = None
    questions = None

    logger.info("Generating %d questions with prompt: %s", n, prompt)

    toml_schema = """[[questions]]
question = "Question for user here"

[[questions.options]]
text = "Some Correct answer. Must be first."
explanation = "Explanation. Must be before is_correct. Correct."
is_correct = true

[[questions.options]]
text = "Wrong Answer. Must be first."
explanation = "Explanation. Must be before is_correct. Incorrect."
is_correct = false
"""
    # Check time it takes in milliseconds
    time_now = datetime.now()

    prompt = f"""Generate {n} medium difficulty certification exam questions. {prompt}.
    Follow the following TOML format:

    ```toml
    {toml_schema}
    ```
    One or more can be correct! 
    Five options. 
    Each explanation must end in  "Correct" or "Incorrect", e.g. "Instance storage is ephemeral. Correct".
    Do not use numbers or letters to represent the answers.
       [[questions.options]]
       text = "A. Answer"  # never do this.
       [[questions.options]]
       text = "1. Answer"  # never do this.
    Do not use "All of the above" or the like as an answer.
    """

    router = Router(conversation)
    time_then = datetime.now()
    retries = 0
    while not toml_content:
        if retries > 2:
            break
        content = router.call(prompt, model)
        toml_content = extract_toml(content)
        if toml_content is None:
            retries += 1
            continue

        try:
            questions = toml.loads(toml_content)
            retries += 1
        except TypeError as e:
            with open(f"error_{service}.txt", "w", encoding="utf-8") as error_file:
                error_file.write(toml_content)
            logger.error("Error loading TOML content: %s", e)
            continue

    logger.info("Time taken to generate questions: %s", time_then - time_now)
    return questions


def extract_toml(content: str) -> str | None:
    """Function to extract TOML content from a markdown response"""
    if not content:
        return None
    match = re.search(r"```toml\n(.*?)\n```", content, re.DOTALL)
    if match:
        logger.info("TOML content found in response.")
        return match.group(1)
    return None


def save_toml_to_file(toml_content: str, file_name: str) -> None:
    """Save TOML to file"""
    if os.path.exists(file_name):
        with open(file_name, encoding="utf-8") as file:
            existing_content = toml.load(file)
        existing_content["questions"].extend(toml.loads(toml_content)["questions"])
        with open(file_name, "w", encoding="utf-8") as file:
            toml.dump(existing_content, file)
    else:
        with open(file_name, "w", encoding="utf-8") as file:
            file.write(toml_content)
    print(f"TOML content saved to {file_name}")


def generate_questions_now(
    questions_per_toc_topic: int,  # Number of questions to generate
    file_name: str,
    toc_file: str,
    system_prompt: str,
    model: str = "fakebot",
) -> int:
    """Main execution"""
    total_so_far = 0
    with open(toc_file, encoding="utf-8") as file:
        toc = file.readlines()
        toc = [line.strip() for line in toc]
        for service in toc:
            prompt = f"They must all be '{service}' questions."
            conversation = create_new_conversation(system_prompt)

            questions = generate_questions(prompt, questions_per_toc_topic, conversation, service, model)
            if not questions:
                continue

            for question in questions["questions"]:
                # Generate uuid for each question as string
                question["id"] = str(uuid.uuid4())

            total_so_far += len(questions["questions"])
            logger.info("Total questions so far: %d", total_so_far)
            toml_content = toml.dumps(questions)
            save_toml_to_file(toml_content, file_name)
    return total_so_far


if __name__ == "__main__":
    generate_questions_now(
        questions_per_toc_topic=10,
        file_name="personal_multiple_choice_tests.toml",
        toc_file="../example_inputs/personally_important.txt",
        model="gpt4",
        system_prompt="We are writing multiple choice tests.",
    )
