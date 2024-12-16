from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path
from typing import Any

import rtoml as toml

from examexam.apis import Conversation
from examexam.apis.model_router import Router
from examexam.utils.custom_exceptions import ExamExamTypeError
from examexam.utils.toml_normalize import normalize_exam_for_toml


def read_questions(file_path: Path) -> list[dict[str, Any]]:
    """Reads a TOML file and returns the list of questions."""
    with open(file_path, encoding="utf-8") as file:
        data = toml.load(file)
    return data.get("questions", [])


def ask_llm(question: str, options: list[str], answers: list[str], model: str, system: str) -> list[str]:
    """Asks the LLM to answer a given question."""
    if "(Select" not in question:
        question = f"{question} (Select {len(answers)})"

    prompt = (
        f"Answer the following question in the format 'Answers: [option1 | option2 | ...]'.\n"
        f"Question: {question}\n"
        f"Options: {options}\n"
    )

    conversation = Conversation(system=system, mode="full", model=model)

    router = Router(conversation)
    answer = router.call(prompt, model)
    if answer is None:
        return []

    answer = answer.strip()
    if answer.startswith("Answers:"):
        return parse_answer(answer)
    raise ExamExamTypeError(f"Unexpected response format, didn't start with Answers:, got {answer}")


def parse_answer(answer: str) -> list[str]:
    """Parses the string response from the LLM to extract the answers."""
    if answer.startswith("Answers:"):
        answer = answer[8:]
        if ("','" in answer or "', '" in answer or '","' in answer or '", "' in answer) and "|" not in answer:
            return parse_quote_lists(answer)

        if "[" in answer and "]" in answer:
            after_square_bracket = answer.split("[")[1]
            answer_part = after_square_bracket.split("]")[0]

            answer_part = answer_part.replace('", "', "|").strip('"')
            answers = answer_part.strip().strip("[]").split("|")
            return [ans.strip("'\" ").strip("'\" ") for ans in answers]
    return []


def parse_quote_lists(answer: str) -> list[str]:
    """Helper function to parse comma-separated, quoted lists."""
    if "[" in answer and "]" in answer:
        after_square_bracket = answer.split("[")[1]
        answer_part = after_square_bracket.split("]")[0]

        if "', '" in answer_part or '","' in answer_part:
            answer_part = StringIO(answer_part)
            reader = csv.reader(answer_part, delimiter=",")
            answers = next(reader)
            return answers

        answer_part = answer_part.replace("â€˜", "").replace("â€™", "")

        answer_part = answer_part.replace('", "', "|").strip('"')
        answers = answer_part.strip("[] ").split("|")
        return [ans.strip("'\" ").strip("'\" ") for ans in answers]
    return []


def ask_if_bad_question(
    question: str, options: list[str], answers: list[str], model: str, exam_name: str
) -> tuple[str, str]:
    """Asks the LLM to evaluate if a question is good or bad."""
    prompt = (
        f"Tell me if the following question is Good or Bad, e.g. would it be unfair to ask this on a test.\n"
        f"It is good if it has an answer, if it not every single option is an answer, if it is not opinion based, if it does not have weasel words such as best, optimal, primary which would "
        f"make many of the answers arguably true on some continuum of truth or opinion, or if the question is about *numerical* ephemeral truths, such as system limitations (max GB, etc) and UI defaults.\n\n"
        f"Question: {question}\n"
        f"Options: {options}\n"
        f"Answers: {answers}\n"
        f"\n"
        f"Think about the answer then write `---\nGood` or `---\nBad`\n"
    )

    system = f"You are a test reviewer for the '{exam_name}'."
    conversation = Conversation(system=system, mode="full", model=model)

    router = Router(conversation)
    answer = router.call(prompt, model)
    if answer is None:
        return "bad", "**** Bot returned None, maybe API failed ****"

    answer = answer.strip()
    if "---" in answer:
        return parse_good_bad(answer)
    raise ExamExamTypeError(f"Unexpected response format, didn't contain ---:, got {answer}")


def parse_good_bad(answer):
    """Parses the good/bad response from the LLM."""
    parts = answer.split("---")
    why = parts[0]
    good_bad = parts[1].strip(" \n").lower()
    if "good" in good_bad:
        return "good", why
    return "bad", why


def grade_test(
    questions: list[dict[str, Any]],
    responses: list[list[str]],
    good_bad: list[tuple[str, str]],
    file_path: Path,
    model: str,
) -> float:
    """Grades the LLM's performance against the correct answers."""
    score = 0
    total = len(questions)

    for question, response in zip(questions, responses, strict=True):
        # MODIFICATION: Derive correct answers from the options list using a set comprehension.
        correct_answers = {opt["text"] for opt in question.get("options", []) if opt.get("is_correct")}
        given_answers = set(response)
        if correct_answers == given_answers:
            score += 1
        else:
            print(f"\nQuestion ID: {question['id']}")
            print(f"Question: {question['question']}")
            print(f"Correct Answers: {correct_answers}")
            print(f"Your Answers: {given_answers}")
            question[f"{model}_answers"] = list(given_answers)

    for question, opinion in zip(questions, good_bad, strict=True):
        good_bad_val, why = opinion
        question["good_bad"] = good_bad_val
        question["good_bad_why"] = why

    with open(file_path, "w", encoding="utf-8") as file:
        # toml.dump({"questions": questions}, file)
        toml.dump(normalize_exam_for_toml({"questions": questions}), file)

    print(f"\nFinal Score: {score}/{total}")
    if total == 0:
        return 0
    return score / total


def validate_questions_now(
    file_name: str,
    exam_name: str,
    model: str = "claude",
) -> float:
    """Main function to orchestrate the validation process."""
    file_path = Path(file_name)
    questions = read_questions(file_path)

    responses = []
    opinions = []
    for question_data in questions:
        question = question_data["question"]
        options_list_of_dicts = question_data.get("options", [])

        # MODIFICATION: Extract simple lists of strings from the list of option dicts
        # to pass to the helper functions.
        option_texts = [opt.get("text", "") for opt in options_list_of_dicts]
        correct_answer_texts = [opt.get("text", "") for opt in options_list_of_dicts if opt.get("is_correct")]

        print(f"Submitting question: {question}")
        response = ask_llm(question, option_texts, correct_answer_texts, model, system="You are test evaluator. ")
        print(f"Received answer: {response}")
        responses.append(response)

        good_bad = ask_if_bad_question(question, option_texts, correct_answer_texts, model, exam_name=exam_name)
        opinions.append(good_bad)

    return grade_test(questions, responses, opinions, file_path, model)
