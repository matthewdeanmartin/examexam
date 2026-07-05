from examexam.validate_questions import (
    check_question_form,
    find_duplicate_questions,
    run_deterministic_checks,
)


def _q(question="Q?", options=None):
    if options is None:
        options = [
            {"text": "A", "is_correct": True},
            {"text": "B", "is_correct": False},
        ]
    return {"id": "q1", "question": question, "options": options}


def test_check_question_form_clean_question_has_no_problems():
    assert check_question_form(_q()) == []


def test_check_question_form_empty_text():
    problems = check_question_form(_q(question=""))
    assert any("empty" in p.lower() for p in problems)


def test_check_question_form_too_few_options():
    problems = check_question_form(_q(options=[{"text": "A", "is_correct": True}]))
    assert any("fewer than 2" in p for p in problems)


def test_check_question_form_no_correct_option():
    problems = check_question_form(
        _q(
            options=[
                {"text": "A", "is_correct": False},
                {"text": "B", "is_correct": False},
            ]
        )
    )
    assert any("no option marked is_correct" in p for p in problems)


def test_check_question_form_all_correct():
    problems = check_question_form(
        _q(
            options=[
                {"text": "A", "is_correct": True},
                {"text": "B", "is_correct": True},
            ]
        )
    )
    assert any("no incorrect options" in p for p in problems)


def test_check_question_form_select_count_mismatch():
    problems = check_question_form(
        _q(
            question="Select two of the following.",
            options=[
                {"text": "A", "is_correct": True},
                {"text": "B", "is_correct": False},
                {"text": "C", "is_correct": False},
            ],
        )
    )
    assert any("select two" in p.lower() for p in problems)


def test_check_question_form_select_count_matches():
    problems = check_question_form(
        _q(
            question="Select two of the following.",
            options=[
                {"text": "A", "is_correct": True},
                {"text": "B", "is_correct": True},
                {"text": "C", "is_correct": False},
            ],
        )
    )
    assert not any("select two" in p.lower() for p in problems)


def test_check_question_form_banned_pattern():
    problems = check_question_form(
        _q(
            options=[
                {"text": "A", "is_correct": True},
                {"text": "All of the above", "is_correct": False},
            ]
        )
    )
    assert any("banned pattern" in p for p in problems)


def test_check_question_form_duplicate_option_text():
    problems = check_question_form(
        _q(
            options=[
                {"text": "Same", "is_correct": True},
                {"text": "Same", "is_correct": False},
            ]
        )
    )
    assert any("Duplicate option text" in p for p in problems)


def test_find_duplicate_questions():
    questions = [
        _q(question="What is S3?"),
        _q(question="What is S3?"),
        _q(question="What is EC2?"),
    ]
    dupes = find_duplicate_questions(questions)
    assert dupes == [(0, 1)]


def test_find_duplicate_questions_ignores_punctuation_and_case():
    questions = [
        _q(question="What is S3?!"),
        _q(question="what is s3"),
    ]
    dupes = find_duplicate_questions(questions)
    assert dupes == [(0, 1)]


def test_run_deterministic_checks_combines_form_and_duplicates():
    questions = [
        _q(question="Dup?"),
        _q(question="Dup?"),
        _q(question="", options=[{"text": "A", "is_correct": True}]),
    ]
    result = run_deterministic_checks(questions)
    assert 1 in result  # duplicate flagged on second occurrence
    assert 2 in result  # empty text + too few options
