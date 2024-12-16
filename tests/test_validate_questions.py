from examexam.validate_questions import parse_answer


def test_parse_answer():
    # Normal case
    assert parse_answer('Answers: ["a"|"b"|"c"]') == ["a", "b", "c"]


def test_parse_answer_delimiter():
    # Wrong delimiter
    assert parse_answer('Answers: ["a","b","c"]') == ["a", "b", "c"]


def test_parse_answer_degenerate():
    # Degenerate
    assert parse_answer("asdf") == []


def test_parse_answer_with_explanation():
    # Explanation
    assert parse_answer('Answers: ["a"|"b"|"c"] Explnation: blah blah') == ["a", "b", "c"]


def test_parse_answer_with_explanation_csv():
    assert parse_answer('Answers: ["a","b","c"] Explnation: blah blah') == ["a", "b", "c"]
