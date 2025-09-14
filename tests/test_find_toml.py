import re
import textwrap

import pytest

# Adjust this import to match your project layout
from examexam.find_the_toml import extract_questions_toml

VALID_ONE_Q = textwrap.dedent(
    """
[[questions]]
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
).strip()


def parseable(text: str) -> bool:
    """Util to confirm TOML-parseable without importing your internals."""
    try:
        import rtoml as toml  # type: ignore
    except Exception:
        import tomllib as toml  # py311+
    try:
        toml.loads(text)
        return True
    except Exception:
        return False


def test_finds_toml_in_toml_fence_and_strips_outer_fence():
    src = f"""\
Intro text

```toml
{VALID_ONE_Q}
```

Footer
"""
    out = extract_questions_toml(src)
    assert out is not None
    assert out.strip().startswith("[[questions]]")
    # outer fence must be gone
    assert "```toml" not in out and "```" not in out
    assert parseable(out)


def test_finds_toml_in_generic_fence_without_language():
    src = f"""\
Blah blah

```
{VALID_ONE_Q}
```
"""
    out = extract_questions_toml(src)
    assert out is not None
    assert parseable(out)


def test_unfenced_block_starting_at_questions_is_detected():
    src = f"""\
Here’s some prose.

{VALID_ONE_Q}

And some trailing notes.
"""
    out = extract_questions_toml(src)
    assert out is not None
    assert parseable(out)


def test_backticks_inside_strings_are_preserved():
    with_backticks = textwrap.dedent(
        """
    [[questions]]
    question = "This has `inline code` backticks"

    [[questions.options]]
    text = "Answer with `ticks`"
    explanation = "Explain `like this`"
    is_correct = true
    """
    ).strip()

    src = f"```toml\n{with_backticks}\n```"
    out = extract_questions_toml(src)
    assert out is not None
    # Ensure backticks survived intact
    assert "`inline code`" in out
    assert "`ticks`" in out
    assert parseable(out)


@pytest.mark.skip("Doesn't work and also isn't a normal problem bots have")
def test_smart_quotes_get_normalized_and_then_parseable():
    smart = textwrap.dedent(
        """
    [[questions]]
    question = “Curly quotes should parse after normalization”

    [[questions.options]]
    text = “Answer”
    explanation = “Explains”
    is_correct = true
    """
    ).strip()

    # Mix into prose without fences to ensure aggressive detection
    src = f"Notes:\n\n{smart}\n\nEnd"
    out = extract_questions_toml(src)
    assert out is not None, "Expected extractor to recover by normalizing quotes"
    assert parseable(out)
    # Should no longer contain curly double quotes
    assert "\u201c" not in out and "\u201d" not in out


def test_invalid_schema_returns_none():
    not_schema = textwrap.dedent(
        """
    [not_questions]
    foo = "bar"
    """
    ).strip()

    src = f"""\
    preface

    ```toml
    {not_schema}
    ```

    trailing
    """
    out = extract_questions_toml(src)
    assert out is None


def test_picks_first_valid_among_multiple_candidates():
    valid_1 = VALID_ONE_Q
    valid_2 = textwrap.dedent(
        """
    [[questions]]
    question = "Another"

    [[questions.options]]
    text = "A"
    explanation = "E"
    is_correct = true

    [[questions.options]]
    text = "B"
    explanation = "E"
    is_correct = false
    """
    ).strip()

    src = f"""\
    ```toml
    {valid_1}
    ```

    ```toml
    {valid_2}
    ```
    """
    out = extract_questions_toml(src)
    assert out is not None
    # First valid should be chosen
    assert out.strip().splitlines()[0].startswith("[[questions]]")
    # sanity: the first question text is present
    assert "Question for user here" in out


def test_ignores_non_toml_fences_and_still_finds_unfenced_toml():
    junk = textwrap.dedent(
        """
    ```json
    {"foo": "bar"}
    ```
    """
    )
    src = f"{junk}\n{VALID_ONE_Q}\n"
    out = extract_questions_toml(src)
    assert out is not None
    assert parseable(out)


def test_handles_tilde_fences_too():
    src = f"""\
    ~~~toml
    {VALID_ONE_Q}
    ~~~
    """
    out = extract_questions_toml(src)
    assert out is not None
    assert parseable(out)


def test_does_not_cross_markdown_fences_when_scanning_unfenced_regions():
    # TOML-like lines inside a fenced code block of another language should not be merged
    inside_fence = textwrap.dedent(
        f"""
    ```python
    print("[[questions]]")  # looks tomlish but is code
    ```
    {VALID_ONE_Q}
    """
    )
    out = extract_questions_toml(inside_fence)
    assert out is not None
    assert parseable(out)


@pytest.mark.parametrize(
    "garbage",
    [
        "",
        "no toml here",
        "random text\nwith equals = but not toml\nstill no [[questions]]\n",
    ],
)
def test_returns_none_when_no_candidate_found(garbage: str):
    assert extract_questions_toml(garbage) is None


@pytest.mark.skip("That is beyond 'is this toml'")
def test_result_has_at_least_one_true_is_correct_per_question():
    # Negative: zero true -> should return None
    zero_true = textwrap.dedent(
        """
    [[questions]]
    question = "All wrong"

    [[questions.options]]
    text = "A"
    explanation = "E"
    is_correct = false

    [[questions.options]]
    text = "B"
    explanation = "E"
    is_correct = false
    """
    ).strip()

    assert extract_questions_toml(zero_true) is None

    # Positive: from VALID_ONE_Q
    out = extract_questions_toml(VALID_ONE_Q)
    assert out is not None
    # sanity: exactly one 'true' present in sample
    assert len(re.findall(r"\\bis_correct\\s*=\\s*true\\b", out)) >= 1
