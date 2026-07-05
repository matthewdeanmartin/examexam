import importlib
from unittest.mock import MagicMock

import pytest
import rtoml as toml

mod = importlib.import_module("examexam.validate_questions")


SAMPLE_TOML = """
[[questions]]
id = "q-1"
question = "What is the primary purpose of Amazon Athena?"
[[questions.options]]
text = "To perform ad-hoc querying on data stored in S3 using SQL."
explanation = "Correct."
is_correct = true
[[questions.options]]
text = "To manage relational databases on EC2."
explanation = "Incorrect."
is_correct = false
""".strip()


# -------------------------
# read_questions
# -------------------------


def test_read_questions_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    file_path = tmp_path / "exam.toml"
    file_path.write_text(SAMPLE_TOML, encoding="utf-8")

    qs = mod.read_questions(file_path)
    assert isinstance(qs, list)
    assert qs and qs[0]["id"] == "q-1"
    assert len(qs[0]["options"]) == 2
    assert qs[0]["options"][0]["is_correct"] is True


# -------------------------
# parse_answer family
# -------------------------


@pytest.mark.parametrize(
    "answer, expected",
    [
        ("Answers: [alpha | beta]", ["alpha", "beta"]),
        ('Answers: ["alpha" | "beta"]', ["alpha", "beta"]),
        ("Answers: ['alpha' | 'beta']", ["alpha", "beta"]),
        ('Answers: ["alpha", "beta"]', ["alpha", "beta"]),  # comma/quotes path
        ("Answers: [alpha]", ["alpha"]),
        ("Answers: []", [""]),
    ],
)
def test_parse_answer_variants(answer, expected):
    assert mod.parse_answer(answer) == expected


def test_parse_answer_degenerate():
    assert mod.parse_answer("asdf") == []


def test_parse_answer_with_explanation():
    assert mod.parse_answer('Answers: ["a"|"b"|"c"] Explnation: blah blah') == [
        "a",
        "b",
        "c",
    ]


def test_parse_answer_with_explanation_csv():
    assert mod.parse_answer('Answers: ["a","b","c"] Explnation: blah blah') == [
        "a",
        "b",
        "c",
    ]


@pytest.mark.parametrize(
    "text, expected",
    [
        ('["a","b"]', ["a", "b"]),
        ("[a | b]", ["a", "b"]),
        ("[]", [""]),
    ],
)
def test_parse_quote_lists(text, expected):
    # function expects the whole "Answers: ..." but it accepts bracket-only paths internally as well
    out = mod.parse_quote_lists(text)
    assert out == expected


def test_parse_answer_unknown_returns_empty():
    assert mod.parse_answer("Totally unrelated") == []


# -------------------------
# ask_llm / ask_if_bad_question (min-mock)
# -------------------------


class _FakeConversation:
    def __init__(self, system: str):
        self.system = system


class _FakeRouter:
    def __init__(self, conversation):
        self.conversation = conversation
        self._response = None

    def set_response(self, text: str | None):
        self._response = text

    def call(self, prompt: str, model: str):
        return self._response


def test_ask_llm_happy_path(monkeypatch):
    # Monkeypatch the Router symbol in the module under test
    fake_router = _FakeRouter(_FakeConversation(system="S"))
    fake_router.set_response('```toml\nanswers = ["X", "Y"]\n```')

    # Replace both Conversation and Router with fakes (Router is what matters)
    monkeypatch.setattr(mod, "Conversation", _FakeConversation)
    monkeypatch.setattr(mod, "Router", lambda conv: fake_router)

    out = mod.ask_llm(
        "Q without (Select)", ["X", "Y", "Z"], ["X", "Y"], "fakebot", "sys"
    )
    assert out == ["X", "Y"]


def test_ask_llm_none_raises_inconclusive(monkeypatch):
    fake_router = _FakeRouter(_FakeConversation(system="S"))
    fake_router.set_response(None)
    monkeypatch.setattr(mod, "Conversation", _FakeConversation)
    monkeypatch.setattr(mod, "Router", lambda conv: fake_router)

    with pytest.raises(mod.ValidationInconclusive):
        mod.ask_llm("Q", ["A"], ["A"], "fakebot", "sys")


def test_ask_llm_bad_format_raises_inconclusive(monkeypatch):
    fake_router = _FakeRouter(_FakeConversation(system="S"))
    fake_router.set_response("No structured content here")
    monkeypatch.setattr(mod, "Conversation", _FakeConversation)
    monkeypatch.setattr(mod, "Router", lambda conv: fake_router)

    with pytest.raises(mod.ValidationInconclusive):
        mod.ask_llm("Q", ["A"], ["A"], "fakebot", "sys", max_retries=0)


def test_ask_if_bad_question_good_and_bad(monkeypatch):
    # Good path
    fake_router_g = _FakeRouter(_FakeConversation(system="S"))
    fake_router_g.set_response(
        '```toml\nverdict = "good"\nreason = "Because reasons"\n```'
    )
    monkeypatch.setattr(mod, "Conversation", _FakeConversation)
    monkeypatch.setattr(mod, "Router", lambda conv: fake_router_g)
    g, why = mod.ask_if_bad_question("Q", ["A"], ["A"], "fakebot")
    assert g == "good" and "Because reasons" in why

    # Bad path
    fake_router_bad = _FakeRouter(_FakeConversation(system="S"))
    fake_router_bad.set_response('```toml\nverdict = "bad"\nreason = "nope"\n```')
    monkeypatch.setattr(mod, "Router", lambda conv: fake_router_bad)
    g_bad, why_bad = mod.ask_if_bad_question("Q", ["A"], ["A"], "fakebot")
    assert g_bad == "bad" and "nope" in why_bad

    # None -> inconclusive
    fake_router_b = _FakeRouter(_FakeConversation(system="S"))
    fake_router_b.set_response(None)
    monkeypatch.setattr(mod, "Router", lambda conv: fake_router_b)
    with pytest.raises(mod.ValidationInconclusive):
        mod.ask_if_bad_question("Q", ["A"], ["A"], "fakebot")

    # Unexpected format -> inconclusive
    fake_router_u = _FakeRouter(_FakeConversation(system="S"))
    fake_router_u.set_response("No structured content here")
    monkeypatch.setattr(mod, "Router", lambda conv: fake_router_u)
    with pytest.raises(mod.ValidationInconclusive):
        mod.ask_if_bad_question("Q", ["A"], ["A"], "fakebot", max_retries=0)


@pytest.mark.parametrize(
    "txt, expected",
    [
        ("why\n---\nGood", ("good", "why\n")),
        ("explain\n---\nBad", ("bad", "explain\n")),
        ("---\nGood", ("good", "")),
    ],
)
def test_parse_good_bad(txt, expected):
    assert mod.parse_good_bad(txt) == expected


# -------------------------
# grade_test
# -------------------------


def test_grade_test_scores_and_writes(tmp_path):
    questions = [
        {
            "id": "q1",
            "question": "Q1",
            "options": [
                {"text": "A", "is_correct": True},
                {"text": "B", "is_correct": False},
            ],
        },
        {
            "id": "q2",
            "question": "Q2",
            "options": [
                {"text": "X", "is_correct": False},
                {"text": "Y", "is_correct": True},
            ],
        },
    ]
    responses = [["A"], ["Z"]]  # second is wrong
    good_bad = [("good", "ok"), ("bad", "nope")]
    out_file = tmp_path / "graded.toml"

    mock_ui = MagicMock()
    score = mod.grade_test(
        questions, responses, good_bad, out_file, model="fakebot", ui=mock_ui
    )
    assert score == 0.5
    assert out_file.exists()

    data = toml.load(out_file)
    assert "questions" in data and len(data["questions"]) == 2
    # incorrect question should have model answers recorded
    q2 = next(q for q in data["questions"] if q["id"] == "q2")
    assert "fakebot_answers" in q2 and set(q2["fakebot_answers"]) == {"Z"}
    # good/bad annotations present
    assert all("good_bad" in q and "good_bad_why" in q for q in data["questions"])


def test_grade_test_zero_total(tmp_path):
    mock_ui = MagicMock()
    out = mod.grade_test([], [], [], tmp_path / "out.toml", model="fakebot", ui=mock_ui)
    assert out == 0


# -------------------------
# validate_questions_now (E2E-ish)
# -------------------------


def test_validate_questions_now_e2e_minimal(tmp_path, monkeypatch):
    # Prepare input toml
    f = tmp_path / "exam.toml"
    f.write_text(SAMPLE_TOML, encoding="utf-8")

    # Monkeypatch ask_llm / ask_if_bad_question only (keeps parsing/grading real)
    monkeypatch.setattr(
        mod, "ask_llm", lambda q, opts, ans, m, system: ans
    )  # always returns true correct answers
    monkeypatch.setattr(
        mod, "ask_if_bad_question", lambda q, opts, ans, m: ("good", "ok")
    )

    score = mod.validate_questions_now(str(f), model="fakebot")
    assert score == 1.0

    # File updated with normalized content
    data = toml.load(f)
    assert "questions" in data and len(data["questions"]) == 1
    assert data["questions"][0]["good_bad"] == "good"
