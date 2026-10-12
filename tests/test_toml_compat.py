"""Verify exam TOML remains readable across native and pure-Python backends."""

from datetime import datetime, UTC
from io import BytesIO, StringIO
import tomllib

import pytest

from examexam.utils import toml_compat as toml
from examexam.utils.toml_reader import loads_from_string


def test_exam_and_session_roundtrip(tmp_path):
    data = {
        "started_at": datetime(2026, 10, 4, tzinfo=UTC),
        "questions": [{
            "id": "q1",
            "question": "Unicode café and a newline\nsecond line",
            "options": [{"text": "Yes", "is_correct": True}],
        }],
    }
    content = toml.dumps(data)
    # Validate with an independent stdlib parser, not only the writer's backend.
    assert tomllib.loads(content) == data
    path = tmp_path / "exam.toml"
    toml.dump(data, path)
    assert toml.load(path) == data
    for stream in (StringIO(content), BytesIO(content.encode("utf-8"))):
        assert toml.load(stream) == data
        assert not stream.closed
    output = StringIO()
    toml.dump(data, output)
    assert tomllib.loads(output.getvalue()) == data
    assert not output.closed


def test_decode_error_contract():
    with pytest.raises(toml.TomlParsingError):
        toml.loads("broken = [")


def test_unanswered_session_preserves_rtoml_null_convention():
    data = {"values": [None, 1], "questions": [{"id": "q2", "user_score": None}]}
    assert tomllib.loads(toml.dumps(data)) == {
        "questions": [{"id": "q2", "user_score": "null"}], "values": ["null", 1]
    }
    assert data["questions"][0]["user_score"] is None


def test_stdlib_reader_receives_text(monkeypatch):
    import examexam.utils.toml_reader as reader

    monkeypatch.setattr(reader, "BACKEND", "tomllib")
    monkeypatch.setattr(reader, "LOADER", tomllib)
    assert loads_from_string('question = "café"') == {"question": "café"}
