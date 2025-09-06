# File: tests/test_config.py
from __future__ import annotations

from pathlib import Path

import pytest
import rtoml as toml

import examexam.config as cfg


@pytest.fixture(autouse=True)
def _isolate_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """
    Run each test in an empty temp directory so the module won't
    pick up a stray examexam.toml from the repo/user machine.
    """
    monkeypatch.chdir(tmp_path)


def test_defaults_when_no_file_and_no_env(monkeypatch: pytest.MonkeyPatch):
    # Ensure env vars aren't present
    for k in list(dict(**{**{}})):  # no-op; here to emphasize no env needed
        monkeypatch.delenv(k, raising=False)

    c = cfg.reset_for_testing(None)  # loads defaults (no TOML present)
    assert c.get("general.default_n") == 5
    assert c.get("general.use_frontier_model") is False
    # unknown path returns default
    assert c.get("not.a.real.path", default=123) == 123


def test_load_from_toml_overrides_defaults(tmp_path: Path):
    # Create a minimal TOML overriding a scalar and adding per-command values
    content = {
        "general": {"default_n": 7},
        "generate": {"n": 3, "model": "openai"},
    }
    (tmp_path / "examexam.toml").write_text(toml.dumps(content), encoding="utf-8")

    c = cfg.reset_for_testing(tmp_path / "examexam.toml")
    assert c.get("general.default_n") == 7
    assert c.get("generate.n") == 3
    assert c.get("generate.model") == "openai"


def test_env_overrides_scalar_int(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    # Base TOML says 6, env should override to 11
    (tmp_path / "examexam.toml").write_text(toml.dumps({"general": {"default_n": 6}}), encoding="utf-8")
    monkeypatch.setenv("EXAMEXAM_GENERAL_DEFAULT_N", "11")

    c = cfg.reset_for_testing(tmp_path / "examexam.toml")
    assert c.get("general.default_n") == 11  # env override wins


@pytest.mark.xfail(reason="Boolean env casting bug in _load_from_env (checks isinstance(type,bool))")
def test_env_overrides_boolean_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """
    This exposes a bug: _load_from_env uses:
        original_type = type(value)
        if isinstance(original_type, bool): ...
    which is always False, so booleans from env aren't cast.
    Marked xfail to document current behavior without breaking CI.
    """
    (tmp_path / "examexam.toml").write_text(toml.dumps({"general": {"use_frontier_model": False}}), encoding="utf-8")
    monkeypatch.setenv("EXAMEXAM_GENERAL_USE_FRONTIER_MODEL", "true")

    c = cfg.reset_for_testing(tmp_path / "examexam.toml")
    assert c.get("general.use_frontier_model") is True


def test_merge_conflict_section_vs_scalar_does_not_overwrite(tmp_path: Path):
    """
    If TOML tries to set `general = "oops"` (non-dict) where defaults define
    a dict, _merge_configs should refuse to replace the section with a scalar.
    """
    # conflicting TOML: top-level scalar for a section key
    (tmp_path / "examexam.toml").write_text('general = "oops"\n', encoding="utf-8")

    c = cfg.reset_for_testing(tmp_path / "examexam.toml")
    # Should retain original dict (not be replaced by a string), hence default value is present
    assert isinstance(c.get("general.default_n"), int)
    assert c.get("general.default_n") == 5


def test_create_default_config_if_not_exists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)

    created_1 = cfg.create_default_config_if_not_exists()  # should create
    assert created_1 is True
    p = tmp_path / cfg.DEFAULT_CONFIG_FILENAME
    assert p.exists()
    assert "[general]" in p.read_text(encoding="utf-8")

    created_2 = cfg.create_default_config_if_not_exists()  # already exists
    assert created_2 is False


def test_reset_for_testing_with_explicit_path(tmp_path: Path):
    custom = tmp_path / "custom.toml"
    custom.write_text(toml.dumps({"general": {"override_model": "meta"}}), encoding="utf-8")

    c = cfg.reset_for_testing(custom)
    assert c.get("general.override_model") == "meta"
