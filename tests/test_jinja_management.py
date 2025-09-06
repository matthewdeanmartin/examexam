# File: tests/test_jinja_management.py
from __future__ import annotations

import importlib
import importlib.resources
from pathlib import Path

import pytest

import examexam.jinja_management as jm


def test_calculate_hash_stable_and_distinct():
    a = b"hello world"
    b = b"hello world!"
    ha = jm._calculate_hash(a)
    hb = jm._calculate_hash(b)
    assert isinstance(ha, str) and isinstance(hb, str)
    assert len(ha) == 64 and len(hb) == 64  # sha256 hex length
    assert ha != hb
    assert jm._calculate_hash(a) == ha  # stable / deterministic


def test_read_write_hashes_roundtrip(tmp_path: Path):
    hashes_file = tmp_path / "hashes.txt"
    data = {"a.j2": "deadbeef", "b.j2": "cafebabe"}
    jm._write_hashes_file(hashes_file, data)
    out = jm._read_hashes_file(hashes_file)
    # Order is irrelevant; content must match
    assert out == data


def test_deploy_for_customization_initial_copy(tmp_path: Path):
    # Deploy built-in prompts into tmp_path/prompts
    jm.deploy_for_customization(tmp_path)

    dest_prompts = tmp_path / "prompts"
    hashes_path = dest_prompts / jm.HASHES_FILENAME

    assert dest_prompts.is_dir()
    assert hashes_path.is_file()

    # Ensure at least one known template is copied
    copied = sorted(p.name for p in dest_prompts.glob("*.j2"))
    assert "study_guide.md.j2" in copied
    # The hashes file should contain entries for all .j2 files we shipped
    hashes = jm._read_hashes_file(hashes_path)
    for name in copied:
        assert name in hashes
        # hash values should be 64-char hex strings
        assert isinstance(hashes[name], str) and len(hashes[name]) == 64


def test_deploy_skips_modified_without_force(tmp_path: Path):
    # First deploy
    jm.deploy_for_customization(tmp_path)
    dest = tmp_path / "prompts"
    study_file = dest / "study_guide.md.j2"

    # Modify one template locally
    original = study_file.read_text(encoding="utf-8")
    modified = original + "\n{# user modified #}\n"
    study_file.write_text(modified, encoding="utf-8")

    # Redeploy without force: should NOT overwrite modified file
    jm.deploy_for_customization(tmp_path)
    after = study_file.read_text(encoding="utf-8")
    assert after == modified  # preserved

    # Hashes file is still regenerated; ensure it's present
    assert (dest / jm.HASHES_FILENAME).is_file()


def test_deploy_overwrites_with_force(tmp_path: Path):
    # Initial deploy
    jm.deploy_for_customization(tmp_path)
    dest = tmp_path / "prompts"
    study_file = dest / "study_guide.md.j2"
    # Modify local file
    study_file.write_text(study_file.read_text(encoding="utf-8") + "\nLOCAL MOD\n", encoding="utf-8")

    # Capture the authoritative source bytes from the installed package
    src_dir = importlib.resources.files("examexam") / "prompts"
    src_bytes = (src_dir / "study_guide.md.j2").read_bytes()

    # Force deploy should overwrite changes
    jm.deploy_for_customization(tmp_path, force=True)
    final_bytes = study_file.read_bytes()
    assert final_bytes == src_bytes


def test_get_jinja_env_prefers_custom_prompts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Create a custom ./prompts in a clean CWD
    monkeypatch.chdir(tmp_path)
    custom_prompts = tmp_path / "prompts"
    custom_prompts.mkdir(parents=True, exist_ok=True)

    # Provide a custom template that we can detect
    tname = "study_guide.md.j2"
    unique_marker = "## CUSTOM SOURCE (CWD prompts)"
    (custom_prompts / tname).write_text(
        f"# Study Guide for {{ {{ topic }} }}\n\n{unique_marker}\n",
        encoding="utf-8",
    )

    # Re-import module to ensure its global decisions are re-evaluated under this CWD
    m = importlib.reload(jm)

    env = m.get_jinja_env()
    tpl = env.get_template(tname)
    out = tpl.render(topic="pytest")
    assert unique_marker in out


def test_get_jinja_env_dev_prompts_render(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """
    With no ./prompts in CWD, the function should fall back to the repo/dev prompts.
    We don't need to assert the exact loader type—just that we can load a known template.
    """
    monkeypatch.chdir(tmp_path)  # no ./prompts here
    # Do NOT create ./prompts so the function seeks dev or package prompts

    env = jm.get_jinja_env()
    # A template that exists in examexam/prompts
    tpl = env.get_template("study_guide.md.j2")
    rendered = tpl.render(topic="pytest")
    # Should include headings from the shipped template
    assert "## Core Concepts" in rendered or "Core Concepts" in rendered
