from __future__ import annotations

import logging
import re
from collections.abc import Iterable

logger = logging.getLogger(__name__)

try:
    import rtoml as _toml  # fast, lenient
except Exception:  # pragma: no cover
    import tomllib as _toml  # stdlib (py3.11+), stricter


def _try_parse_toml(text: str) -> dict | None:
    """Parse TOML; return dict or None."""
    try:
        return _toml.loads(text)
    except Exception as e:
        logger.debug("TOML parse failed: %s", e)
        return None


def _maybe_fix_quotes(s: str) -> str:
    """Replace common smart quotes with ASCII quotes for a second-chance parse."""
    return s.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")


def _valid_schema(data: dict) -> bool:
    """
    Validate expected schema:

    [[questions]]
    question = "..."
    [[questions.options]]
    text = "..."
    explanation = "..."
    is_correct = true/false
    """
    if not isinstance(data, dict):
        return False
    questions = data.get("questions")
    if not isinstance(questions, list) or not questions:
        return False

    for q in questions:
        if not isinstance(q, dict):
            return False
        if "question" not in q or not isinstance(q["question"], str):
            return False

        options = q.get("options")
        if not isinstance(options, list) or not options:
            return False

        saw_true = False
        for opt in options:
            if not isinstance(opt, dict):
                return False
            if not isinstance(opt.get("text"), str):
                return False
            if not isinstance(opt.get("explanation"), str):
                return False
            if not isinstance(opt.get("is_correct"), bool):
                return False
            if opt["is_correct"]:
                saw_true = True
        # At least one correct per question is a reasonable sanity check
        if not saw_true:
            return False

    return True


_FENCE_RE = re.compile(r"(?ms)^(?:`{3,}|~{3,})\s*([A-Za-z0-9_-]+)?\s*\n(.*?)\n(?:`{3,}|~{3,})\s*$")

_TOML_FENCE_RE = re.compile(
    r"(?ms)^(?:`{3,}|~{3,})\s*(toml)\s*\n(.*?)\n(?:`{3,}|~{3,})\s*$",
    re.IGNORECASE,
)

# Any fenced block (not just TOML) – we will attempt parsing if content looks plausible
_ANY_FENCE_FINDALL_RE = re.compile(r"(?ms)^(?:`{3,}|~{3,})\s*([A-Za-z0-9_-]+)?\s*\n(.*?)\n(?:`{3,}|~{3,})\s*")


# Heuristic for TOML-ish line
def _looks_tomlish(line: str) -> bool:
    line = line.strip()
    if not line:
        return True
    if line.startswith("#"):
        return True
    if line.startswith("[[") and line.endswith("]]"):
        return True
    if line.startswith("[") and line.endswith("]"):
        return True
    # key = value (very loose)
    if "=" in line:
        left, _, right = line.partition("=")
        left = left.strip()
        right = right.strip()
        if left and re.match(r"^[A-Za-z0-9_.-]+$", left):
            # value starts like TOML string/bool/number/array/object
            if (
                right.startswith(('"', "'", "[", "{"))
                or right.lower() in {"true", "false"}
                or re.match(r"^[0-9\.\-]+", right)
            ):
                return True
    return False


def _gather_toml_candidates(markdown: str) -> list[str]:
    """
    Collect possible TOML snippets:
      1) ```toml fenced``` blocks
      2) any fenced blocks whose contents parse as TOML
      3) unfenced regions starting at '[[questions]]' and extending while lines look TOML-ish
    """
    candidates: list[str] = []

    # 1) Explicit TOML fences
    for m in _TOML_FENCE_RE.finditer(markdown):
        content = m.group(2)
        candidates.append(content)

    # 2) Any fence content that *might* be TOML
    for m in _ANY_FENCE_FINDALL_RE.finditer(markdown):
        lang = (m.group(1) or "").lower()
        content = m.group(2)
        # Skip if already captured as explicit TOML
        if lang == "toml":
            continue
        # Try as-is
        candidates.append(content)

    # 3) Unfenced heuristic extraction from first [[questions]] occurrence(s)
    lines = markdown.splitlines()
    indices = [i for i, ln in enumerate(lines) if "[[questions]]" in ln.replace(" ", "")]
    for start in indices:
        # Expand upwards slightly if we started in the middle of a TOML block
        s = start
        while s > 0 and _looks_tomlish(lines[s - 1]):
            # stop if we would cross a fence marker (we don't want to merge across markdown code fences)
            if lines[s - 1].lstrip().startswith("```") or lines[s - 1].lstrip().startswith("~~~"):
                break
            s -= 1
        # Expand downwards while TOML-ish and not hitting a fence close/open
        e = start
        while e < len(lines):
            ln = lines[e]
            stripped = ln.lstrip()
            if stripped.startswith("```") or stripped.startswith("~~~"):
                # Don't cross markdown fences in unfenced scan
                break
            if not _looks_tomlish(ln):
                # allow one non-tomlish line as a gap; stop after a second
                # (helps include trailing blank/comment)
                if e + 1 < len(lines) and _looks_tomlish(lines[e + 1]):
                    e += 1
                    continue
                else:
                    break
            e += 1
        block = "\n".join(lines[s:e]).strip()
        if block and "[[questions]]" in block:
            candidates.append(block)

    # De-dup while preserving order
    seen = set()
    uniq: list[str] = []
    for c in candidates:
        key = c.strip()
        if key and key not in seen:
            uniq.append(key)
            seen.add(key)
    return uniq


def _first_valid_toml(candidates: Iterable[str]) -> str | None:
    """
    Try to parse & validate candidates in order, with a second-chance smart-quote fix.
    Return the original candidate text (not re-serialized) on success.
    """
    for cand in candidates:
        data = _try_parse_toml(cand)
        if data is None:
            fixed = _maybe_fix_quotes(cand)
            if fixed != cand:
                data = _try_parse_toml(fixed)
                if data is not None and _valid_schema(data):
                    logger.info("Recovered TOML by normalizing quotes.")
                    return fixed
        if data is not None and _valid_schema(data):
            return cand
    return None


def extract_questions_toml(markdown_content: str) -> str | None:
    """
    Aggressively extract SMBC-quiz TOML from arbitrary markdown/plaintext.

    Rules:
      - If it can load as TOML and matches the schema, it's TOML.
      - If fenced with a TOML language tag, strip only the outer fence.
      - If fenced with some other/no language, still try to parse.
      - If unfenced, scan from [[questions]] and include TOML-ish lines.
      - Never touch backticks *inside* TOML strings.

    Returns:
      The TOML snippet (without outer code fences), or None if not found/valid.
    """
    if not markdown_content:
        logger.debug("No content; skipping TOML extract.")
        return None

    candidates = _gather_toml_candidates(markdown_content)
    if not candidates:
        logger.debug("No TOML candidates found.")
        return None

    result = _first_valid_toml(candidates)
    if result is None:
        logger.debug("No valid TOML matched expected schema among %d candidates.", len(candidates))
    else:
        logger.info("TOML content found and validated.")
    return result
