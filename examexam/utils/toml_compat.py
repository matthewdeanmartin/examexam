"""TOML readers and writers with a pure-Python fallback for Python 3.15."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, BinaryIO, TextIO

try:
    import rtoml as _backend
except ImportError:
    _backend = None

if _backend is None:
    import tomllib

    import tomli_w

    TomlParsingError = tomllib.TOMLDecodeError
else:
    TomlParsingError = _backend.TomlParsingError


def loads(content: str) -> dict[str, Any]:
    """Read a TOML string using the available parser."""
    if _backend is not None:
        return _backend.loads(content)
    return tomllib.loads(content)


def load(source: str | Path | TextIO | BinaryIO) -> dict[str, Any]:
    """Read a path or an open text/binary file, preserving caller ownership."""
    if isinstance(source, (str, Path)):
        return loads(Path(source).read_text(encoding="utf-8"))
    content = source.read()
    return loads(content.decode("utf-8") if isinstance(content, bytes) else content)


def dumps(data: Mapping[str, Any]) -> str:
    """Write values as TOML, including nested exam questions and sessions."""
    if _backend is not None:
        return _backend.dumps(data)

    # rtoml writes None as the string "null"; preserve existing session files.
    def normalize(value: Any) -> Any:
        if value is None:
            return "null"
        if isinstance(value, Mapping):
            return {key: normalize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        return value

    return tomli_w.dumps(normalize(data))


def dump(data: Mapping[str, Any], target: str | Path | TextIO) -> None:
    """Write to a path or an open text file without closing caller-owned files."""
    content = dumps(data)
    if isinstance(target, (str, Path)):
        Path(target).write_text(content, encoding="utf-8")
    else:
        target.write(content)
