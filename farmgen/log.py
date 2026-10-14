"""Grep-friendly FarmGen log lines.

Every line starts with ``FARMGEN`` and uses ``key=value`` fields so a failed
asset can be found with ``grep asset_id=barn_01``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_LOG_PATH: Path | None = None


def set_log_path(path: Path | None) -> None:
    global _LOG_PATH
    _LOG_PATH = path
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)


def format_line(scene: str, seed: int, stage: str, **fields: object) -> str:
    parts = [f"FARMGEN scene={scene} seed={seed} stage={stage}"]
    for key, value in fields.items():
        parts.append(f"{key}={_fmt(value)}")
    return " ".join(parts)


def log_line(scene: str, seed: int, stage: str, **fields: object) -> str:
    line = format_line(scene, seed, stage, **fields)
    print(line, flush=True)
    if _LOG_PATH is not None:
        with _LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    return line


def _fmt(value: object) -> str:
    if isinstance(value, str):
        if value == "" or any(ch.isspace() for ch in value) or any(ch in value for ch in '="'):
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        return value
    if isinstance(value, float):
        return f"{value:.4f}" if abs(value) < 1e6 else repr(value)
    if isinstance(value, (list, tuple)):
        return ",".join(_fmt(item) for item in value)
    return str(value)


def write_raw(line: str) -> None:
    if _LOG_PATH is not None:
        with _LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def eprint(message: str) -> None:
    print(message, file=sys.stderr, flush=True)
