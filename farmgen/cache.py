"""Asset cache keyed by prompt, meter size, and TRELLIS resolution."""

from __future__ import annotations

import hashlib
from pathlib import Path

from farmgen.paths import asset_cache_dir


def cache_key(prompt: str, size_m: tuple[float, ...], resolution: int = 512) -> str:
    size_text = ",".join(f"{float(axis):.4f}" for axis in size_m)
    payload = f"{prompt.strip()}|{size_text}|res={int(resolution)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_path(key: str) -> Path:
    return asset_cache_dir() / f"{key}.glb"


def cache_hit(key: str) -> bool:
    path = cache_path(key)
    return path.is_file() and path.stat().st_size > 0
