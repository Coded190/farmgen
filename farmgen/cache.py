"""Asset cache keyed by prompt, meter size, and TRELLIS resolution."""

from __future__ import annotations

import hashlib
from pathlib import Path

from farmgen.paths import asset_cache_dir


def cache_key(
    prompt: str,
    size_m: tuple[float, ...],
    resolution: int = 512,
    image_sha256: str = "",
) -> str:
    """Key a mesh by prompt, size, resolution, and the photo bytes when those exist.

    An empty ``image_sha256`` keeps the older prompt-and-size key. A new photo
    must pass its hash so TRELLIS does not reuse the previous GLB.
    """
    size_text = ",".join(f"{float(axis):.4f}" for axis in size_m)
    payload = f"{prompt.strip()}|{size_text}|res={int(resolution)}"
    if image_sha256:
        payload += f"|img={image_sha256}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cache_path(key: str) -> Path:
    return asset_cache_dir() / f"{key}.glb"


def cache_hit(key: str) -> bool:
    path = cache_path(key)
    return path.is_file() and path.stat().st_size > 0
