"""Repository paths. Third-party checkouts and outputs stay out of the package."""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def scene_path(scene_id: str) -> Path:
    return repo_root() / "scenes" / scene_id / "scene.json"


def output_dir(scene_id: str) -> Path:
    return repo_root() / "outputs" / scene_id


def asset_cache_dir() -> Path:
    return repo_root() / "outputs" / "asset_cache"


def infinigen_dir() -> Path:
    return repo_root() / "third_party" / "infinigen"


def trellis_dir() -> Path:
    return repo_root() / "third_party" / "TRELLIS.2"


def comfy_dir() -> Path:
    return repo_root() / "third_party" / "ComfyUI"


def cuda_home() -> Path:
    return repo_root() / "toolchains" / "cuda-13.2"
