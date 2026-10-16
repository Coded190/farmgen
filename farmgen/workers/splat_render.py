"""Render a 3D Gaussian splat from the compose cameras.

No splat package ships with FarmGen. When one is not importable, the call
raises and names the ply path before placement starts.
"""

from __future__ import annotations

from pathlib import Path

from farmgen.views import reference_filenames


def render_splat(
    ply: Path,
    directory: Path,
    terrain_m: tuple[float, float],
    render_fn=None,
) -> Path:
    """Write the five reference PNGs for ``ply`` and return ``directory``."""
    if not ply.is_file():
        raise FileNotFoundError(f"splat file missing: {ply}")
    directory.mkdir(parents=True, exist_ok=True)
    if render_fn is None:
        render_fn = _render_with_library
    render_fn(ply, directory, terrain_m)
    for name in reference_filenames():
        if not (directory / name).is_file():
            raise RuntimeError(f"splat renderer missing for {ply}: {name}")
    return directory


def _render_with_library(ply: Path, directory: Path, terrain_m: tuple[float, float]) -> None:
    del directory, terrain_m
    raise RuntimeError(f"splat renderer missing for {ply}")
