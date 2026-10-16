"""Choose the five pictures Qwen uses as a placement target.

A splat is rendered into those pictures. A reference directory is read only
when no splat was passed. Neither flag means there is no target layout.
"""

from __future__ import annotations

from pathlib import Path

from farmgen.paths import output_dir
from farmgen.views import reference_filenames
from farmgen.workers.splat_render import render_splat


def resolve_reference(
    scene_id: str,
    terrain_m: tuple[float, float],
    splat: Path | None,
    reference_dir: Path | None,
    render_fn=None,
) -> Path | None:
    """Return the directory of five reference PNGs, or None when no flag is set.

    ``--splat`` wins when both flags are passed. A missing name inside
    ``--reference-dir`` raises and names that file.
    """
    if splat is not None:
        destination = output_dir(scene_id) / "reference"
        return render_splat(Path(splat), destination, terrain_m, render_fn=render_fn)
    if reference_dir is None:
        return None
    folder = Path(reference_dir)
    for name in reference_filenames():
        if not (folder / name).is_file():
            raise FileNotFoundError(f"reference image missing: {name}")
    return folder
