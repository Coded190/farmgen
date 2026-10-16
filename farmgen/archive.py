"""Numbered copies of the compose stills from one farm run.

Photos stay in ``outputs/<scene>/images``. A run records those paths in
``inputs.json`` instead of copying the PNGs again.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from farmgen.log import log_line
from farmgen.paths import output_dir
from farmgen.scene import Scene

_GATE_FOLDERS = ("mesh_gate", "photo_gate")
_STILLS = (
    "compose_top.png",
    "compose.png",
    "compose_eye_a.png",
    "compose_eye_b.png",
    "compose_view_0.png",
    "compose_view_1.png",
    "compose_view_2.png",
    "compose_view_3.png",
    "place_top.png",
    "ground_kept_top.png",
    "ground_kept_view_0.png",
)


def allocate_run_dir(scene_id: str) -> Path:
    """Create the next ``runs/NNNN`` directory. Numbers start at 0001."""
    root = output_dir(scene_id) / "runs"
    root.mkdir(parents=True, exist_ok=True)
    numbers = [int(path.name) for path in root.iterdir() if path.is_dir() and path.name.isdigit()]
    dest = root / f"{(max(numbers) + 1) if numbers else 1:04d}"
    dest.mkdir()
    return dest


def archive_run(scene_id: str, seed: int, run_dir: Path, scene: Scene | None = None) -> None:
    """Copy the compose stills and the gate JSON. Photos stay in the images directory."""
    out = output_dir(scene_id)
    copied = 0
    for name in _STILLS:
        source = out / name
        if source.is_file():
            shutil.copy2(source, run_dir / name)
            copied += 1
    for folder in _GATE_FOLDERS:
        source_dir = out / folder
        if not source_dir.is_dir():
            continue
        dest_dir = run_dir / folder
        dest_dir.mkdir(parents=True, exist_ok=True)
        for source in source_dir.iterdir():
            if source.suffix.lower() != ".json":
                continue
            shutil.copy2(source, dest_dir / source.name)
            copied += 1
    if scene is not None:
        _write_inputs(run_dir, scene)
        copied += 1
    log_path = out / "farmgen.log"
    if log_path.is_file():
        shutil.copy2(log_path, run_dir / "farmgen.log")
    log_line(scene_id, seed, "archive", run_dir=str(run_dir), files=copied)


def _write_inputs(run_dir: Path, scene: Scene) -> None:
    """Point this run at the live photo and mesh. The PNGs are not copied."""
    payload = {
        "assets": [
            {
                "id": asset.id,
                "image_path": asset.image_path,
                "photo_url": asset.photo_url,
                "mesh_path": asset.mesh_path,
            }
            for asset in scene.assets
        ]
    }
    (run_dir / "inputs.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
