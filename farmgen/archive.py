"""Numbered copies of the compose stills from one farm run.

Photos stay in ``outputs/<scene>/images``. A run records those paths in
``inputs.json`` instead of copying the PNGs again.
"""

from __future__ import annotations

import json
import re
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


def log_offset(scene_id: str) -> int:
    """Byte length of the scene log, so a stage can keep only the lines it added."""
    path = output_dir(scene_id) / "farmgen.log"
    if not path.is_file():
        return 0
    return path.stat().st_size


def record_stage(
    run_dir: Path,
    stage: str,
    scene: Scene,
    images: list[Path],
    *,
    log_from: int = 0,
    include_verdict: bool = False,
) -> Path:
    """Freeze one stage: the scene, the pictures it used, and the verdict it wrote.

    PNGs are copied. GLBs and the USD stay where the stage left them, and their
    paths are recorded. A later stage can overwrite ``outputs/<scene>`` without
    removing this copy.
    """
    dest = Path(run_dir) / "stages" / stage
    image_dir = dest / "images"
    dest.mkdir(parents=True, exist_ok=True)
    copied = _copy_images(image_dir, images)
    payload = {
        "brief": scene.brief,
        "assets": [
            {
                "id": asset.id,
                "prompt": asset.prompt,
                "size_m": [float(axis) for axis in asset.size_m],
                "achieved_m": None if asset.achieved_m is None else [float(axis) for axis in asset.achieved_m],
                "layout": asset.layout,
                "count": asset.count,
                "field_m": None if asset.field_m is None else [float(axis) for axis in asset.field_m],
                "field_origin": None if asset.field_origin is None else [float(axis) for axis in asset.field_origin],
                "image_path": asset.image_path,
                "mesh_path": asset.mesh_path,
            }
            for asset in scene.assets
        ],
        "images": [str(path) for path in images],
    }
    (dest / "input.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    save_from = scene.model_dump()
    (dest / "scene.json").write_text(json.dumps(save_from, indent=2) + "\n", encoding="utf-8")
    output = {
        "verdict": _verdict_rows(scene.id) if include_verdict else [],
        "redos": [],
        "metrics": _metric_rows(scene.id, log_from),
        "seated": [{"id": asset.id, "count": len(asset.instances)} for asset in scene.assets],
        "images": copied,
    }
    output["redos"] = [row["id"] for row in output["verdict"] if row.get("action")]
    (dest / "output.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    log_line(scene.id, scene.seed, "archive", note="stage", recorded=stage, run_dir=str(dest), files=len(copied))
    return dest


def _copy_images(image_dir: Path, images: list[Path]) -> list[str]:
    copied: list[str] = []
    used: set[str] = set()
    for source in images:
        path = Path(source)
        if not path.is_file() or path.suffix.lower() != ".png":
            continue
        name = path.name
        if name in used:
            name = f"{path.parent.name}_{path.name}"
        used.add(name)
        image_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, image_dir / name)
        copied.append(name)
    return copied


def _verdict_rows(scene_id: str) -> list[dict]:
    path = output_dir(scene_id) / "verdict.json"
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    rows = []
    for entry in payload.get("entries", []):
        if not isinstance(entry, dict) or "id" not in entry:
            continue
        rows.append(
            {
                "id": entry.get("id"),
                "label": entry.get("label"),
                "score": entry.get("score"),
                "action": entry.get("action"),
            }
        )
    return rows


def _metric_rows(scene_id: str, log_from: int) -> list[dict]:
    """Logged factor, flatness, sheet fraction, and seated-count lines from this stage."""
    path = output_dir(scene_id) / "farmgen.log"
    if not path.is_file():
        return []
    text = path.read_bytes()[max(0, log_from) :].decode("utf-8", errors="replace")
    rows = []
    for line in text.splitlines():
        fields = _log_fields(line)
        if not fields:
            continue
        if not any(key in fields for key in ("factor", "flatness", "sheet_fraction", "note")):
            continue
        if "note" in fields and fields["note"] not in {"uniform", "field", "grid", "perimeter", "unplaced", "sheet", "post"}:
            if not any(key in fields for key in ("factor", "flatness", "sheet_fraction")):
                continue
        rows.append(fields)
    return rows


def _log_fields(line: str) -> dict[str, str]:
    if "FARMGEN " not in line:
        return {}
    fields: dict[str, str] = {}
    for match in re.finditer(r'([A-Za-z0-9_]+)=("(?:\\.|[^"])*"|[^\s]+)', line):
        key, value = match.group(1), match.group(2)
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1].replace('\\"', '"')
        fields[key] = value
    return fields


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
