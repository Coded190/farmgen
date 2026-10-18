"""Ledger of assets that already passed a stage.

``keep`` is an accepted mesh: the mesh gate passed and the file is still on
disk. ``skip`` is an accepted photo whose bytes already produced a rejected
mesh (``mesh_image_sha`` matches the file). ``generate`` is the remaining
redo. A later critic pass must not fetch or mesh a keep or a skip again.
"""

from __future__ import annotations

from pathlib import Path

from farmgen.cache import file_sha256
from farmgen.log import log_line
from farmgen.scene import Scene

ACCEPTED_STAGES = ("image", "mesh")


def image_sha(asset) -> str:
    path = Path(asset.image_path) if asset.image_path else None
    if path is None or not path.is_file():
        return ""
    return file_sha256(path)


def mark_accepted(asset, stage: str) -> None:
    """Record a passed stage. A new photo clears a rejection of the previous one."""
    if stage not in ACCEPTED_STAGES:
        raise ValueError(f"unknown stage {stage}")
    if stage == "image":
        sha = image_sha(asset)
        if asset.mesh_image_sha and sha and asset.mesh_image_sha != sha:
            asset.mesh_image_sha = ""
            asset.accepted = [item for item in asset.accepted if item != "mesh"]
    if stage not in asset.accepted:
        asset.accepted.append(stage)


def clear_accepted(asset, stage: str) -> None:
    """Drop a stage. Clearing the photo also clears the mesh that came from it."""
    drop = {stage}
    if stage == "image":
        drop.add("mesh")
        asset.mesh_image_sha = ""
    asset.accepted = [item for item in asset.accepted if item not in drop]


def image_ok(asset) -> bool:
    """True when the photo gate accepted this file and the file is still on disk."""
    return "image" in asset.accepted and bool(image_sha(asset))


def mesh_ok(asset) -> bool:
    """True when the mesh gate accepted this file and the file is still on disk."""
    path = Path(asset.mesh_path) if asset.mesh_path else None
    return "mesh" in asset.accepted and path is not None and path.is_file()


def same_rejected_image(asset) -> bool:
    """True when TRELLIS or the mesh gate already rejected this exact photo."""
    sha = image_sha(asset)
    return bool(sha) and sha == asset.mesh_image_sha


def note_mesh_rejected(asset) -> None:
    """Remember this photo's mesh failed. The photo itself stays accepted."""
    sha = image_sha(asset)
    if sha:
        asset.mesh_image_sha = sha
    asset.accepted = [item for item in asset.accepted if item != "mesh"]
    asset.mesh_path = None
    asset.instances = []


def prepare_generation(scene: Scene, actions) -> tuple[list[str], list[str], list[str]]:
    """Split redo ids into keep, generate, and skip.

    ``redo_image`` clears the photo and goes to generate. ``redo_asset`` keeps
    an accepted mesh, skips a photo that already failed meshing, and generates
    only what is still missing.
    """
    catalog = {asset.id for asset in scene.assets}
    choice: dict[str, str] = {}
    for action in actions:
        if action.action not in {"redo_image", "redo_asset"} or action.id not in catalog:
            continue
        asset = scene.asset(action.id)
        if action.action == "redo_image":
            clear_accepted(asset, "image")
            choice[asset.id] = "generate"
            continue
        if choice.get(asset.id) == "generate":
            continue
        if mesh_ok(asset):
            choice[asset.id] = "keep"
        elif image_ok(asset) and same_rejected_image(asset):
            choice[asset.id] = "skip"
        else:
            choice[asset.id] = "generate"
    keep = [asset.id for asset in scene.assets if choice.get(asset.id) == "keep"]
    generate = [asset.id for asset in scene.assets if choice.get(asset.id) == "generate"]
    skip = [asset.id for asset in scene.assets if choice.get(asset.id) == "skip"]
    if choice:
        log_line(
            scene.id,
            scene.seed,
            "generate",
            keep=keep or "-",
            generate=generate or "-",
            skip=skip or "-",
        )
    return keep, generate, skip
