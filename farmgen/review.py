"""Pictures each stage shows Qwen. Missing files are omitted, not searched for."""

from __future__ import annotations

from pathlib import Path

from farmgen.paths import output_dir
from farmgen.scene import Scene
from farmgen.views import ASSET_YAWS, asset_view_filename, compose_view_filename, reference_filenames


def stage_images(stage: str, scene: Scene, reference: Path | None = None) -> tuple[list[str], list[Path]]:
    """Return the ids and PNG paths for one stage review.

    The scene stage requires the four compose views. Other stages return
    whatever pictures that stage has already written.
    """
    out = output_dir(scene.id)
    if stage == "ground":
        images = [out / "ground_kept_top.png", out / "ground_kept_view_0.png"]
        return ["ground"], [path for path in images if path.is_file()]
    if stage == "image":
        ids = []
        images = []
        for asset in scene.assets:
            path = out / "images" / f"{asset.id}.png"
            if path.is_file():
                ids.append(asset.id)
                images.append(path)
        return ids, images
    if stage == "mesh":
        ids = []
        images = []
        for asset in scene.assets:
            path = out / "mesh_thumbs" / f"{asset.id}.png"
            if path.is_file():
                ids.append(asset.id)
                images.append(path)
                photo = Path(asset.image_path) if asset.image_path else None
                if photo is not None and photo.is_file():
                    images.append(photo)
        return ids, images
    if stage == "scale":
        ids = [asset.id for asset in scene.assets]
        images = []
        for asset in scene.assets:
            for yaw in ASSET_YAWS:
                path = out / "thumbs" / asset_view_filename(asset.id, yaw)
                if path.is_file():
                    images.append(path)
        return ids, images
    if stage == "place":
        path = out / "place_top.png"
        ids = [asset.id for asset in scene.assets]
        return ids, [path] if path.is_file() else []
    if stage == "scene":
        images = [out / "compose_top.png"]
        images.extend(out / compose_view_filename(index) for index in range(4))
        missing = [path for path in images[1:] if not path.is_file()]
        if missing:
            raise RuntimeError(f"scene views missing: {missing[0].name}")
        if not images[0].is_file():
            images = images[1:]
        for asset in scene.assets:
            for yaw in ASSET_YAWS:
                thumb = out / "thumbs" / asset_view_filename(asset.id, yaw)
                if thumb.is_file():
                    images.append(thumb)
        if reference is not None:
            for name in reference_filenames():
                path = reference / name
                if path.is_file():
                    images.append(path)
        return [asset.id for asset in scene.assets], images
    raise ValueError(f"unknown review stage {stage}")
