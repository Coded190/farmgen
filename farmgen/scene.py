"""Scene contract written before any generator runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

Layout = Literal["unique", "line", "grid", "rows"]


class GroundSpec(BaseModel):
    size_m: tuple[float, float]
    material: str = "tilled_soil"
    terrain_m: tuple[float, float] = (120.0, 120.0)
    relief_m: float = 0.4
    surround_relief_m: float = 4.0

    @field_validator("size_m", "terrain_m")
    @classmethod
    def _positive_extent(cls, value: tuple[float, float]) -> tuple[float, float]:
        if len(value) != 2 or any(axis <= 0 for axis in value):
            raise ValueError("ground extents must be two positive meters [width, depth]")
        return value

    @property
    def plot_m(self) -> tuple[float, float]:
        """Placement rectangle. ``size_m`` stays the stored name so older files load."""
        return self.size_m


class Instance(BaseModel):
    index: int = Field(ge=0)
    xy: tuple[float, float]
    yaw: float
    z: float = 0.0


class AssetSpec(BaseModel):
    id: str
    prompt: str
    size_m: tuple[float, float, float]
    count: int = Field(ge=1)
    layout: Layout = "unique"
    image_path: str | None = None
    mesh_path: str | None = None
    cache_key: str | None = None
    orient: str = "y_up"
    instances: list[Instance] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _id(cls, value: str) -> str:
        if not value or any(ch.isspace() for ch in value):
            raise ValueError("asset id must be a non-empty token")
        return value

    @field_validator("size_m")
    @classmethod
    def _positive_size(cls, value: tuple[float, float, float]) -> tuple[float, float, float]:
        if len(value) != 3 or any(axis <= 0 or axis != axis for axis in value):
            raise ValueError("asset size_m must be three positive finite meters [width, depth, height]")
        return value

    @model_validator(mode="after")
    def _layout_count(self) -> AssetSpec:
        if self.layout == "unique" and self.count != 1:
            raise ValueError(f"{self.id}: unique assets have count 1")
        return self


class Scene(BaseModel):
    id: str = "farm_000"
    seed: int
    brief: str = "a small dairy farm with a barn, silo, fence, and pasture."
    ground: GroundSpec
    assets: list[AssetSpec]

    @model_validator(mode="after")
    def _unique_ids(self) -> Scene:
        ids = [asset.id for asset in self.assets]
        if len(ids) != len(set(ids)):
            raise ValueError("asset ids must be unique")
        return self

    def asset(self, asset_id: str) -> AssetSpec:
        for item in self.assets:
            if item.id == asset_id:
                return item
        raise KeyError(asset_id)


def load_scene(path: Path) -> Scene:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} is not a JSON object")
    _require_size_fields(raw, path)
    try:
        scene = Scene.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"{path} failed the scene contract: {exc}") from exc
    return scene


def save_scene(scene: Scene, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(scene.model_dump_json(indent=2) + "\n", encoding="utf-8")


def _require_size_fields(raw: dict, path: Path) -> None:
    ground = raw.get("ground")
    if not isinstance(ground, dict) or "size_m" not in ground:
        raise ValueError(f"{path} ground is missing size_m")
    assets = raw.get("assets")
    if not isinstance(assets, list):
        raise ValueError(f"{path} assets must be a list")
    for index, asset in enumerate(assets):
        if not isinstance(asset, dict) or "size_m" not in asset:
            name = asset.get("id", index) if isinstance(asset, dict) else index
            raise ValueError(f"{path} asset {name} is missing size_m")


def assets_from_catalog(payload: dict) -> list[AssetSpec]:
    """Accept a Qwen object list only when every entry passes the scene contract."""
    if not isinstance(payload, dict):
        raise ValueError("catalog payload is not an object")
    raw = payload.get("assets")
    if not isinstance(raw, list) or not raw:
        raise ValueError("catalog assets missing")
    assets = [AssetSpec.model_validate(item) for item in raw]
    ids = [asset.id for asset in assets]
    if len(ids) != len(set(ids)):
        raise ValueError("catalog asset ids must be unique")
    return assets


def catalog_or_starter(scene: Scene, ask, attempts: int = 2) -> Scene:
    """Ask for a catalog up to ``attempts`` times, then keep the handwritten list.

    ``ask`` receives the scene brief and returns a JSON object. A bad payload
    or a connection error is one try. It does not start image generation.
    """
    from farmgen.log import log_line

    last = "none"
    for attempt in range(1, attempts + 1):
        try:
            payload = ask(scene.brief)
            scene.assets = assets_from_catalog(payload)
        except (OSError, RuntimeError, TypeError, ValueError, ValidationError) as exc:
            last = str(exc).replace("\n", " ")
            log_line(scene.id, scene.seed, "catalog", attempt=attempt, ok=0, error=last)
            continue
        log_line(scene.id, scene.seed, "catalog", attempt=attempt, ok=1, assets=len(scene.assets))
        return scene
    scene.assets = starter_scene().assets
    log_line(scene.id, scene.seed, "catalog", fallback="starter", error=last, assets=len(scene.assets))
    return scene


def starter_scene() -> Scene:
    """One farm. Unique props are one mesh each. Repeated props are one mesh, many copies."""
    return Scene(
        id="farm_000",
        seed=0,
        ground=GroundSpec(size_m=(40.0, 40.0), material="tilled_soil"),
        assets=[
            AssetSpec(id="barn_01", prompt="a red wooden barn", size_m=(12.0, 8.0, 7.0), count=1, layout="unique"),
            AssetSpec(
                id="silo_01",
                prompt="an exterior corrugated metal farm silo tower with a conical metal roof, no glass and no visible grain",
                size_m=(5.0, 5.0, 12.0),
                count=1,
                layout="unique",
            ),
            AssetSpec(id="tractor_01", prompt="a green farm tractor", size_m=(4.5, 2.2, 2.8), count=1, layout="unique"),
            AssetSpec(id="trough_01", prompt="a metal livestock water trough", size_m=(2.0, 0.6, 0.6), count=1, layout="unique"),
            AssetSpec(id="crate_01", prompt="a wooden farm crate", size_m=(0.8, 0.6, 0.5), count=1, layout="unique"),
            AssetSpec(
                id="fence_01",
                prompt="one wooden farm fence segment with two horizontal rails between two square posts, no vertical pickets",
                size_m=(2.4, 0.2, 1.2),
                count=10,
                layout="line",
            ),
            AssetSpec(
                id="hay_01",
                prompt="a solid rectangular packed hay bale with flat faces and no loose straw strands",
                size_m=(1.2, 0.6, 0.5),
                count=6,
                layout="grid",
            ),
            AssetSpec(
                id="crop_01",
                prompt="a single corn stalk plant with long green leaves growing upright from roots, not a harvested cob",
                size_m=(0.6, 0.6, 1.8),
                count=16,
                layout="rows",
            ),
        ],
    )
