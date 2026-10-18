"""Scene contract written before any generator runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

Layout = Literal["unique", "line", "grid", "rows", "surround", "perimeter"]
REQUIRED_WORDS = ("barn", "silo", "tractor", "fence", "tree", "bush", "crop", "pasture", "wheat", "corn")


class GroundSpec(BaseModel):
    size_m: tuple[float, float]
    material: str = "tilled_soil"
    terrain_m: tuple[float, float] = (120.0, 120.0)
    relief_m: float = 0.4
    surround_relief_m: float = 0.6

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
    photo_url: str | None = None
    seen_photo_urls: list[str] = Field(default_factory=list)
    accepted: list[str] = Field(default_factory=list)
    mesh_image_sha: str = ""
    field_m: tuple[float, float] | None = None
    field_origin: tuple[float, float] | None = None
    achieved_m: tuple[float, float, float] | None = None
    instances: list[Instance] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _id(cls, value: str) -> str:
        if not value or any(ch.isspace() for ch in value):
            raise ValueError("asset id must be a non-empty token")
        return value

    @field_validator("accepted")
    @classmethod
    def _known_stages(cls, value: list[str]) -> list[str]:
        unknown = [item for item in value if item not in {"image", "mesh"}]
        if unknown:
            raise ValueError(f"accepted stages must be image or mesh, got {unknown}")
        return list(dict.fromkeys(value))

    @field_validator("size_m", "achieved_m")
    @classmethod
    def _positive_size(cls, value: tuple[float, float, float] | None) -> tuple[float, float, float] | None:
        if value is None:
            return None
        if len(value) != 3 or any(axis <= 0 or axis != axis for axis in value):
            raise ValueError("asset size must be three positive finite meters [width, depth, height]")
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


def plan_extent(asset: AssetSpec) -> tuple[float, float]:
    """Footprint width and depth.

    A fence keeps the catalog length so the perimeter stays a rail. Every other
    asset uses the mesh bbox once scale has measured it.
    """
    if asset.layout != "perimeter" and asset.achieved_m is not None:
        return (float(asset.achieved_m[0]), float(asset.achieved_m[1]))
    return (float(asset.size_m[0]), float(asset.size_m[1]))


def crop_band_y(scene: Scene) -> float | None:
    """North edge of the crop blocks, in plot meters. ``None`` when no block has an origin."""
    tops = [
        float(asset.field_origin[1]) + float(asset.field_m[1])
        for asset in scene.assets
        if asset.field_m is not None and asset.field_origin is not None
    ]
    if not tops:
        return None
    return max(tops)


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


def catalog_text(
    brief: str,
    plot_m: tuple[float, float] = (40.0, 40.0),
    terrain_m: tuple[float, float] = (120.0, 120.0),
    *,
    repair_id: str | None = None,
    error: str | None = None,
    name_from_images: bool = False,
) -> str:
    """The catalog request, including the yard limits a footprint must satisfy."""
    width, depth = float(plot_m[0]), float(plot_m[1])
    ring = min(float(terrain_m[0]) - width, float(terrain_m[1]) - depth) / 2.0 - 2.0
    usable = min(width, depth) - 2.0
    text = (
        "Write a farm asset list as JSON. "
        'Schema: {"assets":[{"id":"barn_01","prompt":"...","size_m":[12,8,7],'
        '"count":1,"layout":"unique"}]}. '
        "layout is unique, line, grid, rows, surround, or perimeter. unique requires count 1. "
        "surround copies sit outside the placement plot. "
        "Each prompt names one object and one or two words of form, such as \"a wooden barn\" or \"a leafy tree\". "
        "Form is a material or a shape, so a search and Flux can tell which object this is. "
        "Do not add a second object, a place, a species, or a list of parts. Do not say where it sits. "
        "Image search and Flux already ask for one centered object on a plain background. "
        "Do not repeat the camera or the background in the prompt. "
        "Do not name a tree and a bush in the same prompt. "
        "A surround asset is one object, such as one tree or one bush, not a ring of them. "
        f"One tree uses about as many copies as the ring can hold, each at most 8 meters. "
        "A fence encloses the yard: perimeter segments of at least 2 meters, not one straight line and not a tiny post. "
        "A field, pasture, or other ground-sized surface is one plant about 1 meter, layout rows, copied across that field. "
        "size_m is width, depth, height in meters. "
        f"The placement plot is {width:.0f} by {depth:.0f} meters. "
        "A surround asset is one repeatable object, with count at least 2, "
        "and each copy at most 8 meters on width and depth. "
        "Do not use one mesh for the whole ring. "
        f"A yard asset must be at most half the plot, {width / 2:.0f} by {depth / 2:.0f} meters, "
        "so other yard objects can sit beside it. "
        f"A surround footprint must also be under {ring:.0f} meters on width and depth. "
        f"A line's width times its count must be under {usable:.0f} meters. "
        f"Request: {brief}"
    )
    if name_from_images:
        text += " The images show the scene. Name the objects in them. Those objects are the asset list."
    if repair_id is not None:
        text += (
            f" Change only {repair_id}. The previous footprint does not fit: {error}. "
            "Return JSON for that one asset and keep its id."
        )
    return text


def catalog_or_starter(
    scene: Scene,
    ask,
    attempts: int = 2,
    images=None,
    record_dir: Path | None = None,
    judge=None,
) -> Scene:
    """Keep Qwen's asset list. Repair one unfit asset instead of substituting the starter list.

    ``ask(prompt)`` or ``ask(prompt, images)`` returns a JSON object, or an object
    with ``payload`` and ``reply``. A connection error or a payload that is not a
    catalog is retried. A footprint that cannot sit in the yard is sent back for
    that id only. When ``judge(brief, assets, images)`` is set, it only accepts or
    rejects the normalized list. A rejected list is generated again. Assets
    already on the scene stay when generation fails or every list is rejected.
    """
    import inspect

    from farmgen.log import log_line

    def _ask(prompt: str):
        parameters = inspect.signature(ask).parameters
        if len(parameters) >= 2 or any(item.kind is inspect.Parameter.VAR_POSITIONAL for item in parameters.values()):
            return ask(prompt, images)
        return ask(prompt)

    pictures = list(images or [])
    original = [asset.model_copy(deep=True) for asset in scene.assets]
    last_reason = ""
    for round_index in range(1, attempts + 1):
        if round_index > 1:
            scene.assets = [asset.model_copy(deep=True) for asset in original]
        passed = _generate_catalog(scene, _ask, attempts, pictures, record_dir)
        if passed is None:
            return scene
        missing = missing_required(scene.brief, scene.assets)
        if missing:
            last_reason = "missing=" + ",".join(missing)
            log_line(scene.id, scene.seed, "catalog", missing=",".join(missing), assets=len(scene.assets))
            _record_catalog_gate(record_dir, False, last_reason, "")
            continue
        if judge is None:
            return scene
        try:
            decision = judge(scene.brief, list(scene.assets), pictures)
        except (OSError, RuntimeError, TypeError, ValueError, ValidationError) as exc:
            accept, reason, reply = False, str(exc).replace("\n", " "), ""
        else:
            accept, reason, reply = _catalog_decision(decision)
        _record_catalog_gate(record_dir, accept, reason, reply)
        log_line(
            scene.id,
            scene.seed,
            "catalog",
            gate=round_index,
            accept=int(accept),
            reason=reason,
            assets=len(scene.assets),
        )
        if accept:
            return scene
        last_reason = reason or "rejected"
    scene.assets = [asset.model_copy(deep=True) for asset in original]
    log_line(scene.id, scene.seed, "catalog", note="rejected", reason=last_reason, assets=len(scene.assets))
    return scene


def _generate_catalog(scene: Scene, ask, attempts: int, pictures: list, record_dir: Path | None) -> dict | None:
    """Replace ``scene.assets`` from Qwen. ``None`` means the call never produced a list."""
    from farmgen.log import log_line

    last = "none"
    passed: dict | None = None
    for attempt in range(1, attempts + 1):
        prompt = catalog_text(
            scene.brief,
            scene.ground.plot_m,
            scene.ground.terrain_m,
            name_from_images=bool(pictures),
        )
        payload = None
        reply = ""
        try:
            payload, reply = _unwrap_catalog(ask(prompt))
            scene.assets = assets_from_catalog(payload)
            passed = {"prompt": prompt, "reply": reply, "payload": payload}
            break
        except (OSError, RuntimeError, TypeError, ValueError, ValidationError) as exc:
            last = str(exc).replace("\n", " ")
            _record_catalog(record_dir, attempt, {"prompt": prompt, "reply": reply, "payload": payload, "error": last})
            log_line(scene.id, scene.seed, "catalog", attempt=attempt, ok=0, error=last)
    if passed is None:
        log_line(scene.id, scene.seed, "catalog", note="kept", error=last, assets=len(scene.assets))
        return None

    for repair in range(1, attempts + 1):
        error = _first_fit_error(scene)
        if error is None:
            normalize_catalog(scene)
            _record_catalog_bundle(record_dir, passed, scene)
            log_line(scene.id, scene.seed, "catalog", attempt=repair, ok=1, assets=len(scene.assets))
            return passed
        asset_id = error.split(" ", 1)[0]
        _record_catalog(record_dir, repair, {"error": error, "assets": [asset.model_dump() for asset in scene.assets]})
        log_line(scene.id, scene.seed, "catalog", repair=repair, id=asset_id, error=error)
        try:
            repaired, _reply = _unwrap_catalog(
                ask(
                    catalog_text(
                        scene.brief,
                        scene.ground.plot_m,
                        scene.ground.terrain_m,
                        repair_id=asset_id,
                        error=error,
                        name_from_images=bool(pictures),
                    )
                )
            )
            _replace_asset(scene, asset_id, repaired)
        except (OSError, RuntimeError, TypeError, ValueError, ValidationError) as exc:
            last = str(exc).replace("\n", " ")
            log_line(scene.id, scene.seed, "catalog", repair=repair, id=asset_id, ok=0, error=last)
    normalize_catalog(scene)
    error = _first_fit_error(scene)
    if error is not None:
        asset_id = error.split(" ", 1)[0]
        scene.assets = [asset for asset in scene.assets if asset.id != asset_id]
        log_line(scene.id, scene.seed, "catalog", id=asset_id, note="unplaceable", error=error)
    _record_catalog_bundle(record_dir, passed, scene)
    log_line(scene.id, scene.seed, "catalog", ok=1, assets=len(scene.assets))
    return passed


def _unwrap_catalog(result: object) -> tuple[object, str]:
    """A catalog call may return the JSON object or ``payload`` plus the raw reply."""
    payload = getattr(result, "payload", None)
    reply = getattr(result, "reply", None)
    if isinstance(payload, dict) and isinstance(reply, str):
        return payload, reply
    return result, ""


def _catalog_decision(decision: object) -> tuple[bool, str, str]:
    if isinstance(decision, bool):
        return decision, "", ""
    reason = getattr(decision, "reason", "")
    reply = getattr(decision, "reply", "")
    return bool(getattr(decision, "accept", False)), "" if reason is None else str(reason), "" if reply is None else str(reply)


def _record_catalog(record_dir: Path | None, attempt: int, payload: object) -> None:
    if record_dir is None or payload is None:
        return
    path = Path(record_dir) / f"catalog_attempt_{attempt}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _record_catalog_bundle(record_dir: Path | None, passed: dict, scene: Scene) -> None:
    """Write the prompt, the raw reply, and the normalized asset list in one file."""
    if record_dir is None:
        return
    path = Path(record_dir) / "catalog_record.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "prompt": passed.get("prompt", ""),
        "reply": passed.get("reply", ""),
        "payload": passed.get("payload"),
        "assets": [asset.model_dump() for asset in scene.assets],
        "missing": missing_required(scene.brief, scene.assets),
    }
    path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")


def _record_catalog_gate(record_dir: Path | None, accept: bool, reason: str, reply: str) -> None:
    if record_dir is None:
        return
    path = Path(record_dir) / "catalog_gate.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"accept": accept, "reason": reason, "reply": reply}, indent=2) + "\n",
        encoding="utf-8",
    )


def _first_fit_error(scene: Scene) -> str | None:
    try:
        _assert_placeable(scene)
    except ValueError as exc:
        return str(exc)
    return None


def _replace_asset(scene: Scene, asset_id: str, payload: dict) -> None:
    """Replace one catalog id. Every other asset stays."""
    if not isinstance(payload, dict):
        raise ValueError("repair payload is not an object")
    if "assets" in payload:
        found = [asset for asset in assets_from_catalog(payload) if asset.id == asset_id]
        if not found:
            raise ValueError(f"repair missing {asset_id}")
        replacement = found[0]
    else:
        payload = dict(payload)
        payload["id"] = asset_id
        replacement = AssetSpec.model_validate(payload)
    scene.assets = [replacement if asset.id == asset_id else asset for asset in scene.assets]


def missing_required(brief: str, assets: list) -> list[str]:
    """Lexicon words in the brief that no asset id or prompt contains after normalization.

    A word that is not in the brief is not required. Matching is a substring of
    the lowercased id and prompt, so ``crops`` satisfies ``crop``.
    """
    text = brief.lower()
    named = [word for word in REQUIRED_WORDS if word in text]
    blob = " ".join(
        f"{getattr(asset, 'id', '')} {getattr(asset, 'prompt', '')}".lower()
        if not isinstance(asset, dict)
        else f"{asset.get('id', '')} {asset.get('prompt', '')}".lower()
        for asset in assets
    )
    return [word for word in named if word not in blob]


def is_ring_phrase(prompt: str) -> bool:
    text = prompt.lower()
    return "ring of" in text or text.startswith("a ring")


def is_field(asset: AssetSpec) -> bool:
    """A photo of the ground itself, rather than one object standing on it."""
    if asset.field_m is not None:
        return False
    prompt = asset.prompt.lower()
    height = float(asset.size_m[2])
    span = max(float(asset.size_m[0]), float(asset.size_m[1]))
    named = any(word in prompt for word in ("field", "pasture", "meadow"))
    if named and "one " not in prompt and span >= 8.0:
        return True
    return height <= 0.5 and span >= 10.0


def needs_perimeter(asset: AssetSpec) -> bool:
    if "fence" not in asset.prompt.lower() or asset.layout == "perimeter":
        return False
    long_side = max(float(asset.size_m[0]), float(asset.size_m[1]))
    return asset.layout == "line" or long_side < 2.0 or asset.count >= 20


def normalize_catalog(scene: Scene) -> None:
    """Turn a field, a line of posts, or a ring prompt into objects we can seat."""
    from farmgen.placer import ring_slots

    slots = ring_slots(scene.ground.plot_m, scene.ground.terrain_m, 8.0)
    forest = [
        asset
        for asset in scene.assets
        if asset.layout == "surround" and (is_ring_phrase(asset.prompt) or _fills_the_ring(asset, len(slots)))
    ]
    trees = [asset for asset in forest if _is_tree_prompt(asset.prompt)]
    bushes = [asset for asset in forest if asset not in trees and _is_bush_prompt(asset.prompt)]
    tree_count = len(slots)
    bush_count = len(slots)
    if trees and bushes:
        bush_count = max(4, len(slots) // 4)
        tree_count = max(2, len(slots) - bush_count)
    for asset in scene.assets:
        if is_field(asset):
            _as_plant_grid(asset)
        elif needs_perimeter(asset):
            _as_perimeter(asset, scene.ground.plot_m)
        elif asset in trees:
            _as_one_object(asset, tree_count)
        elif asset in bushes:
            _as_one_object(asset, bush_count)


def _is_tree_prompt(prompt: str) -> bool:
    return "tree" in prompt.lower()


def _is_bush_prompt(prompt: str) -> bool:
    text = prompt.lower()
    return "tree" not in text and ("bush" in text or "shrub" in text)


def _fills_the_ring(asset: AssetSpec, slots: int) -> bool:
    if not _is_tree_prompt(asset.prompt):
        return False
    return max(float(asset.size_m[0]), float(asset.size_m[1])) >= 7.5 and asset.count < slots


def _as_plant_grid(asset: AssetSpec) -> None:
    area = (float(asset.size_m[0]), float(asset.size_m[1]))
    spacing = 2.0
    cols = max(2, int(area[0] // spacing))
    rows = max(2, int(area[1] // spacing))
    asset.field_m = area
    asset.size_m = (1.0, 1.0, 0.8)
    asset.count = cols * rows
    asset.layout = "rows"
    asset.prompt = "one leafy green pasture plant with a short stem and several leaves, standing upright"


def _as_perimeter(asset: AssetSpec, plot_m: tuple[float, float]) -> None:
    from farmgen.placer import perimeter_samples

    length = 2.4
    thickness = 0.3
    samples = perimeter_samples(plot_m, length, thickness)
    asset.size_m = (length, thickness, 1.2)
    asset.count = len(samples)
    asset.layout = "perimeter"
    asset.prompt = "one wooden farm fence segment with two rails between two posts, standing upright"


def _as_one_object(asset: AssetSpec, slots: int) -> None:
    asset.layout = "surround"
    if _is_bush_prompt(asset.prompt):
        asset.prompt = "one rounded green bush with dense leaves, standing upright"
        asset.size_m = (6.0, 6.0, 2.0)
        asset.count = max(2, slots)
        return
    asset.prompt = "one leafy deciduous tree with a brown trunk and a green canopy, standing upright"
    asset.size_m = (8.0, 8.0, 8.0)
    asset.count = max(2, slots)


def _assert_placeable(scene: Scene) -> None:
    """Reject a catalog whose footprints cannot sit in the plot or the ring."""
    width, depth = scene.ground.plot_m
    ring = min(scene.ground.terrain_m[0] - width, scene.ground.terrain_m[1] - depth) / 2.0 - 2.0
    for asset in scene.assets:
        span_x, span_y = asset.size_m[0], asset.size_m[1]
        if is_ring_phrase(asset.prompt):
            raise ValueError(f"{asset.id} surround must be one object, not a ring")
        if is_field(asset):
            raise ValueError(f"{asset.id} field must be one plant about 1 m, copied across the field")
        if needs_perimeter(asset):
            raise ValueError(f"{asset.id} fence must enclose the yard in segments of at least 2 m")
        if asset.layout == "surround":
            if asset.count < 2:
                raise ValueError(f"{asset.id} surround needs several copies")
            if span_x > 8.0 or span_y > 8.0 or span_x >= ring or span_y >= ring:
                raise ValueError(f"{asset.id} surround size {span_x:.1f}x{span_y:.1f} m does not fit the ring")
            continue
        if asset.layout == "perimeter":
            if max(span_x, span_y) < 2.0:
                raise ValueError(f"{asset.id} fence segment {max(span_x, span_y):.1f} m is too short")
            continue
        half = min(width, depth) / 2.0
        if span_x > half or span_y > half:
            raise ValueError(f"{asset.id} size {span_x:.1f}x{span_y:.1f} m covers the yard")
        if span_x >= width - 2.0 or span_y >= depth - 2.0:
            raise ValueError(f"{asset.id} size {span_x:.1f}x{span_y:.1f} m does not fit the plot")
        if asset.layout == "line" and span_x * asset.count >= width - 2.0:
            raise ValueError(f"{asset.id} line does not fit the plot")


def starter_scene() -> Scene:
    """One farm. Unique props are one mesh each. Repeated props are one mesh, many copies.

    The three tree ids split the 8 m ring lattice, so their copies cover the
    ground outside the 40 m yard. ``tree_02`` and ``tree_03`` are separate
    TRELLIS meshes, not extra copies of ``tree_01``.
    """
    from farmgen.placer import ring_slots

    slots = len(ring_slots((40.0, 40.0), (120.0, 120.0), 8.0))
    each = slots // 3
    tree_counts = (each + slots - 3 * each, each, each)
    return Scene(
        id="farm_000",
        seed=0,
        brief="a small dairy farm with a barn, silo, tractor, fence, pasture crops, and a ring of trees and bushes covering the ground outside the yard.",
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
            AssetSpec(
                id="wheat_01",
                prompt="a patch of upright green wheat stalks with seed heads, growing from the soil",
                size_m=(0.8, 0.8, 1.0),
                count=16,
                layout="rows",
            ),
            AssetSpec(
                id="tree_01",
                prompt="a leafy deciduous oak tree with a brown trunk and a broad green canopy, standing upright",
                size_m=(8.0, 8.0, 10.0),
                count=tree_counts[0],
                layout="surround",
            ),
            AssetSpec(
                id="tree_02",
                prompt="a tall evergreen pine tree with a brown trunk and dense dark green needles, standing upright",
                size_m=(8.0, 8.0, 12.0),
                count=tree_counts[1],
                layout="surround",
            ),
            AssetSpec(
                id="tree_03",
                prompt="a young leafy farm tree with a slim trunk and a light green canopy, standing upright",
                size_m=(8.0, 8.0, 7.0),
                count=tree_counts[2],
                layout="surround",
            ),
            AssetSpec(
                id="bush_01",
                prompt="a low rounded green bush with dense leaves",
                size_m=(1.8, 1.8, 1.4),
                count=24,
                layout="surround",
            ),
            AssetSpec(
                id="bush_02",
                prompt="a flowering shrub with green leaves and small white blossoms",
                size_m=(2.2, 2.2, 1.6),
                count=24,
                layout="surround",
            ),
        ],
    )
