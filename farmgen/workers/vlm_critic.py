"""Local Qwen client for the asset catalog and the four-image verdict.

The server is OpenAI-compatible on 127.0.0.1:8080. No key file is read.
Tests pass a fake HTTP body or a fake post function.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from farmgen.critic import Verdict, VerdictEntry
from farmgen.gpu import QWEN_BUDGET_MIB, wait_for_vram
from farmgen.paths import repo_root

VLM_URL = "http://127.0.0.1:8080/v1/chat/completions"
VLM_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"
_SERVER: subprocess.Popen | None = None


def _vllm_env() -> dict[str, str]:
    """Point nvcc at CUDA 13.2 so Blackwell (sm_120) kernels can compile.

    The default ``nvcc`` on this machine is CUDA 12.0, which refuses major
    architecture 12. vLLM JIT-compiles those kernels at startup.
    """
    env = os.environ.copy()
    venv_bin = Path(sys.executable).parent
    env["PATH"] = str(venv_bin) + os.pathsep + env.get("PATH", "")
    cuda = repo_root() / "toolchains" / "cuda-13.2"
    if (cuda / "bin" / "nvcc").is_file():
        env["CUDA_HOME"] = str(cuda)
        env["PATH"] = str(cuda / "bin") + os.pathsep + env["PATH"]
    env.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
    return env


def _vllm_executable() -> str:
    """Prefer the venv binary so a run launched as ``python -m farmgen`` still finds it."""
    beside = Path(sys.executable).with_name("vllm")
    if beside.is_file():
        return str(beside)
    found = shutil.which("vllm")
    if found:
        return found
    return "vllm"


def weights_dir() -> Path:
    return repo_root() / "third_party" / "Qwen3-VL-30B-A3B-Instruct"


def stop_server() -> None:
    global _SERVER
    if _SERVER is None:
        return
    _SERVER.terminate()
    try:
        _SERVER.wait(timeout=30)
    except subprocess.TimeoutExpired:
        _SERVER.kill()
    _SERVER = None


def ensure_server() -> None:
    """Start vLLM when the weights are on disk and the port is closed."""
    global _SERVER
    if not (weights_dir() / "config.json").is_file():
        raise RuntimeError(f"qwen weights missing at {weights_dir()}")
    if _port_open():
        return
    wait_for_vram(QWEN_BUDGET_MIB, stop_own=stop_server, on_event=_log_gpu)
    cmd = [
        _vllm_executable(),
        "serve",
        str(weights_dir()),
        "--host",
        "127.0.0.1",
        "--port",
        "8080",
        "--served-model-name",
        VLM_MODEL,
        "--max-model-len",
        "32768",
        "--gpu-memory-utilization",
        "0.90",
        "--moe-backend",
        "triton",
    ]
    log_path = repo_root() / "outputs" / "vllm.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("a", encoding="utf-8")
    log_file.write(f"\n# vllm start {' '.join(cmd)}\n")
    log_file.flush()
    _SERVER = subprocess.Popen(cmd, cwd=repo_root(), env=_vllm_env(), stdout=log_file, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        if _SERVER.poll() is not None:
            code = _SERVER.returncode
            _SERVER = None
            raise RuntimeError(f"vllm exited early with {code}")
        if _port_open():
            return
        time.sleep(2)
    stop_server()
    raise RuntimeError("vllm did not open port 8080")


def request_catalog(prompt: str, images=None, post=None) -> dict:
    ensure_server()
    body = complete(catalog_message(prompt, images), post=post)
    return parse_catalog(body)


def request_mesh_gate(asset_id: str, prompt: str, images: list[Path], post=None):
    """Qwen's accept flag, the reason sentence, and the raw reply text."""
    from farmgen.mesh_gate import MeshDecision

    ensure_server()
    body = complete(mesh_gate_payload(asset_id, prompt, images), post=post)
    decision = parse_mesh_gate(body, asset_id)
    reply = ""
    try:
        reply = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        reply = ""
    if not isinstance(reply, str):
        reply = str(reply)
    return MeshDecision(decision.accept, decision.reason, reply[:4000])


def request_photo_gate(asset_id: str, prompt: str, image: Path, post=None):
    """Qwen's accept flag for one retrieved photo, plus the reason sentence."""
    from farmgen.mesh_gate import MeshDecision

    ensure_server()
    body = complete(photo_gate_payload(asset_id, prompt, image), post=post)
    decision = parse_mesh_gate(body, asset_id)
    reply = ""
    try:
        reply = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        reply = ""
    if not isinstance(reply, str):
        reply = str(reply)
    return MeshDecision(decision.accept, decision.reason, reply[:4000])


def photo_gate_payload(asset_id: str, prompt: str, image: Path) -> dict:
    text = (
        f"The picture should show one {prompt}. "
        "Accept when that one whole object is the clear subject. "
        "A plain background or a simple outdoor background such as a field or sky is acceptable. "
        "A stock watermark by itself is not a reason to reject. "
        "Reject a person, a hand, a second dominant object, a close-up of a part, a collage, or the wrong object. "
        "Do not require other farm objects, a ring, or a particular placement. "
        "The reason must be your own sentence about this picture. "
        f'Reply with JSON {{"id":"{asset_id}","accept":true,"reason":""}} '
        "and fill reason. Do not leave it blank and do not copy this instruction."
    )
    return {
        "model": VLM_MODEL,
        "temperature": 0,
        "messages": [{"role": "user", "content": [{"type": "text", "text": text}, _image_content(image)]}],
    }


def request_plant_box(asset_id: str, prompt: str, image: Path, post=None) -> tuple[float, float, float, float] | None:
    """One plant rectangle in a field photo, as normalized image coordinates."""
    ensure_server()
    body = complete(plant_box_payload(asset_id, prompt, image), post=post)
    return parse_plant_box(body, asset_id)


def plant_box_payload(asset_id: str, prompt: str, image: Path) -> dict:
    text = (
        f"The picture is a field. Mark one {prompt}. "
        "u and v are 0 to 1, with v 0 at the top. The box must contain that one plant and not the whole field. "
        f'Reply with JSON {{"id":"{asset_id}","u0":0.4,"v0":0.4,"u1":0.6,"v1":0.6}}.'
    )
    return {
        "model": VLM_MODEL,
        "temperature": 0,
        "messages": [{"role": "user", "content": [{"type": "text", "text": text}, _image_content(image)]}],
    }


def parse_plant_box(body: dict, asset_id: str) -> tuple[float, float, float, float] | None:
    try:
        text = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None
    if not isinstance(text, str):
        text = str(text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if str(payload.get("id", asset_id)) != asset_id:
        return None
    try:
        return (float(payload["u0"]), float(payload["v0"]), float(payload["u1"]), float(payload["v1"]))
    except (KeyError, TypeError, ValueError):
        return None


def request_verdict(kind: str, ids: list[str], images: list[Path], post=None) -> Verdict:
    ensure_server()
    body = complete(verdict_payload(kind, ids, images), post=post)
    return parse_verdict(body)


def poses_payload(
    ids: list[str],
    image: Path,
    required: list[tuple[str, int]] | None = None,
    problems: list[tuple[str, int, str]] | None = None,
) -> dict:
    """Ask for one pose per instance, as a point on the top image."""
    copies = ""
    if required:
        grouped: dict[str, list[int]] = {}
        for asset_id, index in required:
            grouped.setdefault(asset_id, []).append(index)
        copies = " Pose every listed copy: " + "; ".join(
            f"{asset_id} index {','.join(str(index) for index in indexes)}" for asset_id, indexes in grouped.items()
        ) + "."
    prompt = (
        "The image is an orthographic top view. The brown square is the yard. "
        "The green area around it is the ring. "
        "Place every copy on its own point. Do not give every asset the same u and v. "
        "A yard asset stays inside the brown square, apart from the other yard assets. "
        "A surround asset goes in the green ring, outside the brown square. "
        "Reply with JSON "
        '{"poses":[{"id":"barn_01","index":0,"u":0.50,"v":0.50,"yaw_deg":0},'
        '{"id":"tree_01","index":0,"u":0.20,"v":0.20,"yaw_deg":0}]}. '
        "u and v are 0 to 1 on this image. u 0 is the left edge. v 0 is the top edge. "
        "0.50,0.50 is the yard center and is only for one yard asset. "
        "yaw_deg is degrees about Z. One pose per copy, so index runs from 0 to count-1. "
        f"Ids: {', '.join(ids)}.{copies}"
    )
    if problems:
        rejected = "; ".join(f"{asset_id} index {index} {reason}" for asset_id, index, reason in problems)
        prompt += f" Previous points were rejected: {rejected}. Choose a different point."
    return {
        "model": VLM_MODEL,
        "temperature": 0,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, _image_content(image)]}],
    }


def request_poses(
    ids: list[str],
    image: Path,
    required: list[tuple[str, int]] | None = None,
    problems: list[tuple[str, int, str]] | None = None,
    post=None,
) -> list[dict]:
    ensure_server()
    body = complete(poses_payload(ids, image, required, problems), post=post)
    return parse_poses(body)


def parse_poses(body: dict) -> list[dict]:
    payload = _json_from_message(body)
    poses = payload.get("poses")
    if not isinstance(poses, list):
        raise ValueError("poses missing")
    return [item for item in poses if isinstance(item, dict)]


def catalog_message(prompt: str, images=None) -> dict:
    pictures = [Path(image) for image in (images or [])]
    if pictures:
        content: str | list[dict] = [{"type": "text", "text": prompt}]
        content.extend(_image_content(image) for image in pictures)
    else:
        content = prompt
    return {
        "model": VLM_MODEL,
        "temperature": 0,
        "messages": [{"role": "user", "content": content}],
    }


def catalog_payload(brief: str, plot_m=(40.0, 40.0), terrain_m=(120.0, 120.0), images=None) -> dict:
    from farmgen.scene import catalog_text

    text = catalog_text(brief, plot_m, terrain_m, name_from_images=bool(images))
    return catalog_message(text, images)


def mesh_gate_payload(asset_id: str, prompt: str, images: list[Path]) -> dict:
    text = (
        "The pictures before the last are orthogonal views of one isolated mesh, stood upright. "
        "The last picture is the photo that mesh was built from. These pictures are not the farm scene. "
        f"Accept the mesh only when these views show one {prompt}. "
        "Reject a forest, a collage, a logo, a flat card, or the wrong object. "
        "The reason must be your own sentence about the object in the first picture. "
        f'Reply with JSON {{"id":"{asset_id}","accept":true,"reason":""}} '
        "and fill reason. Do not leave it blank and do not copy this instruction."
    )
    content: list[dict] = [{"type": "text", "text": text}]
    for image in images:
        content.append(_image_content(Path(image)))
    return {
        "model": VLM_MODEL,
        "temperature": 0,
        "messages": [{"role": "user", "content": content}],
    }


def parse_mesh_gate(body: dict, asset_id: str):
    """Return accept plus the reason sentence. A missing reason is an empty string."""
    from farmgen.mesh_gate import MeshDecision

    payload = _json_from_message(body)
    if "accept" not in payload and "assets" not in payload and "entries" not in payload:
        raise ValueError("mesh gate accept missing")
    reason = payload.get("reason", "")
    if not isinstance(reason, str):
        reason = str(reason)
    if str(payload.get("id", asset_id)) not in {asset_id, ""}:
        return MeshDecision(False, reason or "id mismatch")
    value = payload.get("accept")
    if isinstance(value, str):
        accepted = value.strip().lower() == "true"
    else:
        accepted = bool(value)
    return MeshDecision(accepted, reason)


def _image_content(path: Path) -> dict:
    """Send a small copy so four views plus every yaw thumb fit in one call."""
    from io import BytesIO

    from PIL import Image

    image = Image.open(path).convert("RGB")
    image.thumbnail((384, 384))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.standard_b64encode(buffer.getvalue()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}


def verdict_payload(kind: str, ids: list[str], images: list[Path]) -> dict:
    facing = ""
    if kind == "place":
        prompt = (
            "Judge placement only. Reply with JSON "
            '{"entries":[{"id":"...","label":"ok","action":null,"score":0,"index":0,"u":0.5,"v":0.5,"yaw_deg":0}]}. '
            "The brown square is the yard. The green area is the ring. "
            "score is 0 to 10. A score of 4 or below means acceptable, except wrong_place. "
            "label is ok or wrong_place. action is null or move. "
            "A yard asset that shares another asset's ground point is wrong_place. "
            "A surround asset inside the yard, or absent from the green ring, is wrong_place. "
            "wrong_place requires action move, even when the score is 4 or below. "
            "A move includes index, u, v, and yaw_deg. u and v are 0 to 1 on the top view. "
            "u 0 is the left edge and v 0 is the top edge. "
            "Mesh shape was judged earlier. Do not use redo_asset. "
            f"Ids: {', '.join(ids)}."
        )
        content: list[dict] = [{"type": "text", "text": prompt}]
        for image in images:
            content.append(_image_content(Path(image)))
        return {
            "model": VLM_MODEL,
            "temperature": 0,
            "messages": [{"role": "user", "content": content}],
        }
    if kind == "scene":
        facing = (
            " Also judge heading about Z for every id. "
            "A realistic heading means a barn door faces the open yard, "
            "a tractor front points along open ground rather than into a building or fence, "
            "a fence's long side follows its line, crop rows share a heading along the row, "
            "and a round tree or bush is ok unless a flat face points unnaturally at the yard. "
            "The ring outside the yard should be nearly filled with trees. "
            "If a tree or bush mesh is the wrong object, or the ring looks empty, "
            "label missing, action redo_asset, and score 8 so that mesh is generated again. "
            "If the heading is unrealistic, label wrong_yaw, action turn, and yaw_deg one of 0, 90, 180, 270. "
            "yaw_deg is the heading to store, not a delta. "
            "wrong_orientation is only for an asset that is not upright, and its action is reorient. "
            "Return one entry for every id. "
        )
    prompt = (
        f"Judge this farm {kind}. Reply with JSON "
        '{"entries":[{"id":"...","label":"ok","action":null,"score":0,"yaw_deg":null}]}. '
        "score is 0 to 10. A score of 4 or below means acceptable. "
        "label is ok, missing, wrong_scale, floating, overlapping, "
        "not_agricultural, wrong_orientation, wrong_yaw, or wrong_place. "
        "action is null, redo_ground, redo_asset, reorient, redo_image, redo_place, turn, or move. "
        "wrong_orientation uses reorient. A bad generated picture uses redo_image. "
        "A bad mesh uses redo_asset. A bad arrangement that should be placed again uses redo_place. "
        "A wrong location uses move, with index, u and v from 0 to 1 on the orthographic top image, "
        "and yaw_deg in degrees. u 0 is the left edge and v 0 is the top edge. "
        "index is the copy, starting at 0. A wrong mesh uses redo_asset and does not include a pose. "
        "The ground id is ground and its action is redo_ground. "
        f"{facing}"
        f"Ids: {', '.join(ids)}."
    )
    content: list[dict] = [{"type": "text", "text": prompt}]
    for image in images:
        content.append(_image_content(Path(image)))
    return {
        "model": VLM_MODEL,
        "temperature": 0,
        "messages": [{"role": "user", "content": content}],
    }


def complete(payload: dict, post=None) -> dict:
    if post is None:
        post = _post
    body = post(payload)
    if isinstance(body, str):
        body = json.loads(body)
    if not isinstance(body, dict):
        raise ValueError("vlm response is not an object")
    return body


def parse_catalog(body: dict) -> dict:
    return _json_from_message(body)


def parse_verdict(body: dict) -> Verdict:
    payload = _json_from_message(body)
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("verdict entries missing")
    return Verdict(entries=[VerdictEntry.model_validate(item) for item in entries])


def _json_from_message(body: dict) -> dict:
    try:
        text = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("vlm message missing") from exc
    if not isinstance(text, str):
        raise ValueError("vlm message is not text")
    objects = _json_objects(text)
    for parsed in objects:
        if "entries" in parsed or "assets" in parsed or "poses" in parsed or "accept" in parsed:
            return parsed
    if objects and all("id" in item and "label" in item for item in objects):
        return {"entries": objects}
    if len(objects) == 1:
        return objects[0]
    reply = repo_root() / "outputs" / "vlm_reply.txt"
    reply.parent.mkdir(parents=True, exist_ok=True)
    reply.write_text(text[:8000], encoding="utf-8")
    raise ValueError("vlm message has no JSON object")


def _json_objects(text: str) -> list[dict]:
    """Parse every JSON object in a reply. Extra prose after one object is ignored."""
    decoder = json.JSONDecoder()
    found: list[dict] = []
    index = 0
    while index < len(text):
        start = text.find("{", index)
        if start < 0:
            break
        try:
            parsed, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            index = start + 1
            continue
        if isinstance(parsed, dict):
            found.append(parsed)
        index = end
    return found


def _post(payload: dict) -> dict:
    import requests

    response = requests.post(VLM_URL, json=payload, timeout=300)
    response.raise_for_status()
    return response.json()


def _port_open() -> bool:
    import requests

    try:
        response = requests.get("http://127.0.0.1:8080/v1/models", timeout=2)
    except requests.RequestException:
        return False
    return response.ok


def _log_gpu(when: str, free: int, need: int) -> None:
    print(f"FARMGEN stage=gpu when={when} free_mib={free} need_mib={need}", flush=True)
