"""Local Qwen client for the asset catalog and the four-image verdict.

The server is OpenAI-compatible on 127.0.0.1:8080. No key file is read.
Tests pass a fake HTTP body or a fake post function.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from farmgen.critic import Verdict, VerdictEntry
from farmgen.gpu import QWEN_BUDGET_MIB
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


def cmdline_is_our_vllm(cmdline: str, root: str) -> bool:
    """True for the farmgen vLLM serve process. Another user's server does not match."""
    return "vllm" in cmdline and "serve" in cmdline and str(root) in cmdline


def _parent_pid(pid: int) -> int | None:
    from farmgen.gpu import ppid_from_stat

    try:
        text = open(f"/proc/{int(pid)}/stat", encoding="utf-8").read()
    except OSError:
        return None
    return ppid_from_stat(text)


def _proc_cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\x00", b" ").decode("utf-8", errors="replace")


def our_vllm_pids() -> list[int]:
    """Serve pids whose command line is this repo's vLLM. EngineCore is their child."""
    root = str(repo_root())
    found: list[int] = []
    proc = Path("/proc")
    if not proc.is_dir():
        return found
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        if cmdline_is_our_vllm(_proc_cmdline(int(entry.name)), root):
            found.append(int(entry.name))
    return found


def descendants_of(roots: list[int], *, children=None) -> list[int]:
    """Every process under ``roots``, not including the roots themselves."""
    if children is None:
        by_parent: dict[int, list[int]] = {}
        proc = Path("/proc")
        if proc.is_dir():
            for entry in proc.iterdir():
                if not entry.name.isdigit():
                    continue
                pid = int(entry.name)
                parent = _parent_pid(pid)
                if parent is not None:
                    by_parent.setdefault(parent, []).append(pid)

        def children(pid: int) -> list[int]:
            return by_parent.get(pid, [])
    found: list[int] = []
    seen = set(roots)
    queue = list(roots)
    while queue:
        current = queue.pop(0)
        for child in children(current):
            if child in seen:
                continue
            seen.add(child)
            found.append(child)
            queue.append(child)
    return found


def orphan_engine_pids() -> list[int]:
    """``VLLM::EngineCore`` reparented to init whose environ still names this repo.

    A live serve pid owns its engine through the process tree. This finds the
    engine only after that parent has already exited. Another user's engine
    does not have this repo path in its environment.
    """
    root = str(repo_root()).encode()
    found: list[int] = []
    proc = Path("/proc")
    if not proc.is_dir():
        return found
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if "VLLM::EngineCore" not in _proc_cmdline(pid):
            continue
        parent = _parent_pid(pid)
        if parent is not None and parent > 1:
            continue
        try:
            env = (entry / "environ").read_bytes()
        except OSError:
            continue
        if root in env:
            found.append(pid)
    return found


def stop_targets() -> list[int]:
    """Serve pids, their descendants, and an EngineCore left behind by a dead parent."""
    pids: list[int] = []
    if _SERVER is not None and _SERVER.poll() is None:
        pids.append(_SERVER.pid)
    for pid in our_vllm_pids():
        if pid not in pids:
            pids.append(pid)
    targets = list(pids)
    for pid in descendants_of(pids):
        if pid not in targets:
            targets.append(pid)
    for pid in orphan_engine_pids():
        if pid not in targets:
            targets.append(pid)
    return targets


def _signal_own_group(pid: int, sig: int) -> None:
    """Signal a session this repo started. A shared group is left alone."""
    try:
        pgid = os.getpgid(pid)
    except OSError:
        return
    if pgid != pid:
        return
    try:
        os.killpg(pgid, sig)
    except OSError:
        return


def stop_server() -> None:
    """Stop the vLLM this run started, including ``VLLM::EngineCore``.

    The serve command is matched on this repo's path. Descendants are taken
    before the signal, because the engine's own command line is ``VLLM::EngineCore``
    and does not name the repo. A supervisor is its own session, so that
    session is signaled too. A process outside that tree is left running.
    """
    global _SERVER
    targets = stop_targets()
    for pid in targets:
        _signal_own_group(pid, signal.SIGTERM)
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            continue
    if _SERVER is not None:
        try:
            _SERVER.wait(timeout=30)
        except subprocess.TimeoutExpired:
            _SERVER.kill()
        _SERVER = None
    deadline = time.monotonic() + 30
    for pid in targets:
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except OSError:
                break
            time.sleep(0.2)
        else:
            _signal_own_group(pid, signal.SIGKILL)
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
    if targets:
        from farmgen.log import log_line

        log_line("farmgen", 0, "vlm", note="stopped", pids=targets)


_SUPERVISOR = """
import ctypes
import os
import signal
import sys

def _stop(_signum, _frame):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    try:
        os.killpg(os.getpgrp(), signal.SIGTERM)
    except OSError:
        pass
    os._exit(0)

os.setsid()
ctypes.CDLL(None).prctl(1, int(signal.SIGTERM))
signal.signal(signal.SIGTERM, _stop)
signal.signal(signal.SIGINT, _stop)
if os.getppid() == 1:
    _stop(signal.SIGTERM, None)
pid = os.fork()
if pid == 0:
    ctypes.CDLL(None).prctl(1, int(signal.SIGTERM))
    if os.getppid() == 1:
        os._exit(0)
    os.execvp(sys.argv[1], sys.argv[1:])
try:
    os.waitpid(pid, 0)
except OSError:
    pass
"""


def supervised_command(cmd: list[str]) -> list[str]:
    """Run vLLM in its own session and signal that session when farmgen exits.

    ``PR_SET_PDEATHSIG`` delivers SIGTERM to this supervisor when its parent
    dies, including a kill that never runs Python cleanup. The handler signals
    the whole session, so ``VLLM::EngineCore`` is not left holding the GPU.
    """
    return [sys.executable, "-c", _SUPERVISOR, *cmd]


def ensure_server() -> None:
    """Start vLLM when the weights are on disk and the port is closed."""
    global _SERVER
    if not (weights_dir() / "config.json").is_file():
        raise RuntimeError(f"qwen weights missing at {weights_dir()}")
    if _port_open():
        from farmgen.gpu import note_own_pid

        for pid in our_vllm_pids():
            note_own_pid(pid)
        return
    from farmgen.gpu import note_own_pid, wait_for_plan

    plan = wait_for_plan(QWEN_BUDGET_MIB, stop_own=stop_server, on_event=_log_gpu)
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
    env = _vllm_env()
    if plan.qwen >= 0:
        env["CUDA_VISIBLE_DEVICES"] = str(plan.qwen)
    _SERVER = subprocess.Popen(
        supervised_command(cmd),
        cwd=repo_root(),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    note_own_pid(_SERVER.pid)
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


@dataclass(frozen=True)
class CatalogAnswer:
    """Parsed catalog JSON plus the model text that produced it."""

    payload: dict
    reply: str


def request_catalog(prompt: str, images=None, post=None) -> CatalogAnswer:
    ensure_server()
    body = complete(catalog_message(prompt, images), post=post)
    return CatalogAnswer(parse_catalog(body), _message_text(body))


def request_catalog_gate(brief: str, assets: list, images=None, post=None):
    """Accept or reject a finished asset list. This pass does not write a new list."""
    from farmgen.mesh_gate import MeshDecision

    ensure_server()
    body = complete(catalog_gate_payload(brief, assets, images), post=post)
    decision = parse_mesh_gate(body, "catalog")
    return MeshDecision(decision.accept, decision.reason, _message_text(body)[:4000])


def catalog_gate_payload(brief: str, assets: list, images=None, missing: list[str] | None = None) -> dict:
    from farmgen.scene import missing_required

    rows = [_catalog_row(asset) for asset in assets]
    pictures = [Path(image) for image in (images or [])]
    absent = list(missing) if missing is not None else missing_required(brief, assets)
    text = (
        f"The request was: {brief}. "
        "Accept this asset list only when it names the objects in that request"
        + (" and the objects in the images." if pictures else ".")
        + " Reject a list that misses a named object, describes a different farm, or puts two objects in one prompt. "
        "Do not rewrite the list. "
    )
    if absent:
        text += f" These required objects are absent after normalization: {', '.join(absent)}. Reject the list. "
    text += (
        'Reply with JSON {"id":"catalog","accept":true,"reason":""} and fill reason. '
        f"Assets: {json.dumps(rows)}"
    )
    content: str | list[dict] = [{"type": "text", "text": text}]
    if pictures:
        content.extend(_image_content(image) for image in pictures)
    else:
        content = text
    return {
        "model": VLM_MODEL,
        "temperature": 0,
        "messages": [{"role": "user", "content": content}],
    }


def _catalog_row(asset: object) -> dict:
    if isinstance(asset, dict):
        return {key: asset.get(key) for key in ("id", "prompt", "size_m", "count", "layout")}
    return {
        "id": asset.id,
        "prompt": asset.prompt,
        "size_m": list(asset.size_m),
        "count": asset.count,
        "layout": asset.layout,
    }


def _message_text(body: dict) -> str:
    try:
        reply = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""
    if not isinstance(reply, str):
        return str(reply)
    return reply


def request_mesh_gate(asset_id: str, prompt: str, images: list[Path], post=None, note: str = ""):
    """Qwen's accept flag, the reason sentence, and the raw reply text."""
    from farmgen.mesh_gate import MeshDecision

    ensure_server()
    body = complete(mesh_gate_payload(asset_id, prompt, images, note=note), post=post)
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


def request_verdict(kind: str, ids: list[str], images: list[Path], post=None, note: str = "") -> Verdict:
    ensure_server()
    body = complete(verdict_payload(kind, ids, images, note=note), post=post)
    verdict = parse_verdict(body)
    return verdict.model_copy(update={"reply": _message_text(body)})


def poses_payload(
    ids: list[str],
    image: Path,
    required: list[tuple[str, int]] | None = None,
    problems: list[tuple[str, int, str]] | None = None,
    placed: str = "",
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
        "The south 18 meters of the yard, the lower part of the brown square, is the crop band. "
        "A barn, silo, tractor, trough, crate, or hay bale goes in the north 22 meters. "
        "A surround asset goes in the green ring, outside the brown square. "
        "Look at the environment in the image, then write a point that fits beside the objects already placed. "
        "Reply with JSON "
        '{"poses":[{"id":"barn_01","index":0,"u":0.50,"v":0.50,"yaw_deg":0},'
        '{"id":"tree_01","index":0,"u":0.20,"v":0.20,"yaw_deg":0}]}. '
        "u and v are 0 to 1 on this image. u 0 is the left edge. v 0 is the top edge. "
        "0.50,0.50 is the yard center and is only for one yard asset. "
        "yaw_deg is degrees about Z. One pose per copy, so index runs from 0 to count-1. "
        f"Ids: {', '.join(ids)}.{copies}"
    )
    if placed:
        prompt += f" Already placed: {placed}."
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
    placed: str = "",
    post=None,
) -> list[dict]:
    ensure_server()
    body = complete(poses_payload(ids, image, required, problems, placed), post=post)
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


def mesh_gate_payload(asset_id: str, prompt: str, images: list[Path], note: str = "") -> dict:
    measure = ""
    if note:
        measure = (
            " A following note reports flatness, the sheet-area fraction, and the non-manifold edge count. "
            "An edge-on sheet in the first picture is a reject. "
            f"{note} "
        )
    text = (
        "The first picture is the thinnest view of one isolated mesh, stood upright. "
        "The pictures before the last are the other views. "
        "The last picture is the photo that mesh was built from. These pictures are not the farm scene. "
        f"Accept the mesh only when these views show one {prompt}. "
        "Reject a forest, a collage, a logo, the wrong object, or an edge-on sheet. "
        "A flat card seen from the thin side is a reject even when another view looks like the object. "
        f"{measure}"
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


def _stage_prompt(kind: str, ids: list[str], note: str) -> str | None:
    joined = ", ".join(ids)
    extra = f" {note}" if note else ""
    score = "A label other than ok schedules its action even when the score is 4 or below. "
    if kind == "ground":
        return (
            "Judge the ground only. The pictures are the kept terrain. "
            "Accept tilled soil when the yard is visually distinct from the ring around it. "
            'Reply with JSON {"entries":[{"id":"ground","label":"ok","action":null,"score":0}]}. '
            "label is ok or not_agricultural. action is null or redo_ground. "
            f"{score}{extra} Ids: {joined}."
        )
    if kind == "image":
        return (
            "Judge each picture against the asset it belongs to. "
            "A web photo and a Flux picture use the same rule: the picture must show that one object. "
            'Reply with JSON {"entries":[{"id":"...","label":"ok","action":null,"score":0}]}. '
            "label is ok or missing. A wrong object uses action redo_image. "
            f"{score}{extra} Ids: {joined}."
        )
    if kind == "mesh":
        return (
            "Judge each TRELLIS mesh thumbnail against its asset. "
            "Reject a flat card, a collage, or the wrong object. "
            "A flatness number and a non-manifold edge count in the note are measurements. "
            "The pictures decide. "
            'Reply with JSON {"entries":[{"id":"...","label":"ok","action":null,"score":0}]}. '
            "label is ok or missing. A bad mesh uses action redo_asset. "
            f"{score}{extra} Ids: {joined}."
        )
    if kind == "scale":
        return (
            "The pictures are yaw 0, 90, 180, and 270 of each mesh stood on the ground. "
            "Read the vertical height in the front view and compare it with the meters in the id. "
            "Not upright is wrong_orientation and action reorient. "
            "The wrong size is wrong_scale and action redo_asset. "
            'Reply with JSON {"entries":[{"id":"...","label":"ok","action":null,"score":0}]}. '
            f"{score}{extra} Ids: {joined}."
        )
    if kind == "pose":
        return (
            "Judge the stored poses. Do not invent a new point in this call. "
            "The top image is the environment the points were written on. "
            'Reply with JSON {"entries":[{"id":"...","label":"ok","action":null,"score":0,"index":0}]}. '
            "label is ok or wrong_place. A reject asks for a new point and uses action move. "
            f"{score}{extra} Ids: {joined}."
        )
    return None


def verdict_payload(kind: str, ids: list[str], images: list[Path], note: str = "") -> dict:
    facing = ""
    specific = _stage_prompt(kind, ids, note)
    if specific is not None:
        prompt = specific
        content: list[dict] = [{"type": "text", "text": prompt}]
        for image in images:
            content.append(_image_content(Path(image)))
        return {
            "model": VLM_MODEL,
            "temperature": 0,
            "messages": [{"role": "user", "content": content}],
        }
    if kind == "place":
        prompt = (
            "Judge placement only. Reply with JSON "
            '{"entries":[{"id":"...","label":"ok","action":null,"score":0,"index":0,"u":0.5,"v":0.5,"yaw_deg":0}]}. '
            "The brown square is the yard. The green area is the ring. "
            "score is 0 to 10. A label other than ok schedules its action even when the score is 4 or below. "
            "label is ok or wrong_place. action is null or move. "
            "A yard asset that shares another asset's ground point is wrong_place. "
            "A surround asset inside the yard, or absent from the green ring, is wrong_place. "
            "wrong_place requires action move. "
            "A move includes index, u, v, and yaw_deg. u and v are 0 to 1 on the top view. "
            "u 0 is the left edge and v 0 is the top edge. "
            "Mesh shape was judged earlier. Do not use redo_asset. "
            f"{note} "
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
            " The first picture is the top view when that file is present. "
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
        "score is 0 to 10. A label other than ok schedules its action even when the score is 4 or below. "
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
        f"{note} "
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
