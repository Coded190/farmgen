"""ComfyUI text-to-image client. The server is started and stopped by the caller."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import requests

from farmgen.paths import repo_root


def load_workflow(prompt: str, seed: int) -> dict:
    path = repo_root() / "configs" / "flux_schnell_api.json"
    workflow = json.loads(path.read_text(encoding="utf-8"))
    text = (
        f"a single {prompt}, centered, entire object visible, "
        "plain light gray background, studio product photo, no text"
    )
    workflow["6"]["inputs"]["text"] = text
    workflow["16"]["inputs"]["seed"] = int(seed)
    workflow["9"]["inputs"]["filename_prefix"] = "farmgen"
    return workflow


def queue_image(base: str, prompt: str, seed: int, dest: Path, timeout_s: float = 600.0) -> Path:
    workflow = load_workflow(prompt, seed)
    client_id = str(uuid.uuid4())
    response = requests.post(
        f"{base}/prompt",
        json={"prompt": workflow, "client_id": client_id},
        timeout=60,
    )
    response.raise_for_status()
    body = response.json()
    if body.get("node_errors"):
        raise RuntimeError(f"ComfyUI node_errors: {body['node_errors']}")
    prompt_id = body["prompt_id"]
    deadline = time.monotonic() + timeout_s
    history = None
    while time.monotonic() < deadline:
        poll = requests.get(f"{base}/history/{prompt_id}", timeout=30)
        poll.raise_for_status()
        payload = poll.json()
        if prompt_id in payload:
            history = payload[prompt_id]
            break
        time.sleep(1.0)
    if history is None:
        raise TimeoutError(f"ComfyUI prompt {prompt_id} did not finish in {timeout_s:.0f}s")
    status = history.get("status", {})
    if status.get("status_str") == "error":
        raise RuntimeError(f"ComfyUI error: {status}")
    image_info = _first_image(history)
    view = requests.get(f"{base}/view", params=image_info, timeout=60)
    view.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(view.content)
    if dest.stat().st_size == 0:
        raise RuntimeError(f"ComfyUI wrote an empty image to {dest}")
    return dest


def free_memory(base: str) -> None:
    requests.post(f"{base}/free", json={"unload_models": True, "free_memory": True}, timeout=60).raise_for_status()


def _first_image(history: dict) -> dict:
    for node in history.get("outputs", {}).values():
        images = node.get("images") or []
        if images:
            info = images[0]
            return {"filename": info["filename"], "subfolder": info.get("subfolder", ""), "type": info.get("type", "output")}
    raise RuntimeError(f"ComfyUI history has no images: {list(history.get('outputs', {}))}")
