"""Local Qwen client for the asset catalog and the four-image verdict.

The server is OpenAI-compatible on 127.0.0.1:8080. No key file is read.
Tests pass a fake HTTP body or a fake post function.
"""

from __future__ import annotations

import base64
import json
import subprocess
import time
from pathlib import Path

from farmgen.critic import Verdict, VerdictEntry
from farmgen.gpu import QWEN_BUDGET_MIB, wait_for_vram
from farmgen.paths import repo_root

VLM_URL = "http://127.0.0.1:8080/v1/chat/completions"
VLM_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"
_SERVER: subprocess.Popen | None = None


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
        "vllm",
        "serve",
        str(weights_dir()),
        "--host",
        "127.0.0.1",
        "--port",
        "8080",
        "--served-model-name",
        VLM_MODEL,
    ]
    _SERVER = subprocess.Popen(cmd, cwd=repo_root())
    deadline = time.monotonic() + 600
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


def request_catalog(brief: str, post=None) -> dict:
    ensure_server()
    body = complete(catalog_payload(brief), post=post)
    return parse_catalog(body)


def request_verdict(kind: str, ids: list[str], images: list[Path], post=None) -> Verdict:
    ensure_server()
    body = complete(verdict_payload(kind, ids, images), post=post)
    return parse_verdict(body)


def catalog_payload(brief: str) -> dict:
    prompt = (
        "Write a farm asset list as JSON. "
        'Schema: {"assets":[{"id":"barn_01","prompt":"...","size_m":[12,8,7],'
        '"count":1,"layout":"unique"}]}. '
        "layout is unique, line, grid, or rows. unique requires count 1. "
        "size_m is width, depth, height in meters. "
        f"Request: {brief}"
    )
    return {
        "model": VLM_MODEL,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
    }


def verdict_payload(kind: str, ids: list[str], images: list[Path]) -> dict:
    prompt = (
        f"Judge this farm {kind}. Reply with JSON "
        '{"entries":[{"id":"...","label":"ok","action":null,"score":0}]}. '
        "score is 0 to 10. A score of 4 or below means acceptable. "
        "label is ok, missing, wrong_scale, floating, overlapping, "
        "not_agricultural, or wrong_orientation. "
        "action is null, redo_ground, redo_asset, reorient, redo_image, or redo_place. "
        "wrong_orientation uses reorient. A bad generated picture uses redo_image. "
        "A bad mesh uses redo_asset. A bad arrangement uses redo_place. "
        "The ground id is ground and its action is redo_ground. "
        f"Ids: {', '.join(ids)}."
    )
    content: list[dict] = [{"type": "text", "text": prompt}]
    for image in images:
        encoded = base64.standard_b64encode(Path(image).read_bytes()).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}})
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
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("vlm message has no JSON object")
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("vlm JSON is not an object")
    return parsed


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
