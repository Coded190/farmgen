"""Read GPU memory from nvidia-smi and wait until a job can have the card.

Two usable cards keep Qwen on one and Flux or TRELLIS on the other.
One usable card runs those jobs one after another. No usable card waits
until a foreign process exits, then uses that card.

Image generation, TRELLIS, and Qwen share one GPU. The waiter polls free
memory. When the process holding the card is the vLLM server this run
started, the caller stops that server instead of waiting on itself.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass

HEADROOM_MIB = 4096
IMAGE_BUDGET_MIB = 28000
TRELLIS_BUDGET_MIB = 40000
QWEN_BUDGET_MIB = 70000
WAIT_TIMEOUT_S = 1800.0
WAIT_POLL_S = 10.0


def gpu_memory_mib() -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    line = result.stdout.strip().splitlines()[0]
    return int(line.split()[0])


def query_gpu() -> tuple[int, str]:
    """Free MiB on GPU 0 and ``vllm`` when a vLLM process is a compute app."""
    free = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=process_name", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    free_mib = int(free.stdout.strip().splitlines()[0].split()[0])
    holder = ""
    if "vllm" in apps.stdout.lower():
        holder = "vllm"
    return free_mib, holder


def wait_for_vram(
    need_mib: int,
    *,
    query=None,
    sleep_fn=None,
    clock=None,
    timeout_s: float = WAIT_TIMEOUT_S,
    poll_s: float = WAIT_POLL_S,
    stop_own=None,
    on_event=None,
) -> int:
    """Block until ``free >= need_mib + 4096``.

    ``query`` returns ``(free_mib, holder)``. ``holder == "vllm"`` with a
    ``stop_own`` callback stops that server once, then keeps polling. The
    clock and sleep function are injected so tests do not wait on a GPU.
    """
    query = query_gpu if query is None else query
    sleep_fn = time.sleep if sleep_fn is None else sleep_fn
    clock = time.monotonic if clock is None else clock
    required = int(need_mib) + HEADROOM_MIB
    started = clock()
    stopped = False
    while True:
        free, holder = query()
        free = int(free)
        if free >= required:
            if on_event is not None:
                on_event("ready", free, required)
            return free
        if holder == "vllm" and stop_own is not None and not stopped:
            if on_event is not None:
                on_event("stop_vlm", free, required)
            stop_own()
            stopped = True
            continue
        if clock() - started >= timeout_s:
            if on_event is not None:
                on_event("timeout", free, required)
            raise TimeoutError(f"stage=gpu when=timeout free_mib={free} need_mib={required}")
        if on_event is not None:
            on_event("wait", free, required)
        sleep_fn(poll_s)


@dataclass(frozen=True)
class GpuCard:
    index: int
    free_mib: int
    processes: tuple[tuple[int, str], ...] = ()


@dataclass(frozen=True)
class GpuPlan:
    qwen: int
    worker: int
    serial: bool
    blocking: tuple[int, str] | None = None


_OWN_PIDS: set[int] = set()
_WORKER_DEVICE: int | None = None


def note_own_pid(pid: int) -> None:
    """A compute process this run started. It does not make its card busy."""
    _OWN_PIDS.add(int(pid))


def remember_worker(index: int) -> None:
    """GPU index the next Flux or TRELLIS child should see."""
    global _WORKER_DEVICE
    _WORKER_DEVICE = index


def apply_worker_device(env: dict[str, str]) -> dict[str, str]:
    if _WORKER_DEVICE is not None and _WORKER_DEVICE >= 0:
        env["CUDA_VISIBLE_DEVICES"] = str(_WORKER_DEVICE)
    return env


def _default_own() -> set[int]:
    return {os.getpid(), *_OWN_PIDS}


def ppid_from_stat(text: str) -> int | None:
    """Parent pid from one ``/proc/<pid>/stat`` line.

    The comm field is wrapped in parentheses and can itself contain spaces.
    The field after the closing parenthesis is the state letter, and the
    parent pid is the field after that.
    """
    end = text.rfind(")")
    if end < 0:
        return None
    fields = text[end + 2 :].split()
    if len(fields) < 2:
        return None
    try:
        return int(fields[1])
    except ValueError:
        return None


def _parent_pid(pid: int) -> int | None:
    """Parent pid from ``/proc``. A missing process has no parent."""
    try:
        text = open(f"/proc/{int(pid)}/stat", encoding="utf-8").read()
    except OSError:
        return None
    return ppid_from_stat(text)


def process_is_ours(pid: int, own_pids: set[int], parent_of=None) -> bool:
    """True when ``pid`` or one of its parents is a process this run started.

    ``VLLM::EngineCore`` is a child of the serve process. nvidia-smi reports
    the child, and ``note_own_pid`` stores the parent. Walking parents is how
    that child stays ours. A parent of 1 is init, which is not this run.
    """
    lookup = _parent_pid if parent_of is None else parent_of
    current = int(pid)
    seen: set[int] = set()
    for _step in range(8):
        if current in own_pids:
            return True
        if current in seen:
            return False
        seen.add(current)
        parent = lookup(current)
        if parent is None or int(parent) <= 1:
            return False
        current = int(parent)
    return current in own_pids


def query_cards(rows: list[GpuCard] | None = None) -> list[GpuCard]:
    """Every GPU index, its free MiB, and the compute processes on it."""
    if rows is not None:
        return list(rows)
    listed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    named = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    uuid_of: dict[int, str] = {}
    for line in named.stdout.splitlines():
        if not line.strip():
            continue
        index_text, uuid = (part.strip() for part in line.split(",", 1))
        uuid_of[int(index_text)] = uuid
    by_uuid: dict[str, list[tuple[int, str]]] = {}
    for line in processes.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        by_uuid.setdefault(parts[0], []).append((int(parts[1]), parts[2]))
    cards: list[GpuCard] = []
    for line in listed.stdout.splitlines():
        if not line.strip():
            continue
        index_text, free_text = (part.strip() for part in line.split(",", 1))
        index = int(index_text)
        cards.append(GpuCard(index, int(free_text), tuple(by_uuid.get(uuid_of.get(index, ""), []))))
    return cards


def plan_gpus(
    need_mib: int,
    rows: list[GpuCard] | None = None,
    own_pids: set[int] | None = None,
    parent_of=None,
) -> GpuPlan:
    """Prefer two cards. Fall back to one. Name the blocker when none is free."""
    mine = _default_own() if own_pids is None else set(own_pids)
    cards = query_cards(rows)
    usable = [card for card in cards if _usable(card, need_mib, mine, parent_of)]
    if len(usable) >= 2:
        return GpuPlan(qwen=usable[0].index, worker=usable[1].index, serial=False)
    if len(usable) == 1:
        return GpuPlan(qwen=usable[0].index, worker=usable[0].index, serial=True)
    return GpuPlan(qwen=-1, worker=-1, serial=True, blocking=_blocker(cards, mine, parent_of))


def free_device_indexes(
    need_mib: int,
    rows: list[GpuCard] | None = None,
    own_pids: set[int] | None = None,
    parent_of=None,
) -> list[int]:
    """Every card with enough free memory and no foreign compute process."""
    mine = _default_own() if own_pids is None else set(own_pids)
    cards = query_cards(rows)
    return [card.index for card in cards if _usable(card, need_mib, mine, parent_of)]


def wait_for_plan(
    need_mib: int,
    *,
    query=None,
    sleep_fn=None,
    clock=None,
    timeout_s: float = WAIT_TIMEOUT_S,
    poll_s: float = WAIT_POLL_S,
    own_pids: set[int] | None = None,
    stop_own=None,
    on_event=None,
    parent_of=None,
) -> GpuPlan:
    """Poll until one card can take the job. A timeout names the blocking process."""
    query = (lambda: query_cards()) if query is None else query
    sleep_fn = time.sleep if sleep_fn is None else sleep_fn
    clock = time.monotonic if clock is None else clock
    started = clock()
    stopped = False
    while True:
        cards = list(query())
        plan = plan_gpus(need_mib, cards, own_pids, parent_of)
        if plan.qwen >= 0:
            if on_event is not None:
                on_event("ready", cards[0].free_mib if cards else 0, int(need_mib) + HEADROOM_MIB)
            return plan
        if not stopped and stop_own is not None and _own_vllm(cards, own_pids, parent_of):
            if on_event is not None:
                on_event("stop_vlm", 0, int(need_mib) + HEADROOM_MIB)
            stop_own()
            stopped = True
            continue
        if clock() - started >= timeout_s:
            if plan.blocking is None:
                raise TimeoutError(f"no GPU free for {need_mib} MiB")
            index, name = plan.blocking
            if on_event is not None:
                on_event("timeout", 0, int(need_mib) + HEADROOM_MIB)
            raise TimeoutError(f"GPU {index} blocked by {name}")
        if plan.blocking is not None:
            index, name = plan.blocking
            if on_event is not None:
                on_event("wait", 0, int(need_mib) + HEADROOM_MIB)
            from farmgen.log import log_line

            log_line("gpu", 0, "vram", note="waiting", gpu=index, process=name, need_mib=need_mib)
        sleep_fn(poll_s)


def wait_for_devices(
    need_mib: int,
    *,
    query=None,
    sleep_fn=None,
    clock=None,
    timeout_s: float = WAIT_TIMEOUT_S,
    poll_s: float = WAIT_POLL_S,
    own_pids: set[int] | None = None,
    stop_own=None,
    on_event=None,
    parent_of=None,
) -> list[int]:
    """Poll until at least one card can hold the job. Return every such card.

    Our vLLM, including ``VLLM::EngineCore``, is stopped once so its card can
    be one of the returned devices. A foreign process is left running.
    """
    query = (lambda: query_cards()) if query is None else query
    sleep_fn = time.sleep if sleep_fn is None else sleep_fn
    clock = time.monotonic if clock is None else clock
    started = clock()
    stopped = False
    while True:
        cards = list(query())
        devices = free_device_indexes(need_mib, cards, own_pids, parent_of)
        if devices:
            if on_event is not None:
                on_event("ready", cards[0].free_mib if cards else 0, int(need_mib) + HEADROOM_MIB)
            return devices
        if not stopped and stop_own is not None and _own_vllm(cards, own_pids, parent_of):
            if on_event is not None:
                on_event("stop_vlm", 0, int(need_mib) + HEADROOM_MIB)
            stop_own()
            stopped = True
            continue
        plan = plan_gpus(need_mib, cards, own_pids, parent_of)
        if clock() - started >= timeout_s:
            if plan.blocking is None:
                raise TimeoutError(f"no GPU free for {need_mib} MiB")
            index, name = plan.blocking
            if on_event is not None:
                on_event("timeout", 0, int(need_mib) + HEADROOM_MIB)
            raise TimeoutError(f"GPU {index} blocked by {name}")
        if plan.blocking is not None:
            index, name = plan.blocking
            if on_event is not None:
                on_event("wait", 0, int(need_mib) + HEADROOM_MIB)
            from farmgen.log import log_line

            log_line("gpu", 0, "vram", note="waiting", gpu=index, process=name, need_mib=need_mib)
        sleep_fn(poll_s)


def _usable(card: GpuCard, need_mib: int, own_pids: set[int], parent_of=None) -> bool:
    if card.free_mib < need_mib + HEADROOM_MIB:
        return False
    return all(process_is_ours(pid, own_pids, parent_of) for pid, _name in card.processes)


def _blocker(cards: list[GpuCard], own_pids: set[int], parent_of=None) -> tuple[int, str] | None:
    for card in cards:
        for pid, name in card.processes:
            if not process_is_ours(pid, own_pids, parent_of):
                return card.index, name
    if cards:
        return cards[0].index, "unknown"
    return None


def _own_vllm(cards: list[GpuCard], own_pids: set[int] | None, parent_of=None) -> bool:
    mine = _default_own() if own_pids is None else set(own_pids)
    for card in cards:
        for pid, name in card.processes:
            if "vllm" not in name.lower():
                continue
            if process_is_ours(pid, mine, parent_of) or _farmgen_vllm_pid(pid):
                return True
    return False


def _farmgen_vllm_pid(pid: int) -> bool:
    """True when this compute pid is the farmgen vLLM serve process or its engine.

    ``VLLM::EngineCore`` does not name the repo on its command line. It is ours
    when its environment contains this repo, or when a parent command line is
    this repo's ``vllm serve``. Another user's engine does not match.
    """
    from farmgen.paths import repo_root
    from farmgen.workers.vlm_critic import _proc_cmdline, cmdline_is_our_vllm

    root = str(repo_root())
    text = _proc_cmdline(pid)
    if cmdline_is_our_vllm(text, root):
        return True
    if "vllm" not in text.lower():
        return False
    try:
        env = open(f"/proc/{int(pid)}/environ", "rb").read()
    except OSError:
        env = b""
    if root.encode() in env:
        return True
    current = int(pid)
    for _step in range(8):
        parent = _parent_pid(current)
        if parent is None or parent <= 1:
            return False
        if cmdline_is_our_vllm(_proc_cmdline(parent), root):
            return True
        current = parent
    return False
