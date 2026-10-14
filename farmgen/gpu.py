"""Read GPU memory from nvidia-smi and wait until a job can have the card.

Image generation, TRELLIS, and Qwen share one GPU. The waiter polls free
memory. When the process holding the card is the vLLM server this run
started, the caller stops that server instead of waiting on itself.
"""

from __future__ import annotations

import subprocess
import time

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
