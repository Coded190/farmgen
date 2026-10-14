"""The VRAM waiter polls on an injected clock and stops our own vLLM holder."""

import pytest

from farmgen.gpu import HEADROOM_MIB, wait_for_vram


def test_waiter_polls_then_returns_when_memory_is_free():
    state = {"free": 1000, "sleeps": 0, "now": 0.0, "events": []}

    def query():
        return state["free"], ""

    def sleep_fn(seconds):
        state["sleeps"] += 1
        state["now"] += seconds
        state["free"] = 50000

    def clock():
        return state["now"]

    def on_event(when, free, need):
        state["events"].append(when)

    free = wait_for_vram(
        10000,
        query=query,
        sleep_fn=sleep_fn,
        clock=clock,
        timeout_s=100,
        poll_s=10,
        on_event=on_event,
    )
    assert free == 50000
    assert state["sleeps"] == 1
    assert state["events"] == ["wait", "ready"]
    assert 10000 + HEADROOM_MIB == 14096


def test_waiter_stops_our_vllm_instead_of_sleeping_on_it():
    stopped = {"n": 0}
    calls = {"n": 0}

    def query():
        calls["n"] += 1
        if stopped["n"]:
            return 80000, ""
        return 1000, "vllm"

    def stop_own():
        stopped["n"] += 1

    free = wait_for_vram(1000, query=query, sleep_fn=lambda _s: None, clock=lambda: 0.0, timeout_s=30, stop_own=stop_own)
    assert stopped["n"] == 1
    assert free == 80000
    assert calls["n"] == 2


def test_waiter_times_out():
    def query():
        return 0, "other"

    clock = {"now": 0.0}

    def sleep_fn(seconds):
        clock["now"] += seconds

    with pytest.raises(TimeoutError, match="when=timeout"):
        wait_for_vram(
            100,
            query=query,
            sleep_fn=sleep_fn,
            clock=lambda: clock["now"],
            timeout_s=25,
            poll_s=10,
        )
