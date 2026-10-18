"""The VRAM waiter polls on an injected clock and stops our own vLLM holder."""

import pytest

from farmgen.gpu import (
    HEADROOM_MIB,
    GpuCard,
    apply_worker_device,
    free_device_indexes,
    plan_gpus,
    process_is_ours,
    remember_worker,
    wait_for_devices,
    wait_for_plan,
    wait_for_vram,
)


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


def test_parent_pid_is_the_field_after_the_state_letter():
    from farmgen.gpu import ppid_from_stat, process_is_ours

    engine = "4093180 (VLLM::EngineCore) S 4090306 4012987 4012987 0 -1"
    serve = "4090306 (python3) S 1 4012987 4012987 0 -1"
    assert ppid_from_stat(engine) == 4090306
    assert ppid_from_stat(serve) == 1
    assert ppid_from_stat("no parentheses here") is None
    parents = {4093180: 4090306, 4090306: 250546}
    assert process_is_ours(4093180, {250546}, parent_of=parents.get) is True
    assert process_is_ours(4093180, {10}, parent_of=lambda _pid: 1) is False


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


def test_the_worker_card_is_the_device_a_child_process_sees():
    remember_worker(1)
    assert apply_worker_device({})["CUDA_VISIBLE_DEVICES"] == "1"


def test_two_free_cards_split_qwen_and_the_worker():
    rows = [GpuCard(0, 90000), GpuCard(1, 90000)]
    plan = plan_gpus(28000, rows, own_pids=set())
    assert plan.qwen == 0
    assert plan.worker == 1
    assert plan.serial is False


def test_one_busy_card_runs_the_job_on_the_free_card():
    rows = [GpuCard(0, 90000, ((99, "train.py"),)), GpuCard(1, 90000)]
    plan = plan_gpus(28000, rows, own_pids=set())
    assert plan.serial is True
    assert plan.qwen == 1
    assert plan.worker == 1


def test_both_busy_cards_wait_and_then_use_the_card_that_frees():
    calls = {"n": 0}

    def query():
        calls["n"] += 1
        if calls["n"] == 1:
            return [GpuCard(0, 100, ((9, "train.py"),)), GpuCard(1, 100, ((8, "other"),))]
        return [GpuCard(0, 90000), GpuCard(1, 100, ((8, "other"),))]

    slept = {"n": 0}
    plan = wait_for_plan(
        28000,
        query=query,
        sleep_fn=lambda _seconds: slept.__setitem__("n", slept["n"] + 1),
        clock=lambda: 0.0,
        timeout_s=100,
        poll_s=1,
        own_pids=set(),
    )
    assert slept["n"] == 1
    assert calls["n"] == 2
    assert plan.serial is True
    assert plan.worker == 0
    assert plan.qwen == 0


def test_a_busy_pair_times_out_with_the_blocking_process_name():
    clock = {"now": 0.0}

    def sleep_fn(seconds):
        clock["now"] += seconds

    with pytest.raises(TimeoutError, match="GPU 0 blocked by train.py"):
        wait_for_plan(
            28000,
            query=lambda: [GpuCard(0, 100, ((9, "train.py"),)), GpuCard(1, 100, ((8, "other"),))],
            sleep_fn=sleep_fn,
            clock=lambda: clock["now"],
            timeout_s=15,
            poll_s=10,
            own_pids=set(),
        )
    assert clock["now"] >= 10


def test_an_enginecore_child_of_our_server_is_ours_and_is_stopped():
    assert process_is_ours(50, {10}, parent_of=lambda pid: 10 if pid == 50 else None) is True
    assert process_is_ours(99, {10}, parent_of=lambda pid: 1) is False
    stopped = {"n": 0}
    calls = {"n": 0}

    def query():
        calls["n"] += 1
        if stopped["n"]:
            return [GpuCard(0, 90000), GpuCard(1, 100, ((8, "other"),))]
        return [GpuCard(0, 1000, ((50, "VLLM::EngineCore"),)), GpuCard(1, 100, ((8, "other"),))]

    devices = wait_for_devices(
        40000,
        query=query,
        sleep_fn=lambda _seconds: None,
        clock=lambda: 0.0,
        timeout_s=30,
        own_pids={10},
        stop_own=lambda: stopped.__setitem__("n", stopped["n"] + 1),
        parent_of=lambda pid: 10 if pid == 50 else 1,
    )
    assert stopped["n"] == 1
    assert devices == [0]
    assert calls["n"] == 2


def test_two_free_cards_are_both_trellis_devices():
    rows = [GpuCard(0, 90000), GpuCard(1, 90000)]
    assert free_device_indexes(40000, rows, own_pids=set()) == [0, 1]


def test_an_engine_whose_environ_names_this_repo_is_stopped(monkeypatch):
    import io

    from farmgen import gpu
    from farmgen.paths import repo_root
    from farmgen.workers import vlm_critic

    monkeypatch.setattr(vlm_critic, "_proc_cmdline", lambda _pid: "VLLM::EngineCore")
    monkeypatch.setattr(gpu, "_parent_pid", lambda _pid: None)

    def fake_open(path, mode="r", *args, **kwargs):
        if path == "/proc/77/environ" and "b" in mode:
            return io.BytesIO(str(repo_root()).encode() + b"\x00")
        raise FileNotFoundError(path)

    monkeypatch.setattr("builtins.open", fake_open)
    stopped = {"n": 0}
    devices = gpu.wait_for_devices(
        40000,
        query=lambda: [gpu.GpuCard(0, 1000, ((77, "VLLM::EngineCore"),))]
        if not stopped["n"]
        else [gpu.GpuCard(0, 90000)],
        sleep_fn=lambda _seconds: None,
        clock=lambda: 0.0,
        timeout_s=30,
        own_pids=set(),
        stop_own=lambda: stopped.__setitem__("n", stopped["n"] + 1),
        parent_of=lambda _pid: 1,
    )
    assert stopped["n"] == 1
    assert devices == [0]


def test_a_missing_engine_pid_is_not_treated_as_ours():
    from farmgen.gpu import _farmgen_vllm_pid

    assert _farmgen_vllm_pid(2_000_000_001) is False


def test_a_foreign_enginecore_is_not_stopped():
    with pytest.raises(TimeoutError, match="blocked by VLLM::EngineCore"):
        wait_for_devices(
            40000,
            query=lambda: [GpuCard(0, 1000, ((99, "VLLM::EngineCore"),))],
            sleep_fn=lambda _seconds: None,
            clock=lambda: 0.0,
            timeout_s=0,
            own_pids={10},
            stop_own=lambda: (_ for _ in ()).throw(AssertionError("foreign vllm was stopped")),
            parent_of=lambda _pid: 1,
        )


def test_stop_targets_include_the_engine_child(monkeypatch):
    from farmgen.workers import vlm_critic

    monkeypatch.setattr(vlm_critic, "_SERVER", None)
    monkeypatch.setattr(vlm_critic, "our_vllm_pids", lambda: [10])
    monkeypatch.setattr(vlm_critic, "descendants_of", lambda roots: [11, 12] if roots == [10] else [])
    monkeypatch.setattr(vlm_critic, "orphan_engine_pids", lambda: [99])
    assert vlm_critic.stop_targets() == [10, 11, 12, 99]


def test_descendants_follow_the_serve_pid():
    from farmgen.workers.vlm_critic import descendants_of

    tree = {10: [11], 11: [12]}
    assert descendants_of([10], children=lambda pid: tree.get(pid, [])) == [11, 12]


def test_stop_server_signals_every_target(monkeypatch):
    import signal

    from farmgen.workers import vlm_critic

    signals: list[tuple[int, int]] = []

    def kill(pid, sig):
        signals.append((pid, sig))
        if sig == 0:
            raise ProcessLookupError(pid)

    monkeypatch.setattr(vlm_critic, "stop_targets", lambda: [10, 11])
    monkeypatch.setattr(vlm_critic.os, "kill", kill)
    monkeypatch.setattr(vlm_critic.time, "sleep", lambda _seconds: None)
    vlm_critic.stop_server()
    assert signals == [(10, signal.SIGTERM), (11, signal.SIGTERM), (10, 0), (11, 0)]


def test_supervisor_signals_its_session_when_farmgen_exits():
    from farmgen.workers.vlm_critic import supervised_command

    cmd = supervised_command(["/repo/.venv/bin/vllm", "serve", "/repo/weights"])
    assert "prctl" in cmd[2]
    assert "killpg" in cmd[2]
    assert cmd[3:] == ["/repo/.venv/bin/vllm", "serve", "/repo/weights"]


def test_only_this_repos_vllm_command_is_ours():
    from farmgen.workers.vlm_critic import cmdline_is_our_vllm

    root = "/home/cmoore/workspace/farmgen"
    ours = f"{root}/.venv/bin/vllm serve {root}/third_party/Qwen3-VL-30B-A3B-Instruct"
    assert cmdline_is_our_vllm(ours, root) is True
    assert cmdline_is_our_vllm("/home/yang/miniconda3/envs/qwen_next/bin/python serve", root) is False
    assert cmdline_is_our_vllm("VLLM::EngineCore", root) is False


def test_flux_stops_qwen_before_waiting_for_the_card(monkeypatch):
    from farmgen.run import _generate_images
    from farmgen.scene import GroundSpec, Scene

    order: list[str] = []

    class _Proc:
        def terminate(self):
            order.append("comfy_stop")

        def wait(self, timeout=None):
            return 0

        def kill(self):
            return None

    monkeypatch.setattr("farmgen.run._stop_vlm", lambda: order.append("stop"))
    monkeypatch.setattr("farmgen.run._wait_gpu", lambda *_args, **_kwargs: order.append("wait"))
    monkeypatch.setattr("farmgen.run._start_comfy", lambda: _Proc())
    monkeypatch.setattr("farmgen.run.free_memory", lambda _url: None)
    monkeypatch.setattr("farmgen.run.gpu_memory_mib", lambda: 0)
    monkeypatch.setattr("farmgen.run.log_line", lambda *_args, **_kwargs: "")
    scene = Scene(
        id="farm_test",
        seed=0,
        brief="a barn",
        ground=GroundSpec(size_m=(40.0, 40.0)),
        assets=[],
    )
    _generate_images(scene, [])
    assert order == ["stop", "wait", "comfy_stop"]


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
