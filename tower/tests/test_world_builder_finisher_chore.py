"""The finisher chore: a stall is killed and counted, and owed work gets run.

TWO WAYS A WORLD SAID "IMPROVING" WITH NOTHING IMPROVING IT.

1. **A run that was alive and doing nothing.** On 2026-09-22 the recovery
   finisher held world 2f447162 for 95 minutes at 0% CPU. Its pid was live,
   so the lifecycle said `running` for as long as the Tower was up. Nothing
   watched it. `_BackgroundChore` now samples the child's whole process tree
   and kills a run that has used almost no CPU for ten minutes -- killed
   OUTRIGHT, because the polite stop (closing the pipe) is what the finisher
   answers by giving its attempt back, and a stall has to count.

2. **Work that was owed with nothing scheduled to do it.** The chore ran once
   per Tower START and was retired by the first stream. Work interrupted by
   a walk, a second owed world past `--max-worlds 1`, a stalled run -- all of
   it waited for the next restart, however many days away, and the world
   read "Improving" throughout. The chore now runs again whenever the Tower
   is idle and work may still be owed.

What must NOT change is the old safety argument: a walk never finds the
chore running. The idle rule is what keeps that true, and it is pinned here
from both sides -- a held stream, and a live capture worker.

Almost everything below uses a fake clock and a fake CPU probe, so the
ten-minute rule is exercised in milliseconds. The last class uses the real
probe, on real children, with a short window.
"""

import os
import subprocess
import sys
import time

import pytest

from tower import main as tower_main
from tower.capture_workers import WorkerSpec

# A child that does nothing, for as long as it is allowed to.
_IDLE_CHILD = (sys.executable, "-c", "import time; time.sleep(600)")


class _Clock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class _Cpu:
    """A CPU reading the test controls: `value` is what the probe says."""

    def __init__(self, value: float = 0.5) -> None:
        self.value = value

    def __call__(self, process):
        return self.value


def _chore(argv=_IDLE_CHILD, *, idle=None, clock=None, cpu=None, **kwargs):
    return tower_main._BackgroundChore(
        WorkerSpec(argv=argv, name="probe", stop_via_stdin=True, stop_grace_seconds=0.5),
        idle=idle,
        clock=clock or _Clock(),
        cpu_probe=cpu or _Cpu(),
        **kwargs,
    )


@pytest.fixture
def chores():
    """Every chore a test makes is closed at the end, so no child outlives it."""
    made = []
    yield made
    for chore in made:
        chore.close("test over")


def _advance(chore, clock, seconds, *, step=15.0, until=None):
    """Tick through `seconds` of fake time at the real tick rate.

    In steps, never one jump: a jump far longer than a tick is exactly what
    the chore reads as a machine that slept, and would reset the window.
    """
    target = clock.now + seconds
    while clock.now < target:
        clock.now = min(target, clock.now + step)
        chore.tick()
        if until is not None and until():
            return


def _wait_dead(process, timeout=20.0):
    process.wait(timeout=timeout)
    return process.poll() is not None


class TestAStallIsKilledAndCounted:
    def test_a_run_with_no_cpu_progress_is_killed_after_the_window(self, chores, monkeypatch):
        clock, cpu = _Clock(), _Cpu(1.64)
        chore = _chore(clock=clock, cpu=cpu, stall_seconds=600, stall_cpu_seconds=1.0)
        chores.append(chore)

        # THE POLITE STOP MUST NOT PRECEDE THE KILL. Closing the pipe is how
        # the finisher is asked to stop, and it answers by giving its attempt
        # back -- which is exactly what a stall must not get.
        pipe_state_at_kill = []
        real_terminate_tree = tower_main.terminate_tree

        def spy(process, **kwargs):
            pipe_state_at_kill.append(process.stdin.closed)
            return real_terminate_tree(process, **kwargs)

        monkeypatch.setattr(tower_main, "terminate_tree", spy)

        assert chore.start() is True
        process = chore._process
        _advance(chore, clock, 590.0)
        assert chore._process is process, "killed inside the window"
        _advance(chore, clock, 60.0, until=lambda: chore._process is None)
        assert chore._process is None, "a run that did nothing for ten minutes was left alive"
        assert _wait_dead(process)
        assert pipe_state_at_kill == [False], "the stall was stopped politely, so it was forgiven"

        snapshot = chore.snapshot()
        assert snapshot["state"] == "owed"
        assert snapshot["stalls"] == 1
        assert snapshot["last"]["outcome"] == "stalled"
        assert snapshot["last"]["quiet_seconds"] >= 600

    def test_a_run_that_keeps_using_cpu_is_never_killed_however_long(self, chores):
        """A long walk is a long build. The rule is about CPU, not about time."""
        clock, cpu = _Clock(), _Cpu(0.0)
        chore = _chore(clock=clock, cpu=cpu, stall_seconds=600, stall_cpu_seconds=1.0)
        chores.append(chore)
        assert chore.start() is True
        process = chore._process
        for _ in range(8 * 60 * 4):  # eight hours, 15 s at a time
            clock.now += 15.0
            cpu.value += 0.3  # 2% of a core: a slow build, but a build
            chore.tick()
        assert chore._process is process and process.poll() is None
        assert chore.snapshot()["stalls"] == 0

    def test_a_machine_that_slept_is_not_a_stall(self, chores):
        """A suspended machine uses no CPU, and the monotonic clock ran on."""
        clock, cpu = _Clock(), _Cpu(3.0)
        chore = _chore(clock=clock, cpu=cpu, stall_seconds=600)
        chores.append(chore)
        assert chore.start() is True
        process = chore._process
        chore.tick()
        clock.now += 2 * 3600.0  # a lid closed for two hours: one late tick
        chore.tick()
        _advance(chore, clock, 300.0)
        # The SAME process, and no stall on record: a kill followed by a
        # respawn inside the window would also leave "a process" running.
        assert chore._process is process, "a sleep was read as a stall"
        assert chore.snapshot()["stalls"] == 0
        # And the window still works afterwards, measured from the wake.
        _advance(chore, clock, 400.0, until=lambda: chore._process is None)
        assert chore._process is None

    def test_a_grandchild_that_finishes_is_progress_not_a_stall(self, chores):
        """The tree SHRINKS when the appearance child exits, and its CPU with it."""
        clock, cpu = _Clock(), _Cpu(500.0)
        chore = _chore(clock=clock, cpu=cpu, stall_seconds=600)
        chores.append(chore)
        assert chore.start() is True
        chore.tick()
        _advance(chore, clock, 585.0)
        cpu.value = 20.0  # the busy grandchild left, and took 480 s of CPU with it
        _advance(chore, clock, 45.0)  # 630 s after the last GROWTH
        assert chore._process is not None

    def test_after_a_stall_it_runs_again_only_after_the_backoff(self, chores):
        clock, cpu = _Clock(), _Cpu(1.0)
        chore = _chore(
            clock=clock, cpu=cpu, stall_seconds=600, retry_backoff=(60.0, 300.0)
        )
        chores.append(chore)
        assert chore.start() is True
        chore.tick()
        _advance(chore, clock, 700.0, until=lambda: chore._process is None)
        assert chore._process is None
        _advance(chore, clock, 45.0)
        assert chore._process is None, "re-ran inside the backoff"
        _advance(chore, clock, 30.0)
        assert chore._process is not None, "owed work was never tried again"
        assert chore.snapshot()["runs"] == 2


class TestOwedWorkGetsRun:
    def test_a_yield_leaves_the_work_owed_and_it_runs_again_when_idle(self, chores):
        clock = _Clock()
        chore = _chore(clock=clock, quiet_seconds=120.0)
        chores.append(chore)
        assert chore.start() is True
        first = chore._process
        chore.stop("a capture opened (cap-1)")
        assert _wait_dead(first)
        assert chore.snapshot()["state"] == "owed"

        chore.tick()
        assert chore._process is None, "ran again inside the quiet period"
        clock.now += 121.0
        chore.tick()
        assert chore._process is not None and chore._process is not first

    def test_a_held_stream_keeps_it_stopped_until_released(self, chores):
        clock = _Clock()
        chore = _chore(clock=clock, quiet_seconds=120.0)
        chores.append(chore)
        assert chore.start() is True
        first = chore._process
        owner = object()
        chore.hold(owner, "a stream opened")
        assert _wait_dead(first)
        clock.now += 3600.0
        chore.tick()
        assert chore._process is None, "ran while a stream was open"

        chore.release(owner)
        chore.tick()
        assert chore._process is None, "ran the instant the stream closed"
        clock.now += 121.0
        chore.tick()
        assert chore._process is not None

    def test_a_connection_that_never_streamed_does_not_postpone_it(self, chores):
        """Every disconnect releases; most connections never held anything."""
        clock = _Clock()
        chore = _chore(clock=clock, quiet_seconds=120.0)
        chores.append(chore)
        chore.stop("a capture opened")
        _advance(chore, clock, 100.0)
        for _ in range(10):  # a poller reconnecting, well inside the window
            chore.release(object())
        _advance(chore, clock, 30.0)
        assert chore._process is not None, "a stranger's disconnect postponed owed work"

    def test_a_live_capture_worker_keeps_it_stopped(self, chores):
        """The builder runs its surface for minutes AFTER its stream ends."""
        clock, busy = _Clock(), {"value": True}
        chore = _chore(clock=clock, idle=lambda: not busy["value"])
        chores.append(chore)
        assert chore.start() is False
        clock.now += 3600.0
        chore.tick()
        assert chore._process is None
        busy["value"] = False
        chore.tick()
        assert chore._process is not None

    def test_an_idle_probe_that_raises_counts_as_busy(self, chores):
        def broken():
            raise RuntimeError("supervisor unavailable")

        chore = _chore(idle=broken)
        chores.append(chore)
        assert chore.start() is False
        assert chore._process is None

    def test_a_clean_exit_goes_quiet(self, chores):
        clock = _Clock()
        chore = _chore((sys.executable, "-c", "pass"), clock=clock)
        chores.append(chore)
        assert chore.start() is True
        _wait_dead(chore._process)
        chore.tick()
        assert chore.snapshot()["state"] == "idle"
        assert chore.snapshot()["last"]["outcome"] == "finished"
        clock.now += 24 * 3600.0
        chore.tick()
        assert chore._process is None, "a finisher with nothing to do was run again"
        assert chore.snapshot()["runs"] == 1

    def test_more_owed_runs_again_without_waiting(self, chores):
        clock = _Clock()
        chore = _chore(
            (sys.executable, "-c", f"raise SystemExit({tower_main.CHORE_EXIT_MORE_OWED})"),
            clock=clock,
        )
        chores.append(chore)
        assert chore.start() is True
        _wait_dead(chore._process)
        chore.tick()
        assert chore.snapshot()["last"]["outcome"] == "finished-more-owed"
        chore.tick()
        assert chore._process is not None, "the second owed world waited"
        assert chore.snapshot()["runs"] == 2

    def test_a_failed_exit_is_retried_after_a_backoff(self, chores):
        clock = _Clock()
        chore = _chore(
            (sys.executable, "-c", "raise SystemExit(1)"),
            clock=clock,
            retry_backoff=(60.0,),
        )
        chores.append(chore)
        assert chore.start() is True
        _wait_dead(chore._process)
        chore.tick()
        assert chore.snapshot()["last"]["outcome"] == "exited"
        chore.tick()
        assert chore._process is None
        clock.now += 61.0
        chore.tick()
        assert chore._process is not None

    def test_close_is_final(self, chores):
        clock = _Clock()
        chore = _chore(clock=clock)
        assert chore.start() is True
        process = chore._process
        chore.close("the Tower is shutting down")
        assert _wait_dead(process)
        clock.now += 3600.0
        chore.tick()
        assert chore._process is None
        assert chore.start() is False
        assert chore.snapshot()["state"] == "closed"

    def test_a_yield_before_the_start_prevents_it_and_leaves_the_work_owed(self, chores):
        clock = _Clock()
        chore = _chore(clock=clock, quiet_seconds=120.0)
        chores.append(chore)
        chore.stop("a capture opened while the Tower was starting")
        assert chore.start() is False
        clock.now += 121.0
        chore.tick()
        assert chore._process is not None

    def test_the_watch_thread_is_only_started_by_watch(self, chores):
        chore = _chore()
        chores.append(chore)
        chore.start()
        assert chore._monitor is None
        chore.watch()
        assert chore._monitor is not None and chore._monitor.is_alive()
        chore.close("done")
        chore._monitor.join(timeout=5)
        assert not chore._monitor.is_alive()


class TestTheStreamHoldsIt:
    def test_stream_start_holds_and_stream_stop_and_disconnect_release(
        self, tmp_path, monkeypatch
    ):
        import base64
        import io

        from fastapi.testclient import TestClient
        from PIL import Image

        from tower.main import create_app

        monkeypatch.delenv("TOWER_CAPTURE_ROOT", raising=False)
        monkeypatch.setenv("TOWER_WORLD_ROOT", str(tmp_path))
        calls = []

        class _Chore:
            def start(self):
                return False

            def stop(self, reason, **kwargs):
                calls.append(("stop", reason))

            def hold(self, owner, reason):
                calls.append(("hold", id(owner)))

            def release(self, owner):
                calls.append(("release", id(owner)))

        app = create_app()
        app.state.world_finish_chore = _Chore()
        buffer = io.BytesIO()
        Image.new("RGB", (32, 24), (10, 20, 30)).save(buffer, format="JPEG")
        frame = {
            "type": "frame", "seq": 1, "source_seq": 1,
            "width": 32, "height": 24, "format": "jpeg",
            "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
        }
        with TestClient(app) as client, client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "stream_start"})
            ws.send_json(frame)
            assert ws.receive_json()["type"] == "frame_result"
            ws.send_json({"type": "stream_stop"})
            ws.send_json({"type": "ping"})
            assert ws.receive_json()["type"] == "pong"
            # Released by the STOP, while the phone is still connected: a
            # phone that stays on the result screen after a walk is the
            # ordinary case, and the disconnect may be hours away.
            at_stop = [c for c in calls if c[0] == "release"]
            assert at_stop, "stream_stop did not release the chore"

        holds = [c for c in calls if c[0] == "hold"]
        releases = [c for c in calls if c[0] == "release"]
        assert len(holds) == 1
        assert calls.index(holds[0]) < calls.index(releases[0])
        # And again, harmlessly, at the disconnect -- the path a phone that
        # vanishes without a stream_stop takes.
        assert len(releases) == 2 and all(r[1] == holds[0][1] for r in releases)

    def test_health_reports_the_chore(self, monkeypatch, tmp_path):
        from fastapi.testclient import TestClient

        from tower.main import create_app

        app = create_app()
        app.state.world_finish_chore = _chore()
        body = TestClient(app).get("/health").json()
        assert body["background_chore"]["name"] == "probe"
        assert body["background_chore"]["state"] in ("owed", "idle", "running")

        app.state.world_finish_chore = None
        assert TestClient(app).get("/health").json()["background_chore"] is None


class TestTheRealProbeOnRealChildren:
    """`_tree_cpu_seconds`, and the rule, against processes that really idle
    and really work. A short window keeps it to seconds."""

    def _drive(self, chore, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline and chore._process is not None:
            chore.tick()
            time.sleep(0.25)

    def test_a_child_blocked_forever_is_found_stalled(self, chores):
        # Blocked in a wait, the way the deadlocked finisher was blocked in
        # a critical section: alive, and using nothing.
        chore = tower_main._BackgroundChore(
            WorkerSpec(
                argv=(sys.executable, "-c", "import threading; threading.Event().wait()"),
                name="blocked", stop_via_stdin=True, stop_grace_seconds=0.5,
            ),
            stall_seconds=4.0, stall_cpu_seconds=0.5, tick_seconds=0.25,
        )
        chores.append(chore)
        assert chore.start() is True
        process = chore._process
        time.sleep(1.5)  # let interpreter start-up CPU land before the baseline
        self._drive(chore, 20.0)
        assert chore._process is None, "a child using no CPU was never found stalled"
        assert _wait_dead(process)
        assert chore.snapshot()["last"]["outcome"] == "stalled"

    def test_a_child_that_computes_is_left_alone(self, chores):
        chore = tower_main._BackgroundChore(
            WorkerSpec(
                argv=(sys.executable, "-c", "while True: sum(range(10000))"),
                name="busy", stop_via_stdin=True, stop_grace_seconds=0.5,
            ),
            stall_seconds=4.0, stall_cpu_seconds=0.5, tick_seconds=0.25,
        )
        chores.append(chore)
        assert chore.start() is True
        self._drive(chore, 8.0)
        assert chore._process is not None and chore._process.poll() is None
        assert chore.snapshot()["stalls"] == 0

    def test_the_probe_counts_the_grandchildren(self):
        """A parent waiting on a busy child is not stalled."""
        parent = subprocess.Popen(
            [sys.executable, "-c",
             "import subprocess, sys; subprocess.run([sys.executable, '-c', "
             "'import time\\nt=time.time()\\nwhile time.time()-t<6: sum(range(10000))'])"],
        )
        import psutil

        family = []
        try:
            time.sleep(1.0)
            family = psutil.Process(parent.pid).children(recursive=True)
            first = tower_main._tree_cpu_seconds(parent)
            time.sleep(2.5)
            second = tower_main._tree_cpu_seconds(parent)
            assert first is not None and second is not None
            assert second - first > 1.0, (first, second)
        finally:
            # The grandchild first, collected while its parent still named
            # it: a plain kill of the parent leaves it running on Windows.
            for child in family:
                try:
                    child.kill()
                except Exception:  # noqa: BLE001 -- already gone
                    pass
            parent.kill()
            parent.wait(timeout=10)


class TestTheReviewOfTheChore:
    """2026-09-23: what a process-lifecycle reviewer broke, turned round."""

    def test_a_stall_is_killed_hard(self, chores, monkeypatch):
        """No SIGTERM on POSIX: its handler is the finisher's forgiveness."""
        clock, cpu = _Clock(), _Cpu(1.0)
        chore = _chore(clock=clock, cpu=cpu, stall_seconds=600)
        chores.append(chore)
        calls = []
        real = tower_main.terminate_tree

        def spy(process, **kwargs):
            calls.append(kwargs.get("hard"))
            return real(process, **kwargs)

        monkeypatch.setattr(tower_main, "terminate_tree", spy)
        assert chore.start() is True
        chore.tick()
        _advance(chore, clock, 700.0, until=lambda: chore._process is None)
        assert calls == [True]

    def test_a_finisher_that_survives_its_kill_blocks_the_next_one(
        self, chores, monkeypatch
    ):
        clock = _Clock()
        chore = _chore(clock=clock, quiet_seconds=0.0)
        chores.append(chore)
        # A kill that does not take: the tree is "not gone" and still alive.
        monkeypatch.setattr(tower_main, "terminate_tree", lambda *a, **k: False)
        assert chore.start() is True
        first = chore._process
        chore.stop("a capture opened")
        assert first.poll() is None, "the test needs the child still alive"
        assert chore.snapshot()["lingering_pid"] == first.pid
        _advance(chore, clock, 3600.0)
        assert chore._process is None, "a second finisher started beside a live one"
        first.kill()
        first.wait(timeout=20)
        chore.tick()
        assert chore._process is not None
        assert chore.snapshot()["lingering_pid"] is None

    def test_waiting_is_retried_after_a_backoff(self, chores):
        clock = _Clock()
        chore = _chore(
            (sys.executable, "-c", f"raise SystemExit({tower_main.CHORE_EXIT_WAITING})"),
            clock=clock, retry_backoff=(60.0,),
        )
        chores.append(chore)
        assert chore.start() is True
        _wait_dead(chore._process)
        chore.tick()
        assert chore.snapshot()["last"]["outcome"] == "waiting"
        chore.tick()
        assert chore._process is None
        clock.now += 61.0
        chore.tick()
        assert chore._process is not None

    def test_the_tower_passes_a_forgiveness_bound_sized_for_idle_runs(self):
        from dataclasses import replace

        settings = replace(
            tower_main.get_settings(), world_root="data/world_builder",
            world_finish_pending=True, world_surface=True, world_solve=True,
        )
        argv = list(tower_main._world_finish_spec(settings).argv)
        assert argv[argv.index("--max-forgiven") + 1] == str(tower_main.CHORE_MAX_FORGIVEN)
        assert tower_main.CHORE_MAX_FORGIVEN > 5


@pytest.mark.skipif(os.name == "nt", reason="Windows kills never run a handler")
def test_a_hard_terminate_runs_no_sigterm_handler(tmp_path):
    from tower.process_ownership import terminate_tree

    marker = tmp_path / "handled"
    child = subprocess.Popen([sys.executable, "-c", (
        "import signal, sys, time, pathlib\n"
        f"def h(*_): pathlib.Path({str(marker)!r}).write_text('x'); sys.exit(0)\n"
        "signal.signal(signal.SIGTERM, h)\n"
        "print('ready', flush=True)\n"
        "time.sleep(60)\n"
    )], stdout=subprocess.PIPE, text=True)
    assert child.stdout.readline().strip() == "ready"
    assert terminate_tree(child, timeout=10, hard=True)
    assert not marker.exists(), "the SIGTERM handler ran: a stall would be forgiven"
