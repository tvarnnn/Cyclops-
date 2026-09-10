"""The builder's lifecycle under the things that actually end it.

The 2026-09-06 physical walk ended with a builder that died between
`observe()` and `stop_session()`: a lock naming a dead pid, a session record
with `ended_at: null`, no `session_stopped`, and a derived tree that the
Tower could only call `failed` although 463 keyframes of geometry were in
it. Nothing here asks whether reconstruction works; every test asks
whether the RECORD is truthful when the process is asked to stop, told to
stop, or blows up -- and whether the geometry that exists survives.

Real child processes where a request has to cross a process boundary
(stdin EOF, a console control event); in-process where an exception has to
be injected.
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests import synthetic_scene as ss
from tower.capture import CaptureRecorder
from tower.world_builder.records import (
    FINAL_SOLVE_FAILED,
    FINAL_SOLVE_SKIPPED,
    FINAL_SOLVE_UNAVAILABLE,
    FINALIZATION_COMPLETE,
    FINALIZATION_INTERRUPTED,
)
from tower.world_builder.store import WorldStore

WIDTH, HEIGHT = 160, 120
# Generous: these assert that a process EXITED, on a box that may be
# running the whole suite. Latency is not what is being measured.
EXIT_TIMEOUT_S = 90.0
SETTLE_TIMEOUT_S = 45.0


def _frames(count: int):
    scene = ss.furnished_room()
    poses = ss.strafe(count, step=0.09)
    return ss.render_sequence(scene, poses, ss.camera_matrix(WIDTH, HEIGHT), WIDTH, HEIGHT)


def _write(recorder, images, start=0):
    for offset, image in enumerate(images):
        index = start + offset
        recorder.write_frame(
            ss.encode_jpeg(image), source_seq=index, wire_seq=index, tx_seq=index,
            width=WIDTH, height=HEIGHT,
        )


@pytest.fixture
def open_capture(tmp_path):
    """A capture still being written: frames in the journal, no `ended_at`."""
    recorder = CaptureRecorder(tmp_path / "capture")
    capture_id = recorder.start()
    _write(recorder, _frames(8))
    return recorder, recorder.capture_dir(capture_id), capture_id


@pytest.fixture
def finished_capture(tmp_path):
    recorder = CaptureRecorder(tmp_path / "capture")
    capture_id = recorder.start()
    _write(recorder, _frames(10))
    recorder.stop()
    return recorder.capture_dir(capture_id), capture_id


def _new_group_flags():
    return subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0


def _spawn(capture_dir, root, *extra, stop_on_stdin_close=False):
    """Run the builder the way the Tower's supervisor does.

    `--stop-on-stdin-close` is only passed when the test intends to ASK:
    the supervisor holds the pipe's write end open for the life of the
    worker, whereas `Popen.communicate()` closes it at once -- so a test
    that passed the flag and then merely collected output would be
    sending a soft stop without meaning to.
    """
    return subprocess.Popen(
        [
            sys.executable, "scripts/world_build_session.py",
            "--follow-capture", str(capture_dir),
            "--root", str(root),
            "--rebuild-every", "2",
            "--poll-seconds", "0.1",
            # Long enough that an idle exit cannot be mistaken for a stop
            # that worked: a broken request path times the test out.
            "--max-idle-polls", "40000",
            "--format", "json",
            *(("--stop-on-stdin-close",) if stop_on_stdin_close else ()),
            *extra,
        ],
        stdin=subprocess.PIPE if stop_on_stdin_close else subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        creationflags=_new_group_flags(),
    )


def _wait_for(predicate, *, timeout=SETTLE_TIMEOUT_S, what="condition"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.1)
    raise AssertionError(f"timed out waiting for {what}")


def _the_session(root):
    store = WorldStore(root)
    worlds = store.list_world_ids()
    assert len(worlds) == 1, worlds
    world_id = worlds[0]
    sessions = store.list_session_ids(world_id)
    assert len(sessions) == 1, sessions
    return store, world_id, sessions[0]


def _keyframes_written(root) -> int:
    try:
        store, world_id, session_id = _the_session(root)
    except AssertionError:
        return 0
    path = store.keyframes_path(world_id, session_id)
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _finish(process):
    # The soft-stop tests close `process.stdin` THEMSELVES -- that closure
    # is the stop request under test -- and then ask for the output. The
    # handle has to be dropped before `communicate()` sees it: CPython's
    # `Popen._communicate` unconditionally does `self.stdin.flush()` when
    # `self.stdin` is set and communication has not started, and flushing
    # an already-closed writer raises `ValueError: I/O operation on closed
    # file`, not the `BrokenPipeError` that code is written to tolerate.
    # Worse, `communicate()` sets `_communication_started` in its own
    # `finally`, so the retry in the `finally` below then fails a second
    # time with `AttributeError: 'Popen' object has no attribute
    # '_fileobj2output'` and buries the first error.
    #
    # Detaching is what the standard library itself does once stdin is
    # done with, and it costs nothing: the descriptor is already closed,
    # and nothing here writes to the builder again.
    if process.stdin is not None and process.stdin.closed:
        process.stdin = None
    try:
        stdout, stderr = process.communicate(timeout=EXIT_TIMEOUT_S)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
    return stdout, stderr


def _report(stdout: str) -> dict:
    start = stdout.find("{")
    assert start >= 0, f"no JSON report in stdout: {stdout[-500:]}"
    return json.loads(stdout[start:])


def _events(store, world_id, session_id):
    return [
        json.loads(line)["kind"]
        for line in store.events_path(world_id, session_id).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# -- a normal walk ----------------------------------------------------------


def test_a_normal_stop_records_a_complete_finalization_and_releases_the_lock(
    finished_capture, tmp_path
):
    capture_dir, capture_id = finished_capture
    root = tmp_path / "worlds"
    process = _spawn(capture_dir, root)
    stdout, stderr = _finish(process)
    assert process.returncode == 0, stderr[-2000:]

    store, world_id, session_id = _the_session(root)
    session = store.read_session(world_id, session_id)
    assert session.end_reason == "stop"
    assert session.ended_at is not None
    assert session.finalization["state"] == FINALIZATION_COMPLETE
    # No --solve: there was no final solve to record, and None says so
    # rather than "skipped", which would claim one was planned.
    assert session.finalization["final_solve"] is None
    assert session.finalization["updated_at"] >= session.finalization["started_at"]
    assert not store.lock_path(world_id).exists()
    assert "session_stopped" in _events(store, world_id, session_id)
    assert store.read_derived_manifest(world_id) is not None
    report = _report(stdout)
    assert report["end_reason"] == "stop"
    assert report["finalization"] == FINALIZATION_COMPLETE


# -- asked to stop mid-walk -------------------------------------------------


def test_a_soft_stop_mid_walk_closes_the_session_as_interrupted_and_keeps_the_build(
    open_capture, tmp_path
):
    """The wearer left World Builder (or the cartridge was stopped) while
    the capture was still open. Before 2026-09-06 nothing could ask; the
    builder was terminated at the supervisor's grace and left a lock
    naming a dead pid. Now stdin EOF is the request, and the record says
    exactly what happened."""
    recorder, capture_dir, capture_id = open_capture
    root = tmp_path / "worlds"
    process = _spawn(capture_dir, root, stop_on_stdin_close=True)
    try:
        _wait_for(lambda: _keyframes_written(root) >= 2, what="the first keyframes")
        assert process.poll() is None, "the builder exited before it could be asked"
        process.stdin.close()
        stdout, stderr = _finish(process)
    finally:
        recorder.stop()
    assert process.returncode == 0, stderr[-2000:]

    store, world_id, session_id = _the_session(root)
    session = store.read_session(world_id, session_id)
    assert session.end_reason == "interrupted"
    assert session.ended_at is not None
    # Finalization COMPLETED -- the final build ran -- for a session that
    # was INTERRUPTED. The two facts are recorded apart because they are
    # different facts.
    assert session.finalization["state"] == FINALIZATION_COMPLETE
    assert not store.lock_path(world_id).exists()
    kinds = _events(store, world_id, session_id)
    assert kinds[-1] == "session_stopped"
    manifest = store.read_derived_manifest(world_id)
    assert manifest is not None and manifest["session_id"] == session_id
    assert manifest["keyframes"] >= 2
    assert _report(stdout)["end_reason"] == "interrupted"


def test_a_soft_stop_mid_walk_still_runs_the_final_solve(open_capture, tmp_path):
    """The wearer leaving World Builder must not cost them the world.

    This test asserted the OPPOSITE until 2026-09-09, on the reasoning
    recorded in `StopRequest`: a soft stop means "you are no longer wanted
    for new frames", so the final solve was skipped because "nobody is
    waiting for it".

    The field walk falsifies the premise. Nobody watches a final solve;
    they open the world afterwards, and the final solve is what makes the
    world worth opening. Re-running the one that walk never got, on its own
    images, took 16 components to 6 and put 652 of 795 keyframes into one
    component at 0.82 px -- against a largest component of 156 without it.

    A HARD stop still skips: the Tower is going down, and the solve would
    be killed mid-run anyway. That is the test below this one.
    """
    recorder, capture_dir, capture_id = open_capture
    root = tmp_path / "worlds"
    # A stub that reports itself rather than solving, so what is measured
    # is whether the builder RAN it, not how fast pycolmap is.
    stub = tmp_path / "marker_solve.py"
    stub.write_text(
        "import json, sys\n"
        "print(json.dumps({'solved': False, 'reason': 'stub'}))\n",
        encoding="utf-8",
    )
    process = _spawn(
        capture_dir, root, "--solve", "--solve-every", "0", "--solve-script", str(stub),
        stop_on_stdin_close=True,
    )
    try:
        _wait_for(lambda: _keyframes_written(root) >= 2, what="the first keyframes")
        process.stdin.close()
        stdout, stderr = _finish(process)
    finally:
        recorder.stop()
    assert process.returncode == 0, stderr[-2000:]

    store, world_id, session_id = _the_session(root)
    record = store.read_session(world_id, session_id).finalization
    assert record["state"] == FINALIZATION_COMPLETE
    report = _report(stdout)
    # The solve was ATTEMPTED. What it returned is the stub's business.
    assert report["global_solve"]["attempted"] is True, (
        "a soft stop skipped the final solve; that is the 2026-09-09 defect"
    )
    assert record["final_solve"] != FINAL_SOLVE_SKIPPED


# -- told to stop during finalization --------------------------------------


def _hard_stop(process):
    if os.name == "nt":
        os.kill(process.pid, signal.CTRL_BREAK_EVENT)
    else:
        process.terminate()


def test_a_hard_stop_during_the_final_solve_ends_the_child_and_still_builds(
    finished_capture, tmp_path
):
    """The Tower is shutting down while the final solve (30-135 s on the
    real host) is running. The builder must not be shot mid-write: it
    terminates its own solve child, writes the final build from the last
    solution it has, records that the final solve was skipped and why, and
    exits within a build -- not within the solve."""
    capture_dir, capture_id = finished_capture
    root = tmp_path / "worlds"
    marker = tmp_path / "solve-started"
    stub = tmp_path / "slow_solve.py"
    stub.write_text(
        "import pathlib, time\n"
        f"pathlib.Path({str(marker)!r}).write_text('started')\n"
        "time.sleep(120)\n",
        encoding="utf-8",
    )
    process = _spawn(
        capture_dir, root, "--solve", "--solve-every", "0", "--solve-script", str(stub)
    )
    try:
        _wait_for(marker.exists, what="the final solve child to start")
        import psutil

        children = psutil.Process(process.pid).children(recursive=True)
        assert children, "the final solve should be a child of the builder"
        _hard_stop(process)
        started = time.monotonic()
        stdout, stderr = _finish(process)
    finally:
        pass
    assert process.returncode == 0, stderr[-2000:]
    assert time.monotonic() - started < 45.0, "the builder waited for the solve it was told to abandon"
    _wait_for(
        lambda: not any(child.is_running() for child in children),
        timeout=15.0, what="the solve child to be gone",
    )

    store, world_id, session_id = _the_session(root)
    session = store.read_session(world_id, session_id)
    assert session.end_reason == "stop"
    assert session.finalization["state"] == FINALIZATION_COMPLETE
    assert session.finalization["final_solve"] == FINAL_SOLVE_SKIPPED
    assert "hard stop" in session.finalization["detail"]
    assert not store.lock_path(world_id).exists()
    assert store.read_derived_manifest(world_id) is not None


# -- the final solve's own outcome ----------------------------------------


def test_a_solve_child_starts_while_the_supervisor_still_holds_the_stop_pipe(
    finished_capture, tmp_path
):
    """The supervisor keeps the builder's stdin pipe open for the life of
    the worker, and the builder keeps a thread blocked reading it. A child
    that inherited that handle would not start until the pipe closed
    (Windows serialises operations on a synchronous file object with an
    I/O in flight): measured at 90 s of a 0-CPU child before this test
    existed. The solve children must therefore be spawned without it."""
    capture_dir, capture_id = finished_capture
    root = tmp_path / "worlds"
    stub = tmp_path / "quick_solve.py"
    stub.write_text(
        "import json\nprint(json.dumps({'solved': False, 'reason': 'stub'}))\n",
        encoding="utf-8",
    )
    process = _spawn(
        capture_dir, root, "--solve", "--solve-every", "0", "--solve-script", str(stub),
        stop_on_stdin_close=True,
    )
    started = time.monotonic()
    try:
        # stdin stays OPEN, exactly as the supervisor holds it.
        _wait_for(lambda: process.poll() is not None, timeout=40.0,
                  what="the builder to finish with its stop pipe still open")
    finally:
        stdout, stderr = _finish(process)
    assert process.returncode == 0, stderr[-2000:]
    assert time.monotonic() - started < 40.0
    store, world_id, session_id = _the_session(root)
    record = store.read_session(world_id, session_id).finalization
    assert record["final_solve"] == FINAL_SOLVE_UNAVAILABLE
    assert "stub" in record["detail"]


def test_a_final_solve_that_finds_nothing_is_recorded_as_unavailable(
    finished_capture, tmp_path
):
    capture_dir, capture_id = finished_capture
    root = tmp_path / "worlds"
    stub = tmp_path / "no_solver.py"
    stub.write_text(
        "import json\nprint(json.dumps({'solved': False, 'reason': 'pycolmap is not installed'}))\n",
        encoding="utf-8",
    )
    process = _spawn(
        capture_dir, root, "--solve", "--solve-every", "0", "--solve-script", str(stub)
    )
    stdout, stderr = _finish(process)
    assert process.returncode == 0, stderr[-2000:]
    store, world_id, session_id = _the_session(root)
    record = store.read_session(world_id, session_id).finalization
    assert record["state"] == FINALIZATION_COMPLETE
    assert record["final_solve"] == FINAL_SOLVE_UNAVAILABLE
    assert "pycolmap is not installed" in record["detail"]
    assert _report(stdout)["global_solve"]["reason"] == "pycolmap is not installed"


def test_a_final_solve_child_that_crashes_is_recorded_as_failed_not_fatal(
    finished_capture, tmp_path
):
    capture_dir, capture_id = finished_capture
    root = tmp_path / "worlds"
    stub = tmp_path / "crash.py"
    stub.write_text("import sys\nsys.exit(3)\n", encoding="utf-8")
    process = _spawn(
        capture_dir, root, "--solve", "--solve-every", "0", "--solve-script", str(stub)
    )
    stdout, stderr = _finish(process)
    assert process.returncode == 0, stderr[-2000:]
    store, world_id, session_id = _the_session(root)
    record = store.read_session(world_id, session_id).finalization
    assert record["state"] == FINALIZATION_COMPLETE
    assert record["final_solve"] == FINAL_SOLVE_FAILED
    assert "exited 3" in record["detail"]
    assert store.read_derived_manifest(world_id) is not None


# -- an exception mid-walk (in process) ------------------------------------


def test_an_exception_mid_walk_closes_the_session_as_error_and_keeps_the_build(
    finished_capture, tmp_path, monkeypatch
):
    """The 2026-09-06 failure shape, reproduced by injection: the loop
    raises after a few keyframes. The record must say `error`, the lock
    must be gone, and whatever was built must be built."""
    from scripts import world_build_session as builder
    from tower.world_builder.engine import WorldBuilderEngine

    # Signal handlers belong to the child process in every other test
    # here; installing them in the test runner would outlive this test.
    monkeypatch.setattr(builder.StopRequest, "install", lambda self, **_: None)

    calls = {"n": 0}
    real_observe = WorldBuilderEngine.observe

    def observe_then_blow_up(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 4:
            raise RuntimeError("boom")
        return real_observe(self, *args, **kwargs)

    monkeypatch.setattr(WorldBuilderEngine, "observe", observe_then_blow_up)

    capture_dir, _ = finished_capture
    root = tmp_path / "worlds"
    code = builder.main(
        ["--frames", str(capture_dir / "frames"), "--root", str(root), "--rebuild-every", "2",
         "--format", "json"]
    )
    assert code == 1

    store, world_id, session_id = _the_session(root)
    session = store.read_session(world_id, session_id)
    assert session.end_reason == "error"
    assert session.ended_at is not None
    assert session.finalization["state"] == FINALIZATION_INTERRUPTED
    assert session.finalization["detail"] == "RuntimeError: boom"
    assert not store.lock_path(world_id).exists()
    assert _events(store, world_id, session_id)[-1] == "session_stopped"
    # Three frames were observed before the fourth raised; whatever they
    # produced is on disk as a derived tree, not lost with the process.
    assert store.read_derived_manifest(world_id) is not None


def test_a_stop_request_records_the_channel_it_came_from():
    from scripts.world_build_session import StopRequest

    request = StopRequest()
    assert not request.asked and not request.hard
    request.request(StopRequest.SOFT, "stdin-closed")
    assert request.asked and not request.hard and request.source == "stdin-closed"
    request.request(StopRequest.HARD, "SIGBREAK")
    assert request.hard and request.source == "SIGBREAK"
    # A soft request never lowers a hard one.
    request.request(StopRequest.SOFT, "stdin-closed")
    assert request.hard and request.source == "SIGBREAK"
    assert list(request.bounded(iter([1, 2, 3]))) == []


def test_stopping_the_capture_and_the_workspace_together_is_an_ordinary_stop(
    open_capture, tmp_path
):
    """The ordinary way a walk ends, and the way it used to be mislabelled.

    iOS posts `session/stop` from `.onDisappear`, so the wearer tapping Stop
    and then leaving the World Builder screen sends the capture's end and a
    SOFT stop within milliseconds of each other. Every stop request used to
    make the session `interrupted`, and `results/world_builder.py` maps that
    to Interrupted before it ever looks at `finalization` -- so a walk that
    solved, registered and finalized perfectly was shown as a failure.

    Measured on a real 12 fps capture before the fix: a wearer who left
    immediately got `interrupted`, one who lingered a second got `stop`, on
    identical geometry. That is the 2026-09-09 symptom, and it survived the
    first two fixes because they changed whether the final solve RAN, not
    what `end_reason` recorded.

    The question the builder asks now is whether the CAPTURE had finished,
    not whether anyone had asked this process to go.
    """
    recorder, capture_dir, _capture_id = open_capture
    root = tmp_path / "worlds"
    process = _spawn(capture_dir, root, stop_on_stdin_close=True)
    try:
        _wait_for(lambda: _keyframes_written(root) >= 2, what="the first keyframes")
        # Both at once, no wait between them: the capture ends and the
        # workspace goes away in the same gesture.
        recorder.stop()
        process.stdin.close()
        stdout, stderr = _finish(process)
    finally:
        pass
    assert process.returncode == 0, stderr[-2000:]

    store, world_id, session_id = _the_session(root)
    session = store.read_session(world_id, session_id)
    assert session.end_reason == "stop", (
        "a normal Stop was recorded as an interruption; this is the 2026-09-09 label"
    )
    assert session.finalization["state"] == FINALIZATION_COMPLETE
    # And the classifier the phone actually reads agrees. Asserting only
    # `end_reason` would pass a fix that never reached the label: the
    # mapping at `results/world_builder.py` tests `end_reason` BEFORE it
    # looks at `finalization`, which is why a complete finalization could
    # not rescue an interrupted-looking walk.
    from tower.results.world_builder import _lifecycle

    lifecycle = _lifecycle(
        holder=None, stopped=True, session=session,
        geometry_current=True, has_manifest=True,
    )
    assert lifecycle["state"] != "interrupted", lifecycle


def test_a_stop_while_the_capture_is_still_open_is_still_an_interruption(
    open_capture, tmp_path
):
    """The other half, which must not be lost to the fix above.

    Frames were still being written and somebody asked this process to go.
    That IS an interruption, and saying otherwise would make the label
    useless in the case it exists for.
    """
    recorder, capture_dir, _capture_id = open_capture
    root = tmp_path / "worlds"
    process = _spawn(capture_dir, root, stop_on_stdin_close=True)
    try:
        _wait_for(lambda: _keyframes_written(root) >= 2, what="the first keyframes")
        process.stdin.close()          # the capture is left OPEN
        stdout, stderr = _finish(process)
    finally:
        recorder.stop()
    assert process.returncode == 0, stderr[-2000:]

    store, world_id, session_id = _the_session(root)
    assert store.read_session(world_id, session_id).end_reason == "interrupted"
