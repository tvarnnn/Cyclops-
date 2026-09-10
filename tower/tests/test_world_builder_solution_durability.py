"""Publishing and reading `solution.npz`, the way the 2026-09-09 walk did it.

That walk ended with `BadZipFile: File is not a zip file` on a world holding
795 keyframes, 643 positioned poses and 26,634 points. Nothing about the
capture was wrong; the failure was entirely in how one DERIVED file was
published and read.

Two defects produced it, and these tests pin both.

1.  `write_solution` handed the FINAL path to `np.savez_compressed`, so the
    archive was built up in place. A `.npz` is a zip, a zip is only valid
    once its central directory lands last, and `open(..., "wb")` truncates
    the destination to zero at the start of a write measured in hundreds of
    milliseconds. The reader lives in a DIFFERENT PROCESS -- the builder
    rebuilding its derived tree while the solver child it launched from
    `BackgroundSolver.maybe_launch` is still writing -- so no lock in either
    process could have closed the window.

2.  `load_solution` promised in its own docstring that it "never raises: an
    unreadable or half-written solution is absent", and caught
    `(OSError, KeyError, ValueError, json.JSONDecodeError)`. Measured over
    15 s of the real race, the torn read raises `EOFError` (48,854),
    `zipfile.BadZipFile` (10,295) and two further `BadZipFile` messages --
    and `load_solution` returned None exactly ZERO times. `EOFError`
    descends from Exception and `BadZipFile` from Exception alone; neither
    is an `OSError`. One escaped, reached the builder's `BaseException`
    `BaseException` handler, and turned a rebuildable
    derived file into `finalization.interrupted` on the whole session.
"""

import io
import json
import threading
import zipfile

import numpy as np
import pytest

from tower.storage import write_bytes_atomic
from tower.world_builder import global_solve
from tower.world_builder.global_solve import Solution, load_solution, write_solution


class _Store:
    """The one method `workspace_for` actually asks for."""

    def __init__(self, root):
        self.root = root

    def world_dir(self, world_id: str):
        return self.root / world_id


def _solution(points: int = 64) -> Solution:
    rng = np.random.default_rng(0)
    return Solution(
        solver="test",
        solved_at=1788999871.0,
        input_digest="digest",
        keyframe_ids=[f"s:{i:08d}" for i in range(4)],
        poses={},
        components=[],
        xyz=rng.random((points, 3)).astype(np.float32),
        rgb=rng.integers(0, 255, (points, 3)).astype(np.uint8),
        component=np.zeros(points, np.int32),
        first_keyframe=np.zeros(points, np.int32),
        track_length=np.full(points, 3, np.int32),
        error=np.zeros(points, np.float32),
        observations=np.zeros((points, 3), np.int32),
        observation_xy=np.zeros((points, 2), np.float32),
    )


def _workspace(tmp_path):
    return global_solve.workspace_for(_Store(tmp_path), "world", "session")


# ---------------------------------------------------------------------------
# 1. The write publishes atomically.


def test_the_arrays_are_never_published_half_written(tmp_path):
    """A concurrent reader sees a whole archive or no file, never a part.

    The check that matters is NOT "the final file is valid" -- it was valid
    on the field artifact too, because the writer finished 169 ms after the
    reader had already failed. It is that no intermediate state is ever
    reachable at the published path while the writer is running.
    """
    workspace = _workspace(tmp_path)
    solution = _solution(40_000)  # big enough that the write is not instant

    torn: list[str] = []
    stop = threading.Event()

    def poll() -> None:
        while not stop.is_set():
            try:
                data = workspace.arrays_path.read_bytes()
            except (FileNotFoundError, PermissionError):
                continue
            if not data:
                continue
            try:
                with np.load(io.BytesIO(data)) as arrays:
                    arrays["xyz"]
            except Exception as exc:  # noqa: BLE001 -- catching it IS the assertion
                torn.append(f"{type(exc).__name__}: {exc}")
                return

    reader = threading.Thread(target=poll, daemon=True)
    reader.start()
    try:
        for _ in range(12):
            write_solution(workspace, solution)
    finally:
        stop.set()
        reader.join(timeout=10)

    assert not torn, f"a reader observed an unpublished state: {torn[:3]}"


def test_the_write_leaves_no_temp_file_behind(tmp_path):
    workspace = _workspace(tmp_path)
    write_solution(workspace, _solution())
    assert [p.name for p in workspace.root.iterdir() if p.name.endswith(".tmp")] == []


def test_a_killed_write_does_not_publish_a_torn_file(tmp_path):
    """The worse half of the field bug: a TERMINATED writer.

    `BackgroundSolver.wait` terminates a solve child that outstays a stop,
    and TerminateProcess gives it no chance to finish the zip. Writing in
    place left that torn archive at the published path PERMANENTLY, so every
    later read of the world failed identically -- a poisoned world, not a
    transient race. With a temp file the previous solution survives.
    """
    workspace = _workspace(tmp_path)
    write_solution(workspace, _solution(32))
    good = workspace.arrays_path.read_bytes()

    def explode(handle):
        handle.write(b"PK\x03\x04 not a whole zip")
        raise RuntimeError("terminated mid-write")

    with pytest.raises(RuntimeError):
        write_bytes_atomic(workspace.arrays_path, explode)

    assert workspace.arrays_path.read_bytes() == good, (
        "the interrupted write replaced the last good solution"
    )
    assert not list(workspace.root.glob("*.tmp"))


def test_the_arrays_are_published_before_the_metadata(tmp_path):
    """Order is load-bearing: `load_solution` gates on both files existing.

    `solution.json` carries `schema_version` and `solved_at`, so it is the
    commit marker. Publishing it first would advertise a solution whose
    arrays were still being written -- which is the field bug with the
    files swapped.
    """
    workspace = _workspace(tmp_path)
    order: list[str] = []
    real_bytes, real_json = global_solve.write_bytes_atomic, global_solve.write_json_atomic
    global_solve.write_bytes_atomic = lambda p, w: (order.append("npz"), real_bytes(p, w))[1]
    global_solve.write_json_atomic = lambda p, d: (order.append("json"), real_json(p, d))[1]
    try:
        write_solution(workspace, _solution())
    finally:
        global_solve.write_bytes_atomic, global_solve.write_json_atomic = real_bytes, real_json
    assert order == ["npz", "json"]


# ---------------------------------------------------------------------------
# 2. The read never raises, on any of the ways the archive can be unreadable.


# How each shape of damage surfaces out of numpy, measured rather than
# assumed. Only the first two are the field bug -- a zip whose central
# directory has not landed yet. The other two were already survivable,
# because numpy mistakes them for a pickle and raises `ValueError`, which
# the old tuple did catch. They are here so that a future narrowing of the
# guard has to confront the fact that ONE corrupt-file story produces at
# least two unrelated exception hierarchies.
@pytest.mark.parametrize(
    "corrupt, expected, was_caught_before",
    [
        pytest.param(lambda w: w[: len(w) // 2], zipfile.BadZipFile, False, id="truncated"),
        pytest.param(lambda w: w[:64], zipfile.BadZipFile, False, id="header-only"),
        pytest.param(lambda w: b"", EOFError, False, id="empty"),
        pytest.param(lambda w: b"\x00" * len(w), ValueError, True, id="zeroed"),
    ],
)
def test_an_unreadable_npz_reads_as_absent_rather_than_raising(
    tmp_path, corrupt, expected, was_caught_before
):
    """The field failure, reproduced against the real `load_solution`.

    Truncation is exactly what the reader saw: a file that begins with a
    local header and has no central directory yet.
    """
    store = _Store(tmp_path)
    workspace = _workspace(tmp_path)
    write_solution(workspace, _solution())
    workspace.arrays_path.write_bytes(corrupt(workspace.arrays_path.read_bytes()))

    # The precondition: numpy really does raise this, and for the two field
    # shapes the raised type really is outside the tuple the guard used to
    # catch -- which is why those two ended a session and these two did not.
    with pytest.raises(expected) as raised:
        np.load(workspace.arrays_path)
    caught_by_old_guard = isinstance(
        raised.value, (OSError, KeyError, ValueError, json.JSONDecodeError)
    )
    assert caught_by_old_guard is was_caught_before

    assert load_solution(store, "world", "session") is None


def test_the_dominant_race_exception_is_also_absorbed(tmp_path):
    """`EOFError`, not `BadZipFile`, is what the race raises most often.

    Measured 48,854 to 10,295 over 15 s. It descends from Exception, so a
    fix scoped to `zipfile.BadZipFile` alone would still have left the
    commonest failure uncaught. numpy raises it from a member whose header
    is whole but whose payload is short.
    """
    store = _Store(tmp_path)
    workspace = _workspace(tmp_path)
    write_solution(workspace, _solution())

    real = global_solve.read_bytes_closed
    global_solve.read_bytes_closed = lambda p: (_ for _ in ()).throw(EOFError("No data left in file"))
    try:
        assert load_solution(store, "world", "session") is None
    finally:
        global_solve.read_bytes_closed = real


def test_a_whole_solution_still_round_trips(tmp_path):
    """The guard must not have been widened into swallowing real data."""
    store = _Store(tmp_path)
    workspace = _workspace(tmp_path)
    original = _solution(17)
    write_solution(workspace, original)
    loaded = load_solution(store, "world", "session")
    assert loaded is not None
    assert loaded.solver == "test"
    assert loaded.keyframe_ids == original.keyframe_ids
    np.testing.assert_array_equal(loaded.xyz, original.xyz)
    np.testing.assert_array_equal(loaded.observations, original.observations)


def test_a_stop_still_propagates_through_the_reader(tmp_path):
    """Broad is not blanket: BaseException must still stop the process."""
    store = _Store(tmp_path)
    workspace = _workspace(tmp_path)
    write_solution(workspace, _solution())

    real = global_solve.read_bytes_closed
    global_solve.read_bytes_closed = lambda p: (_ for _ in ()).throw(KeyboardInterrupt())
    try:
        with pytest.raises(KeyboardInterrupt):
            load_solution(store, "world", "session")
    finally:
        global_solve.read_bytes_closed = real


def test_the_reader_does_not_hold_the_file_open(tmp_path):
    """A reader that keeps the zip open is what blocks a writer's replace.

    On Windows `os.replace` onto a destination any handle has open fails
    with WinError 5, so the lazy `np.load(path)` form would make this reader
    the very obstacle the write path has to retry around. Publishing again
    straight after a read proves the handle is gone.
    """
    store = _Store(tmp_path)
    workspace = _workspace(tmp_path)
    write_solution(workspace, _solution())
    assert load_solution(store, "world", "session") is not None
    write_solution(workspace, _solution(21))  # must not raise PermissionError
    again = load_solution(store, "world", "session")
    assert again is not None and again.xyz.shape[0] == 21


# ---------------------------------------------------------------------------
# 3. The two together, as the builder actually runs them: two processes.


def test_a_reader_loop_never_raises_against_a_writer_loop(tmp_path):
    """The field race itself, against the real functions on both sides.

    Before the fix this loop raised within a few hundred milliseconds; the
    instrumented 15 s run recorded 59,283 escapes and zero clean Nones.
    """
    store = _Store(tmp_path)
    workspace = _workspace(tmp_path)
    solution = _solution(20_000)
    write_solution(workspace, solution)

    escaped: list[str] = []
    stop = threading.Event()

    def writer() -> None:
        while not stop.is_set():
            try:
                write_solution(workspace, solution)
            except Exception as exc:  # noqa: BLE001
                escaped.append(f"writer {type(exc).__name__}: {exc}")
                return

    thread = threading.Thread(target=writer, daemon=True)
    thread.start()
    loaded = 0
    try:
        for _ in range(400):
            try:
                if load_solution(store, "world", "session") is not None:
                    loaded += 1
            except Exception as exc:  # noqa: BLE001 -- an escape IS the failure
                escaped.append(f"reader {type(exc).__name__}: {exc}")
                break
    finally:
        stop.set()
        thread.join(timeout=10)

    assert not escaped, f"an exception escaped the race: {escaped[:3]}"
    assert loaded > 0, "the reader never once saw a published solution"


# ---------------------------------------------------------------------------
# 4. Two writers of one destination.
#
# The first version of the atomic write staged at `<name>.tmp` -- a name
# derived only from the destination, so every writer of that destination
# shared it. An adversarial review measured what that does: 656 BadZipFile
# reads at the published path over 12 seconds, and a writer killed by
# FileNotFoundError out of `replace` because its peer's `finally` had already
# unlinked the temp underneath it. The fix it was written to deliver, undone
# by the staging name.
#
# Nothing serialises writers of solution.npz: `acquire_writer_lock` is
# per-world and taken only by the engine, and `scripts/world_solve.py` takes
# no lock at all. A hand-run solve against a world with a live builder is two
# writers, and it is an ordinary operator action -- it is how the field
# artifact was recovered.


def test_two_writers_never_publish_a_torn_archive(tmp_path):
    workspace = _workspace(tmp_path)
    solution = _solution(30_000)

    torn: list[str] = []
    writer_errors: list[str] = []
    stop = threading.Event()

    def writer() -> None:
        while not stop.is_set():
            try:
                write_solution(workspace, solution)
            except Exception as exc:  # noqa: BLE001
                writer_errors.append(f"{type(exc).__name__}: {exc}")
                return

    def reader() -> None:
        while not stop.is_set():
            try:
                data = workspace.arrays_path.read_bytes()
            except (FileNotFoundError, PermissionError):
                continue
            if not data:
                continue
            try:
                with np.load(io.BytesIO(data)) as arrays:
                    arrays["xyz"]
            except Exception as exc:  # noqa: BLE001
                torn.append(f"{type(exc).__name__}: {exc}")
                return

    threads = [threading.Thread(target=writer, daemon=True) for _ in range(2)]
    threads.append(threading.Thread(target=reader, daemon=True))
    for thread in threads:
        thread.start()
    try:
        for _ in range(60):
            if torn or writer_errors:
                break
            stop.wait(0.05)
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=15)

    assert not torn, f"two writers published a torn archive: {torn[:3]}"
    assert not writer_errors, f"a writer was killed by its peer: {writer_errors[:3]}"


def test_two_writers_do_not_share_a_staging_name(tmp_path):
    """The mechanism, stated directly rather than raced for.

    A shared staging name is what let one writer's `finally` delete the
    other's in-flight temp, and what let one writer promote the half-written
    bytes the other was still producing.
    """
    from tower.storage import _temp_path

    target = tmp_path / "solution.npz"
    names = {_temp_path(target).name for _ in range(50)}
    assert len(names) == 50, "staging names collide"
    assert all(n.endswith(".tmp") for n in names), (
        "purge_world and the leftover-temp checks look for a .tmp suffix"
    )
    assert all(n.startswith("solution.npz.") for n in names), (
        "a stray temp must still say which artifact it was staging"
    )
