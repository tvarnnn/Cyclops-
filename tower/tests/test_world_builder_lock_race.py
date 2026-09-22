"""Two processes asking for one world's writer lock.

`acquire_writer_lock` read the holder and then wrote the lock, with nothing
atomic in between. An adversarial review released two children from a spin
barrier and measured **two simultaneous writers admitted in 8 of 8 trials** --
and drove two concurrent `scripts/world_finalize.py` runs to
`finalized: True` on the same world, both running `engine.build()` over one
derived tree. `world_finalize.py` calls this lock "the whole safety story".

Worse than admitting two: the second writer's record overwrote the first's,
so the file named only one of them, and when the first finished it unlinked
a lock the other still believed it held.

Real processes, not threads: the defect is a filesystem race and a
same-process test would be arbitrated by the GIL rather than by the
filesystem.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tower.world_builder.store import WorldLockedError, WorldStore

TRIALS = 8


CHILD = textwrap.dedent(
    """
    import json, sys, time
    sys.path.insert(0, sys.argv[4])
    from tower.world_builder.store import WorldStore, WorldLockedError

    store, world_id, barrier = WorldStore(sys.argv[1]), sys.argv[2], sys.argv[3]
    # Spin until the barrier appears, so both children arrive together.
    while not __import__("os").path.exists(barrier):
        pass
    mine = __import__("os").getpid()
    try:
        store.acquire_writer_lock(world_id)
    except WorldLockedError:
        print(json.dumps({"acquired": False, "pid": mine}))
    except Exception as exc:
        print(json.dumps({"acquired": False, "pid": mine,
                          "error": f"{type(exc).__name__}: {exc}"}))
    else:
        print(json.dumps({"acquired": True, "pid": mine}))
        time.sleep(0.4)          # hold it, so the peer cannot mistake us for gone
    """
)


@pytest.mark.parametrize("trial", range(TRIALS))
def test_only_one_of_two_processes_takes_the_lock(tmp_path, trial):
    root = tmp_path / "worlds"
    store = WorldStore(root)
    world_id = "w" * 32
    store.world_dir(world_id).mkdir(parents=True, exist_ok=True)

    script = tmp_path / "child.py"
    script.write_text(CHILD, encoding="utf-8")
    barrier = tmp_path / "go"
    repo = str(Path(__file__).resolve().parents[1])

    children = [
        subprocess.Popen(
            [sys.executable, str(script), str(root), world_id, str(barrier), repo],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        for _ in range(2)
    ]
    barrier.write_text("go", encoding="utf-8")
    results = []
    for child in children:
        out, err = child.communicate(timeout=60)
        assert child.returncode == 0, err[-2000:]
        results.append(json.loads(out.strip().splitlines()[-1]))

    acquired = [r for r in results if r.get("acquired")]
    assert len(acquired) == 1, (
        f"{len(acquired)} of 2 processes took the lock: {results}"
    )
    # And the file names the process that actually took it. `Popen.pid` is
    # not that process on this host -- the venv's `python.exe` is a launcher
    # stub -- so the child reports its own.
    holder = store.lock_holder(world_id)
    assert holder is not None
    assert holder["pid"] == acquired[0]["pid"], (
        f"the lock names {holder['pid']}, not the winner {acquired[0]['pid']}"
    )


def test_a_dead_holder_is_still_reclaimed(tmp_path):
    """The exclusive create must not have made a stale lock permanent.

    A builder killed by TerminateProcess leaves its LOCK behind; 29 such
    files exist on the development host. If those blocked their worlds
    forever the fix would be worse than the race.
    """
    from tower.world_builder import store as store_module

    root = tmp_path / "worlds"
    store = WorldStore(root)
    world_id = "w" * 32
    store.world_dir(world_id).mkdir(parents=True, exist_ok=True)

    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    original = store_module.os.getpid
    store_module.os.getpid = lambda: dead.pid
    try:
        store.acquire_writer_lock(world_id)
    finally:
        store_module.os.getpid = original
    assert store.lock_holder(world_id)["alive"] is False

    store.acquire_writer_lock(world_id)          # must not raise
    assert store.lock_holder(world_id)["pid"] == original()


def test_a_live_other_holder_is_refused(tmp_path):
    from tower.world_builder import store as store_module

    root = tmp_path / "worlds"
    store = WorldStore(root)
    world_id = "w" * 32
    store.world_dir(world_id).mkdir(parents=True, exist_ok=True)

    holder = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], stdin=subprocess.DEVNULL
    )
    original = store_module.os.getpid
    store_module.os.getpid = lambda: holder.pid
    try:
        store.acquire_writer_lock(world_id)
    finally:
        store_module.os.getpid = original
    try:
        with pytest.raises(WorldLockedError):
            store.acquire_writer_lock(world_id)
    finally:
        holder.terminate()
        holder.wait(timeout=10)


def test_a_process_can_still_retake_its_own_lock(tmp_path):
    """The builder retakes it after `stop_session(hold_lock=True)`."""
    root = tmp_path / "worlds"
    store = WorldStore(root)
    world_id = "w" * 32
    store.world_dir(world_id).mkdir(parents=True, exist_ok=True)
    store.acquire_writer_lock(world_id)
    store.acquire_writer_lock(world_id)          # must not raise


# ---------------------------------------------------------------------------
# What the create-exclusive rewrite itself made possible.


def test_a_zero_byte_lock_does_not_brick_a_world_forever(tmp_path):
    """A lock that names nobody must not refuse everybody, permanently.

    The old code wrote the lock through `write_json_atomic`, which is never
    partial. The rewrite creates the file with `O_CREAT|O_EXCL` and writes
    the record afterwards -- so a kill, a power loss or ENOSPC in that
    window leaves a ZERO-BYTE lock. `lock_holder` returns None for it, the
    loop treated None as "try again", and after eight attempts the world was
    refused. Forever: every retry hit the same file, and
    `scripts/world_finalize.py` -- the recovery tool -- was refused too.

    An adversarial review measured it raising in 1.3 ms and staying that way.
    """
    root = tmp_path / "worlds"
    store = WorldStore(root)
    world_id = "w" * 32
    store.world_dir(world_id).mkdir(parents=True, exist_ok=True)
    store.lock_path(world_id).write_bytes(b"")

    assert store.lock_holder(world_id) is None
    store.acquire_writer_lock(world_id)          # must not raise
    holder = store.lock_holder(world_id)
    assert holder is not None and holder["pid"] == os.getpid()


def test_an_unparseable_lock_does_not_brick_a_world_either(tmp_path):
    root = tmp_path / "worlds"
    store = WorldStore(root)
    world_id = "w" * 32
    store.world_dir(world_id).mkdir(parents=True, exist_ok=True)
    store.lock_path(world_id).write_text("{ this is not json", encoding="utf-8")

    store.acquire_writer_lock(world_id)          # must not raise
    assert store.lock_holder(world_id)["pid"] == os.getpid()


def test_a_lock_being_written_right_now_is_not_stolen(tmp_path):
    """The other half: an empty lock is only an orphan once it STAYS empty.

    A peer between its create and its write leaves exactly the same
    zero-byte file for microseconds. Reclaiming that would delete a live
    writer's lock -- which is the failure the grace period exists to avoid,
    and the reason the fix is a wait rather than an unconditional reclaim.
    """
    import threading

    root = tmp_path / "worlds"
    store = WorldStore(root)
    world_id = "w" * 32
    store.world_dir(world_id).mkdir(parents=True, exist_ok=True)
    path = store.lock_path(world_id)
    path.write_bytes(b"")

    # A "peer" that finishes its write well inside the grace period.
    def finish():
        import json
        import time as _time

        _time.sleep(0.01)
        path.write_text(
            json.dumps({"pid": 999999, "created_at": 0.0}), encoding="utf-8"
        )

    writer = threading.Thread(target=finish, daemon=True)
    writer.start()
    try:
        store.acquire_writer_lock(world_id)
    finally:
        writer.join(timeout=5)
    # 999999 is not a live pid, so reclaiming it is correct -- what matters
    # is that the record was READ rather than the file deleted while empty.
    assert store.lock_holder(world_id)["pid"] == os.getpid()


def test_the_loser_of_a_race_is_told_who_holds_it(tmp_path):
    """"Contending" loses the pid, and is indistinguishable from a brick.

    Without a pause between attempts an adversarial review measured every
    loser of a natural race exhausting all eight in 1.2 ms and reporting
    "could not take the writer lock in 8 attempts" -- 104 of 104 times,
    never once naming the live holder. `world_finalize.py` prints that
    string to an operator.
    """
    import subprocess
    import sys as _sys

    from tower.world_builder import store as store_module

    root = tmp_path / "worlds"
    store = WorldStore(root)
    world_id = "w" * 32
    store.world_dir(world_id).mkdir(parents=True, exist_ok=True)

    holder = subprocess.Popen(
        [_sys.executable, "-c", "import time; time.sleep(60)"], stdin=subprocess.DEVNULL
    )
    original = store_module.os.getpid
    store_module.os.getpid = lambda: holder.pid
    try:
        store.acquire_writer_lock(world_id)
    finally:
        store_module.os.getpid = original
    try:
        with pytest.raises(WorldLockedError) as raised:
            store.acquire_writer_lock(world_id)
        assert str(holder.pid) in str(raised.value), (
            f"the error does not name the holder: {raised.value}"
        )
    finally:
        holder.terminate()
        holder.wait(timeout=10)


def test_a_record_replaced_under_the_winner_is_not_returned_as_a_win(tmp_path):
    """The create is atomic; the reclaim is not.

    A peer that decides this lock is dead unlinks whatever is AT THE PATH --
    not the file it read -- so it can delete a lock created microseconds ago
    and create its own. An adversarial review drove that to both processes
    acquiring, 5 of 5, with a stall injected into the reclaim window (0 of
    120 naturally: narrow, not imaginary). The fix reads the record back
    before returning.

    The stall is injected here deterministically, at the read-back itself:
    the first `read_json_closed` on this path IS the winner's confirmation
    of its own record, so the moment before it runs is the window. What
    happens in that window is exactly what the review drove -- the lock is
    unlinked and replaced with a live stranger's record. (The `os.fsync`
    that ends the write looks like a tidier seam and is not one: it runs
    while the handle is still open, and Windows will not unlink an open
    file.)

    The right outcome is not "acquired" and not a crash: this process lost,
    and being told so by `WorldLockedError` is how a loser learns it. What
    must never happen is returning to write underneath the winner.
    """
    import psutil

    from tower.world_builder import store as store_module

    root = tmp_path / "worlds"
    store = WorldStore(root)
    world_id = "w" * 32
    store.world_dir(world_id).mkdir(parents=True, exist_ok=True)

    peer = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        assert psutil.pid_exists(peer.pid), "the stand-in peer died immediately"
        lock = store.lock_path(world_id)
        real_read = store_module.read_json_closed
        fired = []

        def steal(path):
            if fired:
                return real_read(path)
            fired.append(True)
            # A peer reclaiming what it believed was a dead lock.
            lock.unlink(missing_ok=True)
            # The store's own producer, not a hand-written record: a
            # test that invents its subject's output cannot notice when
            # the subject stops producing it. This campaign learned that
            # from the staging sweeper.
            lock.write_text(
                json.dumps(store_module._lock_record(peer.pid)), encoding="utf-8"
            )

            return real_read(path)

        store_module.read_json_closed = steal
        try:
            with pytest.raises(WorldLockedError) as raised:
                store.acquire_writer_lock(world_id)
        finally:
            store_module.read_json_closed = real_read

        assert fired, "the injection never ran; this test proved nothing"
        assert str(peer.pid) in str(raised.value), (
            f"the loser was not told who holds it: {raised.value}"
        )
        holder = store.lock_holder(world_id)
        assert holder is not None and holder["pid"] == peer.pid, (
            f"the loser wrote underneath the winner: the lock names {holder}"
        )
    finally:
        peer.kill()
        peer.wait(timeout=30)
