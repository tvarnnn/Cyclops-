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
