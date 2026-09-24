"""Review V10, fix round P3.7-REF2: the put-back of a re-finish (`scripts/world_refinish.py`)
and the idle Tower's finisher (`scripts/world_finish_pending.py`).

Each test turns a reviewer's probe into a check of the FIXED behaviour, and fails on the code
of e5f7151 (`RUN/baseline/review/V10/rv10/probes/m5/test_rv10_restore_interrupted.py`,
`.../V10/ref/probes/test_rv10c_*.py`):

* MED-2 -- the put-back reads the disk, is idempotent, and is finished by the next run
  wherever it was killed: EVERY interruption point is tested (`_kill_at`);
* MED-3 -- only this session's `derived/<session>` and (while it names the session) the world
  manifest go back; other derived output that changed since the set-aside parks the session;
* Q3 -- the final solve's child is in the ledger while it runs, and nothing is put back under it;
* L-6, L-7, L-7b, L-8, L-9, L-10b, and the finisher's side of MED-1(b).

Every world is a fixture under `tmp_path`: no GPU, no Tower, no real store.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import stat
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import world_finish_pending as wfp  # noqa: E402
from scripts import world_refinish as wr  # noqa: E402
from scripts.world_build_session import StopRequest  # noqa: E402
from tests.test_world_builder_area_build import S1, W1, fake_depth  # noqa: E402,F401
from tests.test_world_builder_refinish import (  # noqa: E402
    _old_world,
    _Solve,
    _world_with_areas,
    stages,  # noqa: F401 -- a fixture
)
from tests.test_world_builder_refinish_v9 import (  # noqa: E402
    BLACK,
    RAW,
    _images,
    _jpeg,
    _mean,
    _near,
    _PreparingSolve,
    _raw_walk,
    _rows,
    _walk_database,
    _walk_prepared_from_raw,
    dead_process,  # noqa: F401 -- a fixture
)
from tower import storage  # noqa: E402
from tower.world_builder import components as C  # noqa: E402
from tower.world_builder import global_solve as GS  # noqa: E402
from tower.world_builder.engine import WorldBuilderEngine  # noqa: E402
from tower.world_builder.global_solve import load_solution  # noqa: E402
from tower.world_builder.store import _lock_record  # noqa: E402


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    monkeypatch.setattr(wfp, "prewarm_world_builder", lambda *a, **k: ())
    monkeypatch.setattr(wr, "MOVE_BACKOFF_S", 0.001)


class Killed(BaseException):
    """A hard kill: not an `Exception`, so no `except Exception` sees it, and nothing after
    it runs."""


def _aside(store, stamp="k"):
    return store.world_dir(W1) / wr.REFINISH_DIRNAME / stamp


def _ledger(store, stamp="k"):
    return json.loads((_aside(store, stamp) / wr.LEDGER_FILENAME).read_text(encoding="utf-8"))


def _write(store, ledger, stamp="k"):
    (_aside(store, stamp) / wr.LEDGER_FILENAME).write_text(json.dumps(ledger), encoding="utf-8")


def _tree(root: Path) -> dict:
    root = Path(root)
    if not root.is_dir():
        return {}
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*")
            if p.is_file()}


def _run(root) -> int:
    return wfp.main(["--root", str(root), "--format", "json"], stop_request=StopRequest())


def _no_room(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("a room was built")

    monkeypatch.setattr(wfp, "final_surface_stages", refuse)


def _step_one_then_killed(store, process, *, stamp="k", state=wr.LEDGER_SET_ASIDE, mark=True):
    """The re-finish's own step 1 (and its room marker), then a hard kill: the ledger names a
    process that is gone and stays at `state`."""
    engine = WorldBuilderEngine(store)
    store.acquire_writer_lock(W1)
    try:
        wr.set_aside(store, W1, S1, stamp)
        if mark:
            for stage in ("surface", "appearance"):
                engine.mark_stage(W1, S1, stage, state="stopped",
                                  detail=wr.REFINISH_IN_PROGRESS.format(stamp=stamp))
    finally:
        engine.release_world(W1)
    ledger = _ledger(store, stamp)
    ledger["state"] = state
    ledger["process"] = process
    _write(store, ledger, stamp)
    return ledger


def _derived(store) -> Path:
    return store.world_dir(W1) / "derived"


def _state(store) -> dict:
    """Everything a put-back must give back, byte for byte."""
    wd = store.world_dir(W1)
    return {"solve": _tree(wd / "solve" / S1), "areas": _tree(C.areas_dir(store, W1)),
            "session": store.session_path(W1, S1).read_bytes(), "derived": _tree(wd / "derived")}


# ---------------------------------------------------------------------------
# MED-2: the put-back reads the disk, and a killed one is finished by the next
# ---------------------------------------------------------------------------


def _scenario(root, process, which):
    """A world whose re-finish died, and what the put-back owes it back.

    - `set-aside`: after step 1 (the fresh copy placed, the room marked), with built areas,
      and the child had REBUILT this session's derived tree and the world manifest;
    - `solve-moved`: in step 1, after the solve moved and before the fresh copy was placed
      (`setting-aside`; RV10 R1);
    - `copy-placed`: in step 1, after the fresh copy was placed and before `set-aside` was
      recorded (RV10-C probe 5b)."""
    store, _kids = _world_with_areas(root)
    derived = _derived(store)
    (derived / S1).mkdir(parents=True, exist_ok=True)
    (derived / S1 / "poses.json").write_text('{"poses": "the walk"}')
    (derived / "manifest.json").write_text(json.dumps({"session_id": S1, "built": "the walk"}))
    (derived / "s0").mkdir(exist_ok=True)
    (derived / "s0" / "poses.json").write_text('{"poses": "an older session"}')
    before = _state(store)
    digest = load_solution(store, W1, S1).input_digest
    if which == "set-aside":
        _step_one_then_killed(store, process)
        fresh = _tree(store.world_dir(W1) / "solve" / S1)
        # the child rebuilt the derived tree, and died before it published
        (derived / S1 / "poses.json").write_text('{"poses": "rebuilt by the dead re-finish"}')
        (derived / "manifest.json").write_text(json.dumps({"session_id": S1, "built": "rebuilt"}))
    elif which == "solve-moved":
        _step_one_then_killed(store, process, state=wr.LEDGER_SETTING_ASIDE, mark=False)
        fresh = _tree(store.world_dir(W1) / "solve" / S1)
        os.replace(store.world_dir(W1) / "solve" / S1, _aside(store) / wr.FRESH_SOLVE_STAGING)
    else:
        _step_one_then_killed(store, process, state=wr.LEDGER_SETTING_ASIDE, mark=False)
        fresh = _tree(store.world_dir(W1) / "solve" / S1)
    return store, before, digest, fresh


def _kill_at(monkeypatch, k: int, mode: str):
    """Patch every step a put-back takes -- each rename, each copy, each JSON write (the
    ledger's, the session record's, the finisher's counters) -- and hard-kill the k-th:
    `before` it happens, or `after` it happened and before anything records it. Returns the
    call counter."""
    calls = {"n": 0}

    def wrap(real):
        def step(*a, **kw):
            calls["n"] += 1
            if calls["n"] == k and mode == "before":
                raise Killed()
            out = real(*a, **kw)
            if calls["n"] == k:
                raise Killed()
            return out
        return step

    monkeypatch.setattr(wr, "_replace_with_retry", wrap(wr._replace_with_retry))
    monkeypatch.setattr(wr.shutil, "copytree", wrap(wr.shutil.copytree))
    monkeypatch.setattr(wr.shutil, "copy2", wrap(wr.shutil.copy2))
    monkeypatch.setattr(storage, "write_json_atomic", wrap(storage.write_json_atomic))
    monkeypatch.setattr(wfp, "write_json_atomic", wrap(wfp.write_json_atomic))
    return calls


def _put_back_killed(store, monkeypatch, k, mode) -> bool:
    """The finisher's put-back (`recover_dead_refinish`, under the lock), hard-killed at step
    k. False when it has fewer than k steps (then it ran to the end)."""
    store.acquire_writer_lock(W1)
    try:
        with monkeypatch.context() as m:
            _kill_at(m, k, mode)
            wr.recover_dead_refinish(store, W1, "k")
    except Killed:
        return True
    finally:
        store.release_writer_lock(W1)
    return False


def _assert_whole(store, before, digest, fresh):
    """The world is exactly what it was before the re-finish; the dead re-finish's fresh
    solve directory is kept, never deleted, and the ORIGINAL was never filed as failed."""
    assert load_solution(store, W1, S1).input_digest == digest
    assert _state(store) == before
    led = _ledger(store)
    assert led["state"] == wr.LEDGER_RESTORED_AFTER_DEATH, led.get("restore_report")
    assert led["restore_report"]["errors"] == []
    aside = _aside(store)
    for failed in aside.glob("failed-solve*"):
        assert not (failed / S1 / "solution.json").exists(), "the original was filed as failed"
    kept = [_tree(p) for p in list(aside.glob(f"failed-solve*/{S1}"))
            + [aside / wr.FRESH_SOLVE_STAGING]]
    assert fresh in kept, "the dead re-finish's fresh solve directory is kept"
    assert wfp.assess(store, W1, S1).code == "nothing-interrupted"


@pytest.mark.parametrize("which", ["set-aside", "solve-moved", "copy-placed"])
def test_med2_a_put_back_killed_at_any_step_is_finished_by_the_next_run(
        tmp_path, dead_process, stages, fake_depth, monkeypatch, which):  # noqa: F811
    """RV10 R1/R2, RV10-C 1a/1b. At e5f7151 the put-back wrote its ledger only at the end: the
    next one moved the ORIGINAL solve it had already put back under `failed-solve/` (R1), or
    collided with its own `failed-solve/` and parked a world that was whole (R2)."""
    points = 0
    for k in range(1, 60):
        stopped = False
        for mode in ("before", "after"):
            root = tmp_path / f"{k}{mode[0]}"
            store, before, digest, fresh = _scenario(root, dead_process, which)
            if not _put_back_killed(store, monkeypatch, k, mode):
                stopped = True
                break
            points += 1
            with monkeypatch.context() as m:
                _no_room(m)
                assert _run(root) == 0, (k, mode)
            _assert_whole(store, before, digest, fresh)
        if stopped:
            break
    assert points >= 8, points      # every rename, copy and record write of the put-back


def test_med2_a_put_back_killed_twice_uses_its_own_failed_names(tmp_path, dead_process, stages,
                                                                fake_depth, monkeypatch):  # noqa: F811
    """Each attempt keeps what it moves aside under its own `failed-*` name: a second attempt
    killed too, then a third, and nothing collides or is lost."""
    for k in (3, 5, 7):
        root = tmp_path / f"twice{k}"
        store, before, digest, fresh = _scenario(root, dead_process, "set-aside")
        assert _put_back_killed(store, monkeypatch, k, "after")
        assert _put_back_killed(store, monkeypatch, 2, "after")
        with monkeypatch.context() as m:
            _no_room(m)
            assert _run(root) == 0
        _assert_whole(store, before, digest, fresh)
        led = _ledger(store)
        assert [a["attempt"] for a in led["restore_attempts"]] == [1, 2, 3]
        names = [a["failed_under"] for a in led["restore_attempts"]]
        assert names[1] == ["failed-solve-2", "failed-areas-2", "failed-derived-2"]


def test_med2_an_owners_put_back_by_the_ledgers_own_words_is_recognised(tmp_path, dead_process,
                                                                         monkeypatch):
    """RV10-C probe 2: the owner did what the ledger's `restore` says -- the rebuild's solve
    aside, the original back. At e5f7151 the finisher then moved the owner's restored original
    under `failed-solve/`, could not move the gone set-aside copy, and parked the world."""
    store, _kids = _old_world(tmp_path)
    digest = load_solution(store, W1, S1).input_digest
    _step_one_then_killed(store, dead_process)
    live = store.world_dir(W1) / "solve" / S1
    os.replace(live, _aside(store) / "owner-moved-the-rebuild-here")
    os.replace(_aside(store) / "solve" / S1, live)
    _no_room(monkeypatch)
    assert _run(tmp_path) == 0
    led = _ledger(store)
    assert led["state"] == wr.LEDGER_RESTORED_AFTER_DEATH, led["restore_report"]
    assert load_solution(store, W1, S1).input_digest == digest
    assert led["restore_report"]["found_on_disk"] == [str(live)]
    assert not (_aside(store) / "failed-solve").exists()
    assert (_aside(store) / "owner-moved-the-rebuild-here").is_dir()      # untouched


def test_med2_a_solve_is_never_moved_aside_unless_its_original_is_there(tmp_path, dead_process,
                                                                          monkeypatch):
    """The set-aside original is gone (not by this tool) and the rebuild's fresh copy is at
    `solve/<session>`. The put-back must not take the fresh copy for the original, nor move it
    aside to make room for nothing: it moves NOTHING and parks, and says why."""
    store, _kids = _old_world(tmp_path)
    _step_one_then_killed(store, dead_process)
    os.replace(_aside(store) / "solve" / S1, tmp_path / "someone-took-it")
    live = _tree(store.world_dir(W1) / "solve" / S1)
    _no_room(monkeypatch)
    assert _run(tmp_path) == 0
    led = _ledger(store)
    assert led["state"] == wr.LEDGER_RESTORE_INCOMPLETE
    assert any("is not it" in e for e in led["restore_report"]["errors"])
    assert _tree(store.world_dir(W1) / "solve" / S1) == live             # not moved
    assert not (_aside(store) / "failed-solve").exists()
    assert wfp.assess(store, W1, S1).code == wfp.REFINISH_PARKED


def test_med2_a_ledger_that_cannot_record_the_attempt_moves_nothing(tmp_path, dead_process,
                                                                    monkeypatch):
    """The attempt is recorded in the ledger BEFORE anything moves: a ledger that cannot be
    written stops the put-back with the world as it was."""
    store, _kids = _old_world(tmp_path)
    _step_one_then_killed(store, dead_process)
    before = _tree(store.world_dir(W1))

    def full_disk(aside, ledger):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(wr, "_write_ledger", full_disk)
    store.acquire_writer_lock(W1)
    try:
        with pytest.raises(OSError):
            wr.recover_dead_refinish(store, W1, "k")
    finally:
        store.release_writer_lock(W1)
    assert _tree(store.world_dir(W1)) == before


# ---------------------------------------------------------------------------
# MED-3: only this session's derived tree
# ---------------------------------------------------------------------------


def _second_walk(store, s2="s2"):
    """Another walk of the same world, AFTER the re-finish died: its session, and the derived
    output its build writes (its own tree, and the world manifest naming it)."""
    session = store.read_session(W1, S1)
    store.write_session(dataclasses.replace(session, session_id=s2, started_at=100.0,
                                            ended_at=200.0, stages={
                                                "surface": {"state": "ok", "attempted": True},
                                                "appearance": {"state": "ok", "attempted": True}}))
    world = store.read_world(W1)
    store.write_world(dataclasses.replace(world, session_ids=tuple(world.session_ids) + (s2,)))
    derived = _derived(store)
    (derived / s2).mkdir(parents=True, exist_ok=True)
    (derived / s2 / "poses.json").write_text('{"s2": "poses"}')
    (derived / "manifest.json").write_text(json.dumps({"session_id": s2, "new": True}))
    return _tree(derived / s2)


def test_med3_a_later_walk_of_another_session_keeps_its_derived_tree(tmp_path, dead_process,
                                                                     monkeypatch):
    """RV10-C probe 3. At e5f7151 the put-back swapped the WHOLE `derived/` for the s1-era
    snapshot: s2's derived output left the live world, and the world manifest named s1's
    past. Nothing of s1's derived output changed here, so there is nothing of it to put back:
    s2's stays, and s1's solve, areas and record come back."""
    store, _kids = _old_world(tmp_path)
    digest = load_solution(store, W1, S1).input_digest
    _step_one_then_killed(store, dead_process)
    s2_tree = _second_walk(store)
    _no_room(monkeypatch)
    assert _run(tmp_path) == 0
    led = _ledger(store)
    assert led["state"] == wr.LEDGER_RESTORED_AFTER_DEATH, led["restore_report"]
    assert load_solution(store, W1, S1).input_digest == digest
    assert _tree(_derived(store) / "s2") == s2_tree
    assert json.loads((_derived(store) / "manifest.json").read_text()) == {"session_id": "s2",
                                                                          "new": True}
    assert not (_aside(store) / "failed-derived").exists()


def test_med3_a_rebuilt_session_tree_under_a_later_walk_parks_and_moves_nothing(
        tmp_path, dead_process, monkeypatch):
    """The dead re-finish HAD rebuilt s1's derived tree, and s2 was built after it (maybe on
    it). Putting s1's back could undo what s2 was built on, so nothing at all is moved: the
    session is parked, with the closed set's sentence on the row."""
    store, _kids = _old_world(tmp_path)
    _step_one_then_killed(store, dead_process)
    derived = _derived(store)
    (derived / S1).mkdir(parents=True, exist_ok=True)
    (derived / S1 / "poses.json").write_text('{"poses": "rebuilt"}')
    s2_tree = _second_walk(store)
    before = _tree(store.world_dir(W1))
    _no_room(monkeypatch)
    assert _run(tmp_path) == 0
    led = _ledger(store)
    assert led["state"] == wr.LEDGER_RESTORE_INCOMPLETE
    assert "s2/poses.json" in led["restore_report"]["derived"]["changed_elsewhere"]
    after = _tree(store.world_dir(W1))
    changed = {k for k in set(before) | set(after) if before.get(k) != after.get(k)}
    session = store.session_path(W1, S1).relative_to(store.world_dir(W1)).as_posix()
    # only the ledger, the finisher's counter and the row's notice
    assert all(k.startswith("refinish/") or k in (session, wfp.ATTEMPTS_FILENAME)
               for k in changed), changed
    assert load_solution(store, W1, S1) is None           # the solve stays where it was set aside
    assert _tree(derived / "s2") == s2_tree
    fin = store.read_session(W1, S1).finalization
    assert fin["notice"] == wfp.REFINISH_PARKED_NOTICE
    assert wfp.assess(store, W1, S1).code == wfp.REFINISH_PARKED


def test_med3_only_this_sessions_tree_and_its_manifest_go_back(tmp_path, dead_process,
                                                               monkeypatch):
    """The dead re-finish rebuilt s1's tree and the world manifest (naming s1); an older
    session's tree s0 is untouched. s1's and the manifest go back from the snapshot, each
    kept under `failed-derived/`; s0's is not even rewritten."""
    store, _kids = _old_world(tmp_path)
    derived = _derived(store)
    (derived / S1).mkdir(parents=True)
    (derived / S1 / "poses.json").write_text('{"poses": "walk"}')
    (derived / "s0").mkdir()
    (derived / "s0" / "poses.json").write_text('{"s0": 1}')
    (derived / "manifest.json").write_text(json.dumps({"session_id": S1, "built": "walk"}))
    before = _tree(derived)
    s0_mtime = (derived / "s0" / "poses.json").stat().st_mtime_ns
    _step_one_then_killed(store, dead_process)
    (derived / S1 / "poses.json").write_text('{"poses": "rebuilt"}')
    (derived / S1 / "support.json").write_text('{"support": "rebuilt"}')
    (derived / "manifest.json").write_text(json.dumps({"session_id": S1, "built": "rebuilt"}))
    _no_room(monkeypatch)
    assert _run(tmp_path) == 0
    assert _ledger(store)["state"] == wr.LEDGER_RESTORED_AFTER_DEATH
    assert _tree(derived) == before
    assert (derived / "s0" / "poses.json").stat().st_mtime_ns == s0_mtime
    kept = _tree(_aside(store) / "failed-derived")
    assert kept == {f"{S1}/poses.json": b'{"poses": "rebuilt"}',
                    f"{S1}/support.json": b'{"support": "rebuilt"}',
                    "manifest.json": json.dumps({"session_id": S1, "built": "rebuilt"}).encode()}


def test_med3_a_manifest_naming_another_session_is_never_this_put_backs(tmp_path, dead_process,
                                                                        monkeypatch):
    store, _kids = _old_world(tmp_path)
    derived = _derived(store)
    (derived / "manifest.json").write_text(json.dumps({"session_id": "s0"}))
    manifest = (derived / "manifest.json").read_bytes()
    _step_one_then_killed(store, dead_process)
    (derived / S1).mkdir()
    (derived / S1 / "poses.json").write_text('{"poses": "rebuilt"}')
    _no_room(monkeypatch)
    assert _run(tmp_path) == 0
    assert _ledger(store)["state"] == wr.LEDGER_RESTORED_AFTER_DEATH
    assert not (derived / S1).exists()                      # the world had none for s1
    assert (derived / "manifest.json").read_bytes() == manifest
    assert _tree(_aside(store) / "failed-derived") == {f"{S1}/poses.json": b'{"poses": "rebuilt"}'}


def test_med3_a_dead_writers_staging_file_is_not_another_sessions_output(tmp_path, dead_process,
                                                                         monkeypatch):
    store, _kids = _old_world(tmp_path)
    _step_one_then_killed(store, dead_process)
    derived = _derived(store)
    (derived / S1).mkdir()
    (derived / S1 / "poses.json").write_text('{"poses": "rebuilt"}')
    (derived / "manifest.json.p99999.0badc0de.tmp").write_text("{")    # killed mid-write
    _no_room(monkeypatch)
    assert _run(tmp_path) == 0
    assert _ledger(store)["state"] == wr.LEDGER_RESTORED_AFTER_DEATH


# ---------------------------------------------------------------------------
# Q3: the final solve's child is in the ledger, and nothing is put back under it
# ---------------------------------------------------------------------------


def _fake_finalize(tmp_path, body: str) -> Path:
    script = tmp_path / "fake_world_finalize.py"
    script.write_text(textwrap.dedent(body), encoding="utf-8")
    return script


def test_q3_the_ledger_names_the_solve_child_while_it_runs(tmp_path, stages, monkeypatch):  # noqa: F811
    """The child reads the ledger and finds ITSELF there, with its start time. (Itself, or --
    with a venv's `python.exe`, a launcher that starts the interpreter as ITS child and lives
    exactly as long -- its parent: the process `Popen` started.)"""
    store, _kids = _old_world(tmp_path)
    ledger_path = _aside(store, "c") / wr.LEDGER_FILENAME
    monkeypatch.setattr(wr, "FINALIZE_SCRIPT", _fake_finalize(tmp_path, f"""
        import json, os, sys
        child = json.load(open({str(ledger_path)!r}, encoding="utf-8")).get("child") or {{}}
        print(json.dumps({{"saw_itself": child.get("pid") in (os.getpid(), os.getppid()),
                          "with_start_time": isinstance(child.get("created_at"), float)}}))
        sys.exit(1)   # publishes nothing: the result is put back
    """))
    report = wr.refinish(store, tmp_path, W1, S1, stamp="c")
    assert report["final_solve"]["saw_itself"] is True, report["final_solve"]
    assert report["final_solve"]["with_start_time"] is True
    led = _ledger(store, "c")
    assert led["child"]["exit_code"] == 1 and led["state"] == wr.LEDGER_RESTORED
    assert wr.refinish_liveness(led)["child"] is False


@pytest.fixture
def live_child():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        yield child
    finally:
        child.kill()
        child.wait(timeout=30)


def test_q3_nothing_is_put_back_while_the_solve_child_runs(tmp_path, dead_process, live_child,
                                                          monkeypatch):
    """V10 Q3: the re-finish's own process was killed, its `world_finalize` child runs on. At
    e5f7151 the finisher saw a dead re-finish and put the original solve back -- where the
    child could then publish over it."""
    store, _kids = _old_world(tmp_path)
    ledger = _step_one_then_killed(store, dead_process)
    ledger["child"] = dict(_lock_record(live_child.pid), script="scripts/world_finalize.py")
    _write(store, ledger)
    before = _tree(store.world_dir(W1))
    _no_room(monkeypatch)
    assert wr.refinish_process_alive(_ledger(store)) is True
    verdict = wfp.assess(store, W1, S1)
    assert verdict.code == wfp.REFINISH_IN_PROGRESS_CODE and "final solve" in verdict.reason
    assert wr.dead_before_publish(store, W1, S1) is None
    assert _run(tmp_path) == wfp.EXIT_WAITING
    store.acquire_writer_lock(W1)
    try:
        out = wr.recover_dead_refinish(store, W1, "k")
    finally:
        store.release_writer_lock(W1)
    assert out["recovered"] is False and "still running" in out["why"]
    assert _tree(store.world_dir(W1)) == before
    live_child.kill()
    live_child.wait(timeout=30)
    assert _run(tmp_path) == 0
    assert _ledger(store)["state"] == wr.LEDGER_RESTORED_AFTER_DEATH


def test_q3_a_child_that_cannot_be_recorded_is_stopped_before_the_put_back(tmp_path, stages,
                                                                          monkeypatch):  # noqa: F811
    store, _kids = _old_world(tmp_path)
    digest = load_solution(store, W1, S1).input_digest
    monkeypatch.setattr(wr, "FINALIZE_SCRIPT", _fake_finalize(tmp_path, """
        import time
        time.sleep(120)
    """))
    seen = {}
    real_record, real_write = wr._child_record, wr._write_ledger

    def record(process):
        seen["process"] = process
        return real_record(process)

    def write(aside, ledger):
        if "child" in ledger and "exit_code" not in ledger["child"] and not seen.get("failed"):
            seen["failed"] = True
            raise OSError(28, "No space left on device")
        return real_write(aside, ledger)

    monkeypatch.setattr(wr, "_child_record", record)
    monkeypatch.setattr(wr, "_write_ledger", write)
    started = time.monotonic()
    report = wr.refinish(store, tmp_path, W1, S1, stamp="c")
    assert time.monotonic() - started < 60, "the child was waited for, not stopped"
    assert seen["process"].poll() is not None, "the child is gone before the put-back"
    assert report["error"].startswith("running the final solve: OSError")
    assert load_solution(store, W1, S1).input_digest == digest
    assert _ledger(store, "c")["state"] == wr.LEDGER_RESTORED_AFTER_ERROR


def test_q3_an_interrupt_stops_the_child_before_the_put_back(tmp_path, stages, monkeypatch):  # noqa: F811
    store, _kids = _old_world(tmp_path)
    digest = load_solution(store, W1, S1).input_digest
    monkeypatch.setattr(wr, "FINALIZE_SCRIPT", _fake_finalize(tmp_path, """
        import time
        time.sleep(120)
    """))
    seen = {}
    real_popen = subprocess.Popen

    class Interrupted(real_popen):
        def communicate(self, *a, **k):
            seen["process"] = self
            raise KeyboardInterrupt

    monkeypatch.setattr(wr.subprocess, "Popen", Interrupted)
    with pytest.raises(KeyboardInterrupt):
        wr.refinish(store, tmp_path, W1, S1, stamp="c")
    assert seen["process"].poll() is not None
    assert load_solution(store, W1, S1).input_digest == digest
    assert _ledger(store, "c")["state"] == wr.LEDGER_RESTORED_AFTER_ERROR


# ---------------------------------------------------------------------------
# L-6, L-7, L-7b: the finisher's counters
# ---------------------------------------------------------------------------


def test_l6_a_publish_just_before_the_kill_restarts_the_counters(tmp_path, dead_process,
                                                                 stages):  # noqa: F811
    """RV10-C probe 5c. The old room had been retired at its bound; the re-finish's child
    published and the parent died before restarting the counters. At e5f7151 the new room was
    marked `failed` on the OLD build's attempts, and never built."""
    store, kids = _old_world(tmp_path)
    wfp._write_ledger(store, W1, {S1: {"attempts": wfp.DEFAULT_MAX_ATTEMPTS, "forgiven": 0,
                                       "detail": "old room gave up"}})
    _step_one_then_killed(store, dead_process)
    _Solve(store, kids, gate_writes=False)(["child"])
    WorldBuilderEngine(store, clock=lambda: time.time() + 1.0).mark_finalization(
        W1, S1, state="complete", final_solve="solved")
    assert _run(tmp_path) == 0
    led = _ledger(store)
    assert led["state"] == wr.LEDGER_PUBLISHED
    assert led["previous"]["finish_attempts"][S1]["attempts"] == wfp.DEFAULT_MAX_ATTEMPTS
    assert len(stages) == 1, "the new solve's room is built"
    assert store.read_session(W1, S1).stages["surface"]["state"] == "ok"


def test_l7b_a_published_refinish_restarts_the_put_back_counter_too(tmp_path):
    store, _kids = _old_world(tmp_path)
    key = wfp.refinish_ledger_key(S1)
    wfp._write_ledger(store, W1, {key: {"attempts": 2, "forgiven": 0, "detail": "put-back"}})
    prior = wr._restart_attempts(store, W1, S1, "z")
    assert prior == {key: {"attempts": 2, "forgiven": 0, "detail": "put-back"}}
    assert wfp.read_attempts(store, W1, key) == 0
    assert wr._restart_attempts(store, W1, S1, "z") == {}     # not restarted twice


@pytest.mark.skipif(os.name != "nt", reason="Windows read-only attribute")
def test_l7_an_uncountable_put_back_is_made_and_starves_nothing(tmp_path, dead_process, stages,
                                                                monkeypatch):  # noqa: F811
    """RV10-C probes 6 and 6b: a read-only `finish_attempts.json`. At e5f7151 every run refused
    the put-back, exited 1 and spent its one world on it: the chore backed off to an hour, and
    another world's owed room was never built."""
    from tower.world_builder.global_solve import workspace_for, write_solution

    monkeypatch.setattr(storage, "REPLACE_BUDGET_S", 0.05)
    store, _kids = _old_world(tmp_path)
    digest = load_solution(store, W1, S1).input_digest
    w2 = "w2"
    world = store.read_world(W1)
    store.write_world(dataclasses.replace(world, world_id=w2))
    s = store.read_session(W1, S1)
    store.write_session(dataclasses.replace(s, world_id=w2, stages={
        "surface": {"state": "stopped", "attempted": True, "detail": "killed"},
        "appearance": {"state": "stopped", "attempted": True, "detail": "killed"}}))
    write_solution(workspace_for(store, w2, S1), load_solution(store, W1, S1))
    _step_one_then_killed(store, dead_process)
    wfp._write_ledger(store, W1, {"x": {"attempts": 0}})
    attempts = wfp._attempts_path(store, W1)
    os.chmod(attempts, stat.S_IREAD)
    try:
        code = _run(tmp_path)
    finally:
        os.chmod(attempts, stat.S_IREAD | stat.S_IWRITE)
    assert code == 0
    assert load_solution(store, W1, S1).input_digest == digest           # put back
    assert _ledger(store)["state"] == wr.LEDGER_RESTORED_AFTER_DEATH
    assert len(stages) == 1                                              # w2's room, built
    assert store.read_session(w2, S1).stages["surface"]["state"] == "ok"


# ---------------------------------------------------------------------------
# L-8: a raw walk image its record proves is kept when the raw frame is gone
# ---------------------------------------------------------------------------


def test_l8_a_recorded_raw_walk_image_is_carried_back_when_its_frame_is_gone(tmp_path, stages,
                                                                            monkeypatch):  # noqa: F811
    """RV10-C `test_rv10c_rawgone.py`, after a re-finish: the provenance record names each raw
    frame. At e5f7151 the capture's move away withheld all three images (`differs-from-plan`)
    and cleared their features: the solve was then made from the redacted copies."""
    store, kids, root = _raw_walk(tmp_path)
    monkeypatch.setenv(wr.CAPTURE_ROOT_ENV, str(root))
    ws = _walk_prepared_from_raw(store, root)
    names = ["00000001.jpg", "00000002.jpg", "00000003.jpg"]
    wr.refinish(store, tmp_path, W1, S1, solve_runner=_Solve(store, kids, gate_writes=False),
                stamp="r1")
    _walk_database(ws.database_path, names)
    images = {p.name: p.read_bytes() for p in ws.images_dir.iterdir()}
    os.replace(root, tmp_path / "captures-moved-away")          # the raw frames are gone
    solve = _PreparingSolve(store, kids, gate_writes=False)     # the REAL prepare_images
    report = wr.refinish(store, tmp_path, W1, S1, solve_runner=solve, stamp="r2")
    frames = report["solver_frames"]
    walk = frames["walk_images"]
    assert walk["carried_back"] == 3 and walk["by_record_frame_gone"] == 3, walk
    assert frames["raw_walk_images_frame_gone"] == 3 and frames["source"] == wr.SOURCE_RAW
    # ... and the solve keeps them (`global_solve.recorded_raw_frame_gone`): raw pixels
    assert solve.written == 0
    assert all(_near(_mean(_images(store) / f"{s:08d}.jpg"), RAW[s]) for s in (1, 2, 3))
    assert {p.name: p.read_bytes() for p in ws.images_dir.iterdir()} == images
    record = json.loads((ws.root / wr.SOLVER_IMAGES_PROVENANCE).read_text())["images"]
    assert {e["source"] for e in record.values()} == {"raw"}
    assert {e["verified"] for e in record.values()} == {"record-frame-gone"}
    assert _rows(ws.database_path)["keypoints"] == [1, 2, 3]       # nothing cleared


def test_l8_without_a_record_a_raw_walk_image_is_still_not_proven(tmp_path, stages,
                                                                  monkeypatch):  # noqa: F811
    """The walk-only shape: `sources.json` names the raw frame, but a walk's own solves write
    no provenance record, and `sources.json` does not prove what an image was made from."""
    store, kids, root = _raw_walk(tmp_path)
    monkeypatch.setenv(wr.CAPTURE_ROOT_ENV, str(root))
    _walk_prepared_from_raw(store, root)
    os.replace(root, tmp_path / "captures-moved-away")
    report = wr.refinish(store, tmp_path, W1, S1,
                         solve_runner=_Solve(store, kids, gate_writes=False), stamp="w")
    walk = report["solver_frames"]["walk_images"]
    assert walk["carried_back"] == 0 and walk["by_record_frame_gone"] == 0


def test_l8_a_record_of_another_frame_or_keyframe_is_not_the_gone_frame(tmp_path, stages,
                                                                        monkeypatch):  # noqa: F811
    store, kids, root = _raw_walk(tmp_path)
    monkeypatch.setenv(wr.CAPTURE_ROOT_ENV, str(root))
    ws = _walk_prepared_from_raw(store, root)
    wr.refinish(store, tmp_path, W1, S1, solve_runner=_Solve(store, kids, gate_writes=False),
                stamp="r1")
    record_path = ws.root / wr.SOLVER_IMAGES_PROVENANCE
    record = json.loads(record_path.read_text())
    record["images"]["00000001.jpg"]["frame"] = record["images"]["00000002.jpg"]["frame"]
    record["images"]["00000003.jpg"]["keyframe_id"] = f"{S1}:00000002"
    record_path.write_text(json.dumps(record))
    os.replace(root, tmp_path / "captures-moved-away")
    report = wr.refinish(store, tmp_path, W1, S1,
                         solve_runner=_Solve(store, kids, gate_writes=False), stamp="r2")
    walk = report["solver_frames"]["walk_images"]
    assert walk["by_record_frame_gone"] == 1
    assert sorted(walk["withheld_images"]) == ["00000001.jpg", "00000003.jpg"]


# ---------------------------------------------------------------------------
# L-9: an encoder drift is said, loudly
# ---------------------------------------------------------------------------


def test_l9_an_encoder_drift_is_warned_and_flagged(tmp_path, stages, monkeypatch, caplog):  # noqa: F811
    """RV10-C probe 7: JPEG q94 standing in for an OpenCV/libjpeg upgrade. At e5f7151 every
    walk image was withheld and the frozen matching lost, with no warning at all."""
    store, kids, root = _raw_walk(tmp_path)
    monkeypatch.setenv(wr.CAPTURE_ROOT_ENV, str(root))
    _walk_prepared_from_raw(store, root)
    monkeypatch.setattr(wr, "SOLVER_JPEG_QUALITY", 94)
    with caplog.at_level(logging.WARNING, logger=wr.__name__):
        report = wr.refinish(store, tmp_path, W1, S1,
                             solve_runner=_Solve(store, kids, gate_writes=False), stamp="d")
    walk = report["solver_frames"]["walk_images"]
    assert walk["withheld"] == {"differs-from-plan": 3}                  # still fail-safe
    assert walk["reproduction"]["differs_unexplained"] == 3
    assert walk["reproduction"]["encoder_drift_suspected"] is True
    warned = [r.getMessage() for r in caplog.records if r.name == wr.__name__]
    assert any("encoder" in w and "3 of the walk's solver images" in w for w in warned), warned
    assert _ledger(store, "d")["solver_frames"]["walk_images"]["reproduction"][
        "encoder_drift_suspected"] is True


def test_l9_an_image_made_from_its_stored_copy_is_explained_not_drift(tmp_path, stages,
                                                                      monkeypatch, caplog):  # noqa: F811
    """RV9-D D1's shape: the walk's images were made from the redacted copies, and this
    re-finish plans raw frames. All three differ, for a reason: no drift, no warning."""
    store, kids, root = _raw_walk(tmp_path)
    monkeypatch.delenv(wr.CAPTURE_ROOT_ENV, raising=False)
    wr.refinish(store, tmp_path, W1, S1, solve_runner=_PreparingSolve(store, kids,
                                                                     gate_writes=False),
                stamp="r1")
    assert all(_near(_mean(p), BLACK) for p in _images(store).iterdir())
    monkeypatch.setenv(wr.CAPTURE_ROOT_ENV, str(root))
    with caplog.at_level(logging.WARNING, logger=wr.__name__):
        report = wr.refinish(store, tmp_path, W1, S1,
                             solve_runner=_PreparingSolve(store, kids, gate_writes=False),
                             stamp="r2")
    reproduction = report["solver_frames"]["walk_images"]["reproduction"]
    assert reproduction["differs"] == 3 and reproduction["differs_explained"] == 3
    assert reproduction["encoder_drift_suspected"] is False
    assert not [r for r in caplog.records if r.name == wr.__name__
                and r.levelno >= logging.WARNING]
    assert all(_near(_mean(_images(store) / f"{s:08d}.jpg"), RAW[s]) for s in (1, 2, 3))


def test_l9_one_unexplained_image_is_warned_without_calling_it_drift(tmp_path, stages,
                                                                     monkeypatch, caplog):  # noqa: F811
    """RV9-D D2's shape: one walk image is another capture's frame (by name)."""
    store, kids, root = _raw_walk(tmp_path)
    monkeypatch.setenv(wr.CAPTURE_ROOT_ENV, str(root))
    ws = _walk_prepared_from_raw(store, root)
    _jpeg(ws.images_dir / "00000002.jpg", (250, 250, 0))
    with caplog.at_level(logging.WARNING, logger=wr.__name__):
        report = wr.refinish(store, tmp_path, W1, S1,
                             solve_runner=_Solve(store, kids, gate_writes=False), stamp="u")
    walk = report["solver_frames"]["walk_images"]
    assert walk["unexplained_images"] == ["00000002.jpg"]
    assert walk["reproduction"]["encoder_drift_suspected"] is False
    assert any("00000002.jpg" in r.getMessage() for r in caplog.records if r.name == wr.__name__)


# ---------------------------------------------------------------------------
# The copy-back: an image that changed while it was copied loses its features too
# ---------------------------------------------------------------------------


def test_an_image_changed_while_copying_has_its_features_cleared(tmp_path, stages, monkeypatch):  # noqa: F811
    """`set_aside` read the changed images from the DATABASE's ledger entry, which never has
    them: such an image was withheld, rewritten by the solve, and mapped with its old
    features."""
    store, kids, root = _raw_walk(tmp_path)
    monkeypatch.setenv(wr.CAPTURE_ROOT_ENV, str(root))
    ws = _walk_prepared_from_raw(store, root)
    _walk_database(ws.database_path, ["00000001.jpg", "00000002.jpg", "00000003.jpg"])
    real = wr._carry_back_images

    def changed_meanwhile(src, dst, carried):
        _jpeg(src / "00000002.jpg", (1, 2, 3))              # a writer, between judge and copy
        return real(src, dst, carried)

    monkeypatch.setattr(wr, "_carry_back_images", changed_meanwhile)
    wr.refinish(store, tmp_path, W1, S1, solve_runner=_Solve(store, kids, gate_writes=False),
                stamp="c")
    entry = next(c for c in _ledger(store, "c")["copied_back"] if c["name"] == "database.db")
    assert entry["features_cleared"] == {"images": 1, "pairs": 2}
    assert _rows(ws.database_path)["keypoints"] == [1, 3]


def test_the_walks_owed_clears_are_made_in_the_copied_database(tmp_path, stages, monkeypatch):  # noqa: F811
    """SOL's owed clears (review V10, L-11): a clear the walk's record still owes is made in
    REF's copy of the walk database, even for an image carried back."""
    store, kids, root = _raw_walk(tmp_path)
    monkeypatch.setenv(wr.CAPTURE_ROOT_ENV, str(root))
    ws = _walk_prepared_from_raw(store, root)
    _walk_database(ws.database_path, ["00000001.jpg", "00000002.jpg", "00000003.jpg"])
    (ws.root / wr.SOLVER_IMAGES_PROVENANCE).write_text(json.dumps({
        "record": wr.PROVENANCE_RECORD, "images": {},
        GS.FEATURES_CLEAR_OWED_KEY: ["00000003.jpg"]}))
    report = wr.refinish(store, tmp_path, W1, S1,
                         solve_runner=_Solve(store, kids, gate_writes=False), stamp="o")
    assert report["solver_frames"]["walk_images"]["carried_back"] == 3
    assert _rows(ws.database_path)["keypoints"] == [1, 2]
    record = json.loads((ws.root / wr.SOLVER_IMAGES_PROVENANCE).read_text())
    assert GS.FEATURES_CLEAR_OWED_KEY not in record


# ---------------------------------------------------------------------------
# L-10b: a refused or given-up re-gate keeps its reason in `detail`, scrubbed
# ---------------------------------------------------------------------------

RAW_TEXT = ("RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB at "
            r"C:\Users\tvllo\Projects\Glasses\tower\.venv\Lib\moge\model.py:411"
            "\n\tsecond line")


def test_l10b_a_given_up_regate_keeps_the_gates_reason_scrubbed(tmp_path):
    from tests.test_world_builder_finish_pending_gate import ROOM_OK, _at_the_bound, _gated
    from tower.world_builder import coherence_publish as CP

    gate = {"state": CP.GATE_STATE_APPLIED, "masks_applied": True, "metric_available": False,
            "retryable": True, "cause": CP.CAUSE_DEPTH_UNAVAILABLE,
            "depth": {"state": "unavailable", "detail": RAW_TEXT}}
    store = _gated(tmp_path, stages=ROOM_OK, gate=gate)
    _at_the_bound(store)
    assert _run(tmp_path) == 0
    fin = store.read_session("w1", "s1").finalization
    assert fin["detail"].startswith(fin["notice"] + " (the gate's record: RuntimeError: CUDA "
                                                    "out of memory")
    for raw in ("tvllo", "\\", "\n", "second line"):
        assert raw not in fin["detail"] and raw not in fin["notice"], raw
    assert "RuntimeError" not in fin["notice"]


@pytest.mark.parametrize("refusal", [{"database": False}, {"loadable": False}])
def test_l10b_a_refused_regate_keeps_the_refusal_in_detail(tmp_path, refusal):
    from tests.test_world_builder_finish_pending_gate import ROOM_OK, _gated

    store = _gated(tmp_path, stages=ROOM_OK, **refusal)
    why = wfp._regate_refusal(store, "w1", "s1")
    assert _run(tmp_path) == 0
    fin = store.read_session("w1", "s1").finalization
    assert "cannot re-run the gate" in fin["notice"]
    assert fin["detail"] == f"{fin['notice']} (refused: {why})"


# ---------------------------------------------------------------------------
# MED-1(b), the finisher's side: a re-gate a stop kept from writing writes nothing here either
# ---------------------------------------------------------------------------


def test_med1b_a_regate_that_wrote_nothing_leaves_the_row_and_the_room_as_they_were(tmp_path):
    from tests.test_world_builder_finish_pending_gate import ROOM_OK, _gated

    store = _gated(tmp_path, stages=ROOM_OK)
    session_before = store.read_session("w1", "s1")
    stop = StopRequest()

    def regate(store_, world_id, session_id, should_stop=None):
        stop.request(StopRequest.SOFT, "capture opened")
        return {"stopped": True, "publish": {"written": False, "why": "a stop"},
                "notice": "must not be written", "detail": "must not be written"}

    verdict = wfp.assess(store, "w1", "s1")
    assert verdict.stage == wfp.REGATE_STAGE
    out = wfp.finish_regate(store, verdict, appearance=True, prune_depth_work=False,
                            stop_request=stop, regate=regate)
    assert out["finished"] is False and "nothing was written" in out["reason"]
    after = store.read_session("w1", "s1")
    assert after.finalization == session_before.finalization
    assert after.stages == session_before.stages                     # the room's mark went back
    assert not wfp.read_attempts(store, "w1", wfp.regate_ledger_key("s1"))   # the stop's
    assert store.lock_holder("w1") is None
