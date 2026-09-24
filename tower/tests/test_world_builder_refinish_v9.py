"""Review V9, fix round P3.6-REF: the re-finish (`scripts/world_refinish.py`) and the idle
Tower's finisher (`scripts/world_finish_pending.py`).

Each test turns one of the reviewers' probes into a check of the FIXED behaviour, and fails
on the code of 6d4b567 (`RUN/baseline/review/V9/refinish/probes/test_rv9d_probes.py`,
`.../notice/test_rv9e_notice_probe.py`):

* M-5 -- a re-finish killed before it publishes is put back, not built over (RV9-D C, RV9-E F4);
* M-6 -- step 1 has one rollback from its first move to its last ledger write (RV9-D A);
* M-7 -- a carried-back solver image never outranks capture identity (RV9-D D1, D2, D3);
* M-12 -- the depth-prediction cache is left out of the set-aside copy;
* M-4 -- the given-up notice carries fixed phrases, never raw error text;
* LOWs -- a refused re-gate's notice and exit, the notice after a failed write, the finisher's
  counters on a put-back, `int(wire_seq)`, a relative `--capture-dir`, the ledger after steps 3
  and 4, liveness without a start time.

Every world is a fixture under `tmp_path`: no GPU, no Tower, no real store.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import sqlite3
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import world_build_session as wbs  # noqa: E402
from scripts import world_finish_pending as wfp  # noqa: E402
from scripts import world_refinish as wr  # noqa: E402
from scripts.world_build_session import StopRequest  # noqa: E402
from tests.test_world_builder_area_build import S1, W1, fake_depth  # noqa: E402,F401
from tests.test_world_builder_refinish import (  # noqa: E402
    CAP_B,
    CAPTURE_ID,
    _old_world,
    _Solve,
    _world_with_areas,
    stages,  # noqa: F401 -- a fixture
)
from tower.world_builder import components as C  # noqa: E402
from tower.world_builder import global_solve as GS  # noqa: E402
from tower.world_builder.global_solve import load_solution  # noqa: E402
from tower.world_builder.records import Keyframe  # noqa: E402


@pytest.fixture(autouse=True)
def _no_native_warm(monkeypatch):
    monkeypatch.setattr(wfp, "prewarm_world_builder", lambda *a, **k: ())


def _ledger(store, stamp):
    return json.loads((store.world_dir(W1) / wr.REFINISH_DIRNAME / stamp / wr.LEDGER_FILENAME)
                      .read_text(encoding="utf-8"))


def _write_ledger(store, stamp, ledger):
    (store.world_dir(W1) / wr.REFINISH_DIRNAME / stamp / wr.LEDGER_FILENAME).write_text(
        json.dumps(ledger), encoding="utf-8")


@pytest.fixture
def dead_process():
    """The pid and start time of a process that has exited."""
    from tower.world_builder.store import _lock_record

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        record = _lock_record(child.pid)
    finally:
        child.kill()
        child.wait(timeout=30)
    assert "created_at" in record
    return record


def _tree(root: Path) -> dict:
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def _run(root) -> int:
    return wfp.main(["--root", str(root), "--format", "json"], stop_request=StopRequest())


def _no_room_build(monkeypatch):
    def refuse(*a, **k):
        pytest.fail("a room was built")

    monkeypatch.setattr(wfp, "final_surface_stages", refuse)


def _step_one_then_killed(store, stamp, process, *, state=wr.LEDGER_SET_ASIDE, mark=True):
    """The re-finish's own step 1 (and its room marker), then a hard kill: the ledger names a
    process that is gone and stays at `state`."""
    from tower.world_builder.engine import WorldBuilderEngine

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
    _write_ledger(store, stamp, ledger)
    return ledger


# ---------------------------------------------------------------------------
# M-5: a re-finish killed before it publishes
# ---------------------------------------------------------------------------


def test_m5_a_refinish_killed_after_step_1_is_put_back_and_its_room_is_not_built(
        tmp_path, dead_process, monkeypatch):
    """RV9-D probe C: before the fix the finisher owed the stopped room and ran the REAL
    `final_surface_stages` on a session with no solution (`unavailable`), and the set-aside
    solve was never restored."""
    store, _kids = _old_world(tmp_path)
    before_session = store.read_session(W1, S1)
    digest = load_solution(store, W1, S1).input_digest
    _step_one_then_killed(store, "k", dead_process)
    assert load_solution(store, W1, S1) is None
    _no_room_build(monkeypatch)
    verdict = wfp.assess(store, W1, S1)
    assert verdict.owed and verdict.stage == "refinish", verdict      # wfp.REFINISH_STAGE
    out = wfp.finish(store, verdict, appearance=True, prune_depth_work=False,
                     stop_request=StopRequest())
    assert out["finished"] is True, out
    assert load_solution(store, W1, S1).input_digest == digest          # put back
    assert store.read_session(W1, S1).stages == before_session.stages  # the room as it was
    ledger = _ledger(store, "k")
    assert ledger["state"] == wr.LEDGER_RESTORED_AFTER_DEATH
    assert ledger["state"] in wr.LEDGER_TERMINAL_STATES
    assert (store.world_dir(W1) / "refinish" / "k" / "failed-solve" / S1).exists()  # kept
    assert store.lock_holder(W1) is None
    assert not wfp.assess(store, W1, S1).owed


def test_m5_a_kill_before_the_solve_moved_leaves_the_solve_where_it_is(
        tmp_path, stages, fake_depth, dead_process, monkeypatch):  # noqa: F811
    """Killed in step 1 after the areas moved and before the solve did (the ledger names
    the areas moved, `setting-aside`). The put-back moves the areas back and must NOT take
    the solve still in place for a failed rebuild."""
    store, _kids = _world_with_areas(tmp_path)
    digest = load_solution(store, W1, S1).input_digest
    areas = _tree(C.areas_dir(store, W1))
    real = wr._replace_with_retry

    class Killed(BaseException):
        pass

    def killed_at_the_solve(src, dst):
        if Path(src) == store.world_dir(W1) / "solve" / S1:
            raise Killed()
        return real(src, dst)

    monkeypatch.setattr(wr, "_replace_with_retry", killed_at_the_solve)
    monkeypatch.setattr(wr, "_move_back", lambda done, ledger, aside: True)   # a hard kill
    monkeypatch.setattr(wr, "_write_ledger_quietly", lambda aside, ledger: None,
                        raising=False)
    store.acquire_writer_lock(W1)
    try:
        with pytest.raises(Killed):
            wr.set_aside(store, W1, S1, "k")
    finally:
        store.release_writer_lock(W1)
    monkeypatch.undo()
    monkeypatch.setattr(wfp, "prewarm_world_builder", lambda *a, **k: ())
    ledger = _ledger(store, "k")
    assert ledger["state"] == wr.LEDGER_SETTING_ASIDE
    ledger["process"] = dead_process
    _write_ledger(store, "k", ledger)
    assert not any(C.areas_dir(store, W1).iterdir())       # the areas are away ...
    assert load_solution(store, W1, S1).input_digest == digest   # ... the solve is not
    _no_room_build(monkeypatch)
    assert _run(tmp_path) == 0
    assert load_solution(store, W1, S1).input_digest == digest
    assert _tree(C.areas_dir(store, W1)) == areas          # the built areas, byte for byte
    after = _ledger(store, "k")
    assert after["state"] == wr.LEDGER_RESTORED_AFTER_DEATH
    assert "solve_never_moved" in after["restore_report"]
    assert not (store.world_dir(W1) / "refinish" / "k" / "failed-solve").exists()


def test_m5_a_kill_between_a_rename_and_its_record_is_read_off_the_disk(
        tmp_path, dead_process, monkeypatch):
    """Killed right after the solve's rename, before the ledger recorded it: the ledger says
    `setting-aside` and the solve NOT moved, the disk says moved. The disk wins."""
    store, _kids = _old_world(tmp_path)
    digest = load_solution(store, W1, S1).input_digest
    ledger = _step_one_then_killed(store, "k", dead_process, state=wr.LEDGER_SETTING_ASIDE,
                                   mark=False)
    # undo the placement and the record of the solve's move, as the kill left them
    aside = store.world_dir(W1) / "refinish" / "k"
    os.replace(store.world_dir(W1) / "solve" / S1, aside / wr.FRESH_SOLVE_STAGING)
    for m in ledger["moved"]:
        m.pop("moved", None)
    _write_ledger(store, "k", ledger)
    _no_room_build(monkeypatch)
    assert _run(tmp_path) == 0
    assert load_solution(store, W1, S1).input_digest == digest
    assert _ledger(store, "k")["ended_before_publish"]["moves_found_on_disk"] == [
        str(store.world_dir(W1) / "solve" / S1)]


def test_m5_a_solve_that_published_before_the_kill_stands_and_its_room_is_owed(
        tmp_path, dead_process, stages):  # noqa: F811
    """The solve child published (a solution in the fresh directory, a solved finalization
    written after the set-aside) and THEN the re-finish died: the new solve stands, the ledger
    says `published`, and the room it marked is rebuilt from it."""
    import time

    from tower.world_builder.engine import WorldBuilderEngine

    store, kids = _old_world(tmp_path)
    _step_one_then_killed(store, "k", dead_process)
    _Solve(store, kids, gate_writes=False)(["child"])            # the child's publish
    WorldBuilderEngine(store, clock=lambda: time.time() + 1.0).mark_finalization(
        W1, S1, state="complete", final_solve="solved")
    assert _run(tmp_path) == 0
    assert _ledger(store, "k")["state"] == wr.LEDGER_PUBLISHED
    assert len(stages) == 1, "the marked room is rebuilt from the new solve"
    assert not (store.world_dir(W1) / "refinish" / "k" / "failed-solve").exists()


def test_m5_a_put_back_that_cannot_complete_parks_the_session_and_says_so(
        tmp_path, dead_process, monkeypatch):
    store, _kids = _old_world(tmp_path)
    _step_one_then_killed(store, "k", dead_process)
    real = wr._replace_with_retry

    def solve_will_not_go_back(src, dst):
        if Path(dst) == store.world_dir(W1) / "solve" / S1:
            raise PermissionError(5, "Access is denied")
        return real(src, dst)

    monkeypatch.setattr(wr, "_replace_with_retry", solve_will_not_go_back)
    _no_room_build(monkeypatch)
    assert _run(tmp_path) == 0
    assert _ledger(store, "k")["state"] == wr.LEDGER_RESTORE_INCOMPLETE
    v = wfp.assess(store, W1, S1)
    assert not v.owed and v.code == wfp.REFINISH_PARKED and not v.exhausted, v
    fin = store.read_session(W1, S1).finalization
    assert fin["notice"] == wfp.REFINISH_PARKED_NOTICE
    assert "\\" not in fin["notice"] and "/" not in fin["notice"]
    assert _run(tmp_path) == 0                                   # and it stays parked


def test_m5_a_rerun_of_the_refinish_puts_the_dead_one_back_first(tmp_path, stages,
                                                                 dead_process):  # noqa: F811
    """An owner who re-runs the re-finish after the first died gets the same put-back first,
    so the new one sets aside the PREVIOUS solve, not the dead one's half-built copy."""
    store, kids = _old_world(tmp_path)
    digest = load_solution(store, W1, S1).input_digest
    _step_one_then_killed(store, "k", dead_process)
    report = wr.refinish(store, tmp_path, W1, S1, solve_runner=_SolveFrom(store, kids, "z"),
                         stamp="z")
    assert report["earlier_refinish"]["state"] == wr.LEDGER_RESTORED_AFTER_DEATH
    assert report["done"] is True, report
    assert load_solution(_ShadowAside(store, "z"), W1, S1).input_digest == digest


class _ShadowAside:
    def __init__(self, store, stamp):
        self._dir = store.world_dir(W1) / wr.REFINISH_DIRNAME / stamp

    def world_dir(self, world_id):
        return self._dir


class _SolveFrom(_Solve):
    """The solve child, publishing the solve set aside under `stamp`."""

    def __init__(self, store, kids, stamp):
        super().__init__(store, kids)
        self.stamp = stamp

    def __call__(self, argv, env=None, capture_output=True, text=True):
        from tests.test_world_builder_area_build import _components
        from tower.world_builder.global_solve import workspace_for, write_solution

        write_solution(workspace_for(self.store, W1, S1),
                       load_solution(_ShadowAside(self.store, self.stamp), W1, S1))
        _components(self.store, self.kids)
        return SimpleNamespace(returncode=0, stdout="{}", stderr="")


# ---------------------------------------------------------------------------
# M-6: step 1's one rollback
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.name != "nt", reason="Windows refuses to replace a read-only file")
def test_m6_a_read_only_attempt_ledger_costs_the_restart_not_the_refinish(
        tmp_path, stages, fake_depth, monkeypatch):  # noqa: F811
    """RV9-D probe A: a read-only `finish_attempts.json` (a copied tree keeps Windows'
    read-only attribute) raised `PermissionError` after every move of step 1, outside every
    rollback: the solve and areas stranded, the ledger at `setting-aside`, nothing reported."""
    from tower import storage

    monkeypatch.setattr(storage, "REPLACE_BUDGET_S", 0.2)
    store, kids = _world_with_areas(tmp_path)
    wfp._write_ledger(store, W1, {S1: {"attempts": 2, "detail": "finishing the surface stage"}})
    attempts = wfp._attempts_path(store, W1)
    before = attempts.read_bytes()
    os.chmod(attempts, stat.S_IREAD)
    try:
        report = wr.refinish(store, tmp_path, W1, S1, solve_runner=_SolveFrom(store, kids, "ro"),
                             stamp="ro")
    finally:
        os.chmod(attempts, stat.S_IREAD | stat.S_IWRITE)
    assert report["done"] is True, report
    assert "PermissionError" in report["finish_attempts_error"]
    assert attempts.read_bytes() == before
    assert _ledger(store, "ro")["state"] == wr.LEDGER_DONE
    assert store.lock_holder(W1) is None


def test_m6_a_failure_after_the_moves_rolls_everything_back(tmp_path, stages, fake_depth,
                                                            monkeypatch):  # noqa: F811
    """The last ledger write of step 1 fails (a full disk): every move is undone -- the fresh
    copy taken back out of `solve/<session>` first -- and the refusal is `SetAsideFailed`,
    which `main` reports. Before: an `OSError` out of `set_aside`, the world stranded."""
    store, kids = _world_with_areas(tmp_path)
    digest = load_solution(store, W1, S1).input_digest
    areas = sorted(p.name for p in C.areas_dir(store, W1).iterdir())
    real = wr._write_ledger

    def full_disk_at_the_end(aside, ledger):
        if ledger.get("state") == wr.LEDGER_SET_ASIDE:
            raise OSError(28, "No space left on device")
        return real(aside, ledger)

    monkeypatch.setattr(wr, "_write_ledger", full_disk_at_the_end)
    with pytest.raises(wr.SetAsideFailed) as refused:
        wr.refinish(store, tmp_path, W1, S1, solve_runner=_SolveFrom(store, kids, "b"),
                    stamp="b")
    assert isinstance(refused.value, wr.Refused)          # `main` reports it: exit 2
    assert load_solution(store, W1, S1).input_digest == digest
    assert sorted(p.name for p in C.areas_dir(store, W1).iterdir()) == areas
    ledger = _ledger(store, "b")
    assert ledger["state"] == wr.LEDGER_ROLLED_BACK and "recording the set-aside" in ledger["error"]
    assert all(m.get("moved_back") for m in ledger["moved"])
    assert (store.world_dir(W1) / "refinish" / "b" / wr.FRESH_SOLVE_STAGING).is_dir()  # kept
    assert store.read_session(W1, S1).stages["surface"]["state"] == "ok"   # never marked
    assert store.lock_holder(W1) is None


def test_m6_any_exception_in_the_moves_is_rolled_back(tmp_path, stages, fake_depth,
                                                      monkeypatch):  # noqa: F811
    """Not only an `OSError`: before, a `RuntimeError` from a move escaped the rollback and
    left the areas moved."""
    store, kids = _world_with_areas(tmp_path)
    areas = sorted(p.name for p in C.areas_dir(store, W1).iterdir())
    real = wr._replace_with_retry

    def broken_at_the_solve(src, dst):
        if Path(src) == store.world_dir(W1) / "solve" / S1:
            raise RuntimeError("a driver bug")
        return real(src, dst)

    monkeypatch.setattr(wr, "_replace_with_retry", broken_at_the_solve)
    with pytest.raises(wr.SetAsideFailed, match="RuntimeError"):
        wr.refinish(store, tmp_path, W1, S1, solve_runner=_SolveFrom(store, kids, "b"),
                    stamp="b")
    assert sorted(p.name for p in C.areas_dir(store, W1).iterdir()) == areas
    assert _ledger(store, "b")["state"] == wr.LEDGER_ROLLED_BACK
    assert store.lock_holder(W1) is None


# ---------------------------------------------------------------------------
# M-7: carried-back solver images never outrank capture identity
# ---------------------------------------------------------------------------


def _jpeg(path: Path, colour, quality=100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((180, 240, 3), np.uint8)
    img[:] = colour
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    path.write_bytes(buf.tobytes())


def _mean(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return tuple(int(round(v)) for v in img.reshape(-1, 3).mean(0))


def _near(a, b, tol=10):
    return all(abs(int(x) - int(y)) <= tol for x, y in zip(a, b))


def _capture(root, capture_id, frames, continues=None):
    """frames = [(seq, t, colour)] -- real JPEGs and the recorder's journal."""
    d = root / "captures" / capture_id
    (d / "frames").mkdir(parents=True, exist_ok=True)
    (d / "capture.json").write_text(json.dumps({
        "capture_id": capture_id, "continues_capture": continues,
        "started_at": min(t for _s, t, _c in frames) - 1.0, "retains_raw_imagery": True}))
    lines = []
    for seq, t, colour in frames:
        _jpeg(d / "frames" / f"{seq:08d}.jpg", colour)
        lines.append(json.dumps({"schema_version": 1, "source_seq": seq, "wire_seq": seq,
                                 "received_at": t, "time_basis": "tower-receipt",
                                 "relpath": f"frames/{seq:08d}.jpg", "byte_count": 1}))
    (d / "frames.jsonl").write_text("\n".join(lines) + "\n")
    return d


RAW = {1: (0, 0, 250), 2: (0, 250, 0), 3: (250, 0, 250)}
PAIR_BASE = 2147483647          # COLMAP: pair_id = min_id * kMaxNumImages + max_id
BLACK = (0, 0, 0)


def _raw_walk(tmp_path):
    """An old world whose session followed two captures, the second restarting the
    numbering. Keyframes (CAPTURE_ID, 1), (CAPTURE_ID, 2), (CAP_B, 3): their raw frames are
    bright and unique; their stored (redacted) keyframes are black."""
    store, kids = _old_world(tmp_path)
    store.write_session(dataclasses.replace(store.read_session(W1, S1), capture_id=CAPTURE_ID))
    root = tmp_path / "caproot"
    _capture(root, CAPTURE_ID, [(1, 101.0, RAW[1]), (2, 102.0, RAW[2])])
    _capture(root, CAP_B, [(1, 201.0, (250, 0, 0)), (2, 202.0, (250, 250, 0)),
                           (3, 203.0, RAW[3])], continues=CAPTURE_ID)
    for seq, t in ((1, 101.0), (2, 102.0), (3, 203.0)):
        _jpeg(store.session_dir(W1, S1) / "images" / f"{seq:08d}.jpg", BLACK)
        store.append_keyframe(W1, Keyframe(
            keyframe_id=f"{S1}:{seq:08d}", session_id=S1, source_seq=seq, received_at=t,
            image_relpath=f"images/{seq:08d}.jpg", width=240, height=180, byte_count=1,
            wire_seq=seq))
    return store, kids, root


class _PreparingSolve(_Solve):
    """The solve child as far as the solver's images go: the REAL `prepare_images`, then
    the publish of the set-aside solution."""

    def __call__(self, argv, env=None, capture_output=True, text=True):
        _cam, self.written = GS.prepare_images(self.store, W1, S1,
                                               self.store.read_keyframes(W1, S1))
        return super().__call__(argv, env=env, capture_output=capture_output, text=text)


def _images(store):
    return GS.workspace_for(store, W1, S1).images_dir


def _walk_prepared_from_raw(store, root):
    """The walk's own solve directory as its builder left it: `sources.json` naming each
    keyframe's raw frame, and the solver images `prepare_images` undistorted from them."""
    ws = GS.workspace_for(store, W1, S1)
    raw = {f"{S1}:{s:08d}": str(root / "captures" / c / "frames" / f"{s:08d}.jpg")
           for c, s in ((CAPTURE_ID, 1), (CAPTURE_ID, 2), (CAP_B, 3))}
    GS.write_sources_records(ws, raw)
    GS.prepare_images(store, W1, S1, store.read_keyframes(W1, S1))
    return ws


def test_m7_d1_a_walk_solved_from_redacted_copies_gets_its_raw_frames(tmp_path, stages,
                                                                      monkeypatch):  # noqa: F811
    """RV9-D D1: re-finished once without `TOWER_CAPTURE_ROOT` (redacted solver images),
    then with it: every raw frame was found by identity and NONE reached the solver,
    because the carried-back images counted as `already_undistorted`."""
    store, kids, root = _raw_walk(tmp_path)
    monkeypatch.delenv(wr.CAPTURE_ROOT_ENV, raising=False)
    wr.refinish(store, tmp_path, W1, S1, solve_runner=_PreparingSolve(store, kids,
                                                                     gate_writes=False),
                stamp="r1")
    assert all(_near(_mean(p), BLACK) for p in _images(store).iterdir())
    monkeypatch.setenv(wr.CAPTURE_ROOT_ENV, str(root))
    second = _PreparingSolve(store, kids, gate_writes=False)
    report = wr.refinish(store, tmp_path, W1, S1, solve_runner=second, stamp="r2")
    frames = report["solver_frames"]
    assert frames["source"] == wr.SOURCE_RAW and frames["already_undistorted"] == 0
    assert frames["walk_images"]["withheld"] == {"differs-from-plan": 3}
    assert second.written == 3
    for seq in (1, 2, 3):
        assert _near(_mean(_images(store) / f"{seq:08d}.jpg"), RAW[seq])


def test_m7_d2_another_captures_frame_in_the_walk_does_not_survive(tmp_path, stages,
                                                                   monkeypatch):  # noqa: F811
    """RV9-D D2: the walk's by-name lookup undistorted keyframe (CAPTURE_ID, 2) from CAP_B's
    frame 2. It is withheld and rewritten from its own frame; the two right images are
    proven by reproduction and carried back."""
    store, kids, root = _raw_walk(tmp_path)
    monkeypatch.setenv(wr.CAPTURE_ROOT_ENV, str(root))
    ws = GS.workspace_for(store, W1, S1)
    # The builder recorded keyframes 1 and 3; keyframe 2 was found BY NAME in CAP_B -- the
    # defect of 3abe763's lookup, which records nothing.
    GS.write_sources_records(ws, {
        f"{S1}:00000001": str(root / "captures" / CAPTURE_ID / "frames" / "00000001.jpg"),
        f"{S1}:00000003": str(root / "captures" / CAP_B / "frames" / "00000003.jpg")})
    GS.prepare_images(store, W1, S1, store.read_keyframes(W1, S1),
                      capture_dirs=[root / "captures" / CAP_B])
    assert _near(_mean(ws.images_dir / "00000002.jpg"), (250, 250, 0))   # another moment
    solve = _PreparingSolve(store, kids, gate_writes=False)
    report = wr.refinish(store, tmp_path, W1, S1, solve_runner=solve, stamp="w")
    assert _near(_mean(ws.images_dir / "00000002.jpg"), RAW[2])      # its own moment
    assert solve.written == 1
    assert Path(GS.read_sources(ws)[f"{S1}:00000002"]).parent.parent.name == CAPTURE_ID
    walk = report["solver_frames"]["walk_images"]
    assert walk["withheld_images"] == {"00000002.jpg": "differs-from-plan"}
    assert walk["carried_back"] == 2 and walk["by_reproduction"] == 2


def test_m7_d3_a_duplicate_keyframe_id_is_solved_from_its_stored_copy(tmp_path, stages,
                                                                      monkeypatch):  # noqa: F811
    """RV9-D D3: keyframes (CAPTURE_ID, 1) and (CAP_B, 1) share id and image name. The
    contract says both keep their stored (redacted) copy; before the fix they were solved
    from the walk's raw image of one of them."""
    store, kids = _old_world(tmp_path)
    store.write_session(dataclasses.replace(store.read_session(W1, S1), capture_id=CAPTURE_ID))
    root = tmp_path / "caproot"
    _capture(root, CAPTURE_ID, [(1, 101.0, (0, 0, 250))])
    _capture(root, CAP_B, [(1, 201.0, (250, 0, 0)), (2, 202.0, (0, 250, 0))],
             continues=CAPTURE_ID)
    for seq, t in ((1, 101.0), (1, 201.0), (2, 202.0)):
        _jpeg(store.session_dir(W1, S1) / "images" / f"{seq:08d}.jpg", BLACK)
        store.append_keyframe(W1, Keyframe(
            keyframe_id=f"{S1}:{seq:08d}", session_id=S1, source_seq=seq, received_at=t,
            image_relpath=f"images/{seq:08d}.jpg", width=240, height=180, byte_count=1,
            wire_seq=seq))
    monkeypatch.setenv(wr.CAPTURE_ROOT_ENV, str(root))
    ws = GS.workspace_for(store, W1, S1)
    GS.write_sources_records(ws, {f"{S1}:00000001": str(root / "captures" / CAPTURE_ID /
                                                        "frames" / "00000001.jpg")})
    GS.prepare_images(store, W1, S1, store.read_keyframes(W1, S1))   # the walk's: raw
    assert _near(_mean(ws.images_dir / "00000001.jpg"), (0, 0, 250))
    solve = _PreparingSolve(store, kids, gate_writes=False)
    report = wr.refinish(store, tmp_path, W1, S1, solve_runner=solve, stamp="amb")
    frames = report["solver_frames"]
    assert frames["redacted_ambiguous_in_session"] == 2
    assert frames["walk_images"]["withheld_images"]["00000001.jpg"] == "differs-from-plan"
    assert _near(_mean(ws.images_dir / "00000001.jpg"), BLACK)     # its stored copy


def test_m7_images_proven_by_reproduction_are_carried_back_and_recorded(
        tmp_path, stages, monkeypatch):  # noqa: F811
    """The positive half: a walk whose images ARE its raw frames keeps them (no rewrite, so
    the frozen matching and mask cache keyed by their SHA-1 still apply), and the fresh solve
    directory records their provenance for the next re-finish."""
    store, kids, root = _raw_walk(tmp_path)
    monkeypatch.setenv(wr.CAPTURE_ROOT_ENV, str(root))
    ws = _walk_prepared_from_raw(store, root)
    before = {p.name: p.read_bytes() for p in ws.images_dir.iterdir()}
    solve = _PreparingSolve(store, kids, gate_writes=False)
    report = wr.refinish(store, tmp_path, W1, S1, solve_runner=solve, stamp="p")
    assert (ws.root / "images.provenance.json").is_file(), "the record SOL reads is written"
    walk = report["solver_frames"]["walk_images"]
    assert walk["carried_back"] == 3 and walk["by_reproduction"] == 3 and not walk["withheld"]
    assert report["solver_frames"]["already_undistorted"] == 3
    assert solve.written == 0
    assert {p.name: p.read_bytes() for p in ws.images_dir.iterdir()} == before
    record = json.loads((ws.root / wr.SOLVER_IMAGES_PROVENANCE).read_text())
    assert record["record"] == wr.PROVENANCE_RECORD
    entry = record["images"]["00000002.jpg"]
    assert entry["source"] == wr.IMAGE_SOURCE_RAW and entry["keyframe_id"] == f"{S1}:00000002"
    assert Path(entry["frame"]).parent.parent.name == CAPTURE_ID
    from tower.world_builder.solve_masks import file_sha1

    assert entry["sha1"] == file_sha1(ws.images_dir / "00000002.jpg")
    # ... and a third re-finish trusts that record, without reproducing anything.
    monkeypatch.setattr(wr, "_solver_image_sha1", lambda *a, **k: pytest.fail("reproduced"))
    third = wr.refinish(store, tmp_path, W1, S1, solve_runner=_Solve(store, kids,
                                                                    gate_writes=False),
                        stamp="q")
    assert third["solver_frames"]["walk_images"]["by_record"] == 3


def test_m7_a_record_is_trusted_only_for_the_planned_frame_and_its_own_bytes(
        tmp_path, stages, monkeypatch):  # noqa: F811
    store, kids, root = _raw_walk(tmp_path)
    monkeypatch.setenv(wr.CAPTURE_ROOT_ENV, str(root))
    ws = _walk_prepared_from_raw(store, root)
    from tower.world_builder.solve_masks import file_sha1

    planned = {n: str(root / "captures" / c / "frames" / n)
               for c, n in ((CAPTURE_ID, "00000001.jpg"), (CAPTURE_ID, "00000002.jpg"),
                            (CAP_B, "00000003.jpg"))}
    # Images the reproduction cannot vouch for (not what `prepare_images` writes) ...
    for n, colour in (("00000001.jpg", RAW[1]), ("00000002.jpg", RAW[2]),
                      ("00000003.jpg", RAW[3])):
        _jpeg(ws.images_dir / n, colour)
    entries = {n: {"keyframe_id": f"{S1}:{n[:-4]}", "source": "raw", "frame": planned[n],
                   "sha1": file_sha1(ws.images_dir / n)} for n in planned}
    entries["00000002.jpg"]["sha1"] = "0" * 40                       # ... not its bytes
    entries["00000003.jpg"]["frame"] = planned["00000001.jpg"]      # ... not its frame
    (ws.root / "images.provenance.json").write_text(json.dumps(       # the agreed format
        {"record": "wb-solver-image-provenance/1", "images": entries}))
    report = wr.refinish(store, tmp_path, W1, S1,
                         solve_runner=_Solve(store, kids, gate_writes=False), stamp="t")
    assert sorted(p.name for p in ws.images_dir.iterdir()) == ["00000001.jpg"]
    walk = report["solver_frames"]["walk_images"]
    assert walk["by_record"] == 1
    assert walk["withheld_images"] == {"00000002.jpg": "differs-from-plan",
                                       "00000003.jpg": "differs-from-plan"}


def _walk_database(path: Path, names) -> None:
    """A COLMAP-shaped database: one image per name, keypoints, descriptors, and a match
    and a two-view geometry for every pair."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()   # the fixture's placeholder bytes, in tmp_path
    con = sqlite3.connect(str(path))
    try:
        con.executescript(
            "create table images (image_id integer primary key, name text, camera_id int);"
            "create table keypoints (image_id int primary key, rows int, cols int, data blob);"
            "create table descriptors (image_id int primary key, rows int, cols int, data blob);"
            "create table matches (pair_id int primary key, rows int, cols int, data blob);"
            "create table two_view_geometries (pair_id int primary key, rows int, cols int,"
            " data blob, config int);")
        for i, name in enumerate(names, start=1):
            con.execute("insert into images values (?, ?, 1)", (i, name))
            con.execute("insert into keypoints values (?, 1, 2, x'00')", (i,))
            con.execute("insert into descriptors values (?, 1, 128, x'00')", (i,))
        n = len(names)
        for a in range(1, n + 1):
            for b in range(a + 1, n + 1):
                pid = a * PAIR_BASE + b
                con.execute("insert into matches values (?, 20, 2, x'00')", (pid,))
                con.execute("insert into two_view_geometries values (?, 20, 2, x'00', 2)", (pid,))
        con.commit()
    finally:
        con.close()


def _rows(path: Path):
    con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        return {t: sorted(r[0] for r in con.execute(f"select * from {t}"))
                for t in ("images", "keypoints", "descriptors", "matches", "two_view_geometries")}
    finally:
        con.close()


def test_m7_a_withheld_image_loses_its_features_in_the_copied_database(tmp_path, stages,
                                                                       monkeypatch):  # noqa: F811
    """COLMAP's extractor skips a name that still has keypoints, and its matcher a pair that
    still has matches: a rewritten image would be mapped with the OLD image's features. So a
    withheld image's keypoints, descriptors and every pair touching it leave the re-finish's
    COPY of the walk database; the walk's own database is untouched."""
    store, kids, root = _raw_walk(tmp_path)
    monkeypatch.setenv(wr.CAPTURE_ROOT_ENV, str(root))
    ws = _walk_prepared_from_raw(store, root)
    _jpeg(ws.images_dir / "00000002.jpg", (250, 250, 0))            # not its own frame
    names = ["00000001.jpg", "00000002.jpg", "00000003.jpg"]
    _walk_database(ws.database_path, names)
    walk_rows = _rows(ws.database_path)
    report = wr.refinish(store, tmp_path, W1, S1,
                         solve_runner=_Solve(store, kids, gate_writes=False), stamp="db")
    aside_db = store.world_dir(W1) / "refinish" / "db" / "solve" / S1 / "database.db"
    assert _rows(aside_db) == walk_rows
    fresh = _rows(ws.database_path)
    base = PAIR_BASE
    assert fresh["images"] == [1, 2, 3]                              # the rows stay
    assert fresh["keypoints"] == [1, 3] and fresh["descriptors"] == [1, 3]
    assert fresh["matches"] == [1 * base + 3] and fresh["two_view_geometries"] == [1 * base + 3]
    entry = next(c for c in _ledger(store, "db")["copied_back"] if c["name"] == "database.db")
    assert entry["features_cleared"] == {"images": 1, "pairs": 2}
    assert report["solver_frames"]["walk_images"]["features_to_clear"] == 1


def test_m7_the_provenance_record_is_the_one_global_solve_reads():
    """The format is agreed with SOL (`global_solve.prepare_images` honours it)."""
    name = getattr(GS, "SOLVER_IMAGE_PROVENANCE_FILENAME", wr.SOLVER_IMAGES_PROVENANCE)
    record = getattr(GS, "SOLVER_IMAGE_PROVENANCE_RECORD", wr.PROVENANCE_RECORD)
    assert (wr.SOLVER_IMAGES_PROVENANCE, wr.PROVENANCE_RECORD) == (name, record)
    assert wr.SOLVER_IMAGES_PROVENANCE not in wr.SOLVE_COPY_BACK   # written, not copied


# ---------------------------------------------------------------------------
# M-12: the prediction cache is not copied into the set-aside
# ---------------------------------------------------------------------------


def test_m12_the_prediction_cache_is_left_out_of_the_set_aside_copy(tmp_path, stages):  # noqa: F811
    from tower.world_builder import dense_pipeline

    store, kids = _old_world(tmp_path)
    dense = store.world_dir(W1) / "dense" / S1
    (dense / "predictions" / "tok").mkdir(parents=True)
    (dense / "predictions" / "tok" / "a.npy").write_bytes(b"x" * 1000)
    (dense / "align.json").write_text("{}")
    (dense / "work" / "predictions").mkdir(parents=True)             # deeper: not the cache
    (dense / "work" / "predictions" / "keep.txt").write_text("k")
    wr.refinish(store, tmp_path, W1, S1, solve_runner=_Solve(store, kids, gate_writes=False),
                stamp="p")
    aside = store.world_dir(W1) / "refinish" / "p" / "dense" / S1
    assert (aside / "align.json").exists()
    assert (aside / "work" / "predictions" / "keep.txt").exists()
    assert not (aside / "predictions").exists()
    assert (dense / "predictions" / "tok" / "a.npy").read_bytes() == b"x" * 1000   # untouched
    assert wr.DENSE_PREDICTIONS_DIRNAME == dense_pipeline.PREDICTIONS_DIRNAME
    entry = next(c for c in _ledger(store, "p")["copied"] if c["kind"] == "dense")
    assert entry["skipped"] == ["predictions"] and "re-derivable" in entry["skipped_why"]


# ---------------------------------------------------------------------------
# M-4: the given-up notice carries fixed phrases, never raw error text
# ---------------------------------------------------------------------------

RAW_TEXT = ("RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 8.00 GiB "
            "total capacity; 6.10 GiB already allocated) at "
            r"C:\Users\tvllo\Projects\Glasses\tower\.venv\Lib\moge\model.py:411"
            "\n\tsecond line")


@pytest.mark.parametrize("gate", [
    {"state": "applied", "retryable": True, "cause": "depth-unavailable",
     "masks_applied": True, "metric_available": False,
     "depth": {"state": "unavailable", "detail": RAW_TEXT}},
    {"state": "failed", "retryable": True, "cause": "gate-failed", "detail": RAW_TEXT},
    {"state": "failed", "retryable": True, "cause": "gate-failed",
     "detail": "DatabaseError: file is not a database"},
], ids=["depth-oom", "gate-failed", "database"])
def test_m4_the_given_up_notice_has_no_raw_text(gate):
    """RV9-E F1b: the given-up sentence put `gate.detail` -- an exception, GiB figures, a
    local path, a newline -- on the phone word for word."""
    meta = {"gate": gate, "transients": {"state": "applied"}}
    for sentence in (lambda m: wfp.regate_given_up_notice(m, 3),
                     lambda m: wfp.regate_refused_notice(m)):
        notice = sentence(meta)
        for raw in ("Error", "GiB", "tvllo", "\\", "/", "\n", "model.py", "Tried"):
            assert raw not in notice, (raw, notice)
        assert "an owner can re-finish this walk" in notice
    assert "stopped trying" in wfp.regate_given_up_notice(meta, 3)


# ---------------------------------------------------------------------------
# LOW, the finisher
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("refusal", [{"database": False}, {"loadable": False}])
def test_low_a_refused_regate_stops_promising_and_stops_waiting(tmp_path, refusal):
    """RV9-E F2: the row kept "the Tower re-runs the gate when it is idle" for ever and
    every run exited `EXIT_WAITING`, driving the chore's backoff to an hour for every world."""
    from tests.test_world_builder_finish_pending_gate import ROOM_OK, _gated

    store = _gated(tmp_path, stages=ROOM_OK, **refusal)
    codes = [_run(tmp_path) for _ in range(6)]
    assert codes == [0] * 6, codes
    notice = store.read_session("w1", "s1").finalization["notice"]
    assert "re-runs" not in notice and "an owner can re-finish this walk" in notice
    assert not wfp.read_attempts(store, "w1", wfp.regate_ledger_key("s1"))


def _clean_gate_runner(store, world_id, session_id, candidate, **kwargs):
    from tower.world_builder import coherence_publish as CP

    record = {"state": CP.GATE_STATE_APPLIED, "retryable": False, "cause": None,
              "masks_applied": True, "metric_available": True, "attach": True,
              "depth": {"state": CP.DEPTH_OK}, "components": None}
    return CP.GateResult(solution=candidate, record=record, components=None)


@pytest.mark.parametrize("how", ["the write fails", "a kill"])
def test_low_the_notice_is_reconciled_when_its_write_did_not_happen(tmp_path, monkeypatch, how):
    """RV9-E F3a/F3b: a re-gate published, and the notice write after it failed (or the
    process died): the next run rebuilt the room and the row promised a re-gate for ever."""
    from tests.test_world_builder_finish_pending_gate import ROOM_OK, _gated
    from tower.world_builder import coherence_publish as CP
    from tower.world_builder.engine import WorldBuilderEngine

    rooms = []

    def room(store_, world_id, session_id, **kwargs):
        rooms.append(session_id)
        kwargs["record"]("surface", state="ok", detail=None)
        kwargs["record"]("appearance", state="ok", detail=None)

    monkeypatch.setattr(wfp, "final_surface_stages", room)
    store = _gated(tmp_path, stages=ROOM_OK)
    assert "re-runs" in store.read_session("w1", "s1").finalization["notice"]
    real_regate = CP.regate_published
    if how == "the write fails":
        monkeypatch.setattr(CP, "regate_published",
                            lambda *a, **k: real_regate(*a, gate_runner=_clean_gate_runner, **k))
        real_mark = WorldBuilderEngine.mark_finalization
        fails = {"n": 1}

        def mark(self, *a, **k):
            if fails["n"]:
                fails["n"] -= 1
                raise PermissionError(5, "Access is denied", "session.json")
            return real_mark(self, *a, **k)

        monkeypatch.setattr(WorldBuilderEngine, "mark_finalization", mark)
        assert _run(tmp_path) == 1
    else:
        def killed(store_, world_id, session_id, should_stop=None):
            real_regate(store_, world_id, session_id, gate_runner=_clean_gate_runner)
            raise KeyboardInterrupt("killed")

        monkeypatch.setattr(CP, "regate_published", killed)
        with pytest.raises(KeyboardInterrupt):
            _run(tmp_path)
    assert load_solution(store, "w1", "s1").gate["retryable"] is False   # published
    assert [_run(tmp_path) for _ in range(2)] == [0, 0]
    assert rooms == ["s1"]
    fin = store.read_session("w1", "s1").finalization
    assert "notice" not in fin, fin              # the published solve owes nothing
    assert not wfp.assess(store, "w1", "s1").owed


def test_low_a_put_back_leaves_the_finishers_counters_as_they_were(tmp_path, stages):  # noqa: F811
    """RV9-D probe B: a room the finisher had retired at its bound was owed again after a
    re-finish that changed nothing -- step 1 restarted the counters and the put-back never
    wrote them back."""
    store, kids = _old_world(tmp_path)
    session = store.read_session(W1, S1)
    store.write_session(dataclasses.replace(session, stages={
        "surface": {"state": "stopped", "attempted": True, "detail": "killed"},
        "appearance": {"state": "stopped", "attempted": True, "detail": "killed"}}))
    wfp._write_ledger(store, W1, {S1: {"attempts": wfp.DEFAULT_MAX_ATTEMPTS, "forgiven": 0,
                                       "detail": "x"}})
    assert wfp.assess(store, W1, S1).code == "attempt-bound"
    report = wr.refinish(store, tmp_path, W1, S1, solve_runner=_Solve(store, kids, code=1),
                         stamp="b")
    assert report["done"] is False and "put back" in report["reason"]
    after = wfp.assess(store, W1, S1)
    assert after.code == "attempt-bound" and after.exhausted and not after.owed, after
    assert wfp.read_attempts(store, W1, S1) == wfp.DEFAULT_MAX_ATTEMPTS


def test_low_an_older_ledgers_restarted_counters_go_back_with_the_put_back(
        tmp_path, dead_process, monkeypatch):
    """A re-finish from before this fix restarted the counters in step 1 and kept the old
    ones in its ledger; its put-back (here the finisher's, after it died) writes them back."""
    store, _kids = _old_world(tmp_path)
    exhausted = {"attempts": 3, "forgiven": 0, "detail": "re-gate failed"}
    wfp._write_ledger(store, W1, {wfp.regate_ledger_key(S1): dict(exhausted)})
    ledger = _step_one_then_killed(store, "k", dead_process)
    ledger["previous"]["finish_attempts"] = wr._restart_attempts(store, W1, S1, "k")
    _write_ledger(store, "k", ledger)
    assert wfp.read_attempts(store, W1, wfp.regate_ledger_key(S1)) == 0
    _no_room_build(monkeypatch)
    _run(tmp_path)
    assert wfp._read_ledger(store, W1)[0][wfp.regate_ledger_key(S1)] == exhausted


# ---------------------------------------------------------------------------
# LOW, the re-finish
# ---------------------------------------------------------------------------


def _one_frame_capture(tmp_path, wire_seq):
    d = tmp_path / "captures" / "c1"
    (d / "frames").mkdir(parents=True)
    (d / "frames" / "00000001.jpg").write_bytes(b"x")
    (d / "frames.jsonl").write_text(json.dumps({
        "source_seq": 1, "wire_seq": wire_seq, "received_at": 10.0,
        "time_basis": "tower-receipt", "relpath": "frames/00000001.jpg"}) + "\n")
    return d


def _keyframe(wire_seq=1):
    return Keyframe(keyframe_id="s:00000001", session_id="s", source_seq=1, received_at=10.0,
                    image_relpath="images/00000001.jpg", width=1, height=1, byte_count=1,
                    wire_seq=wire_seq)


@pytest.mark.parametrize("wire_seq, matched", [("1.0", False), ("x", False), (True, False),
                                               ("1", True), (1.0, True)])
def test_low_a_malformed_wire_seq_is_no_match_and_raises_nothing(tmp_path, wire_seq, matched):
    """RV9-D probe F: `int(wire_seq)` on one malformed journal field raised out of the whole
    plan. A value that is not an integer cannot confirm the identity: that record is no
    match, and the keyframe keeps its stored copy."""
    mapping, why = wr.map_raw_frames([_keyframe()], [_one_frame_capture(tmp_path, wire_seq)])
    assert (mapping == {"s:00000001": str(tmp_path / "captures" / "c1" / "frames" /
                                          "00000001.jpg")}) is matched
    assert (why["no_frame"] == ["s:00000001"]) is not matched


def test_low_a_relative_capture_dir_is_resolved_from_the_working_directory(tmp_path,
                                                                            monkeypatch):
    """RV9-D probe G: a relative `--capture-dir` was counted raw here and then resolved by
    the solve against `tower/` (or `TOWER_SOURCES_ROOT`): not found, silently redacted."""
    store, _kids, root = _raw_walk(tmp_path / "w")
    monkeypatch.chdir(root)
    monkeypatch.setenv("TOWER_SOURCES_ROOT", str(tmp_path / "elsewhere"))
    got = wr.resolve_capture_dirs(store, W1, S1, [f"captures/{CAPTURE_ID}"])
    assert got["capture_dirs"] == [str((root / "captures" / CAPTURE_ID).resolve())]
    planned = wr.plan_solver_frames(store, W1, S1, got["capture_dirs"])
    entries = {k: v for k, v in planned["sources"].items() if k.startswith(f"{S1}:")}
    assert len(entries) == 2 and planned["raw_from_capture_dir"] == 2
    assert all(GS.resolve_source_path(v).is_file() for v in entries.values())


def test_low_an_exception_in_steps_3_and_4_ends_the_ledger_truthfully(tmp_path, monkeypatch):
    store, kids = _old_world(tmp_path)

    def broken_room(*a, **k):
        raise RuntimeError("the depth network fell over")

    monkeypatch.setattr(wbs, "final_surface_stages", broken_room)
    report = wr.refinish(store, tmp_path, W1, S1,
                         solve_runner=_Solve(store, kids, gate_writes=False), stamp="s")
    assert report["done"] is False and "RuntimeError" in report["error"]
    ledger = _ledger(store, "s")
    assert ledger["state"] == wr.LEDGER_STOPPED and "the depth network" in ledger["detail"]
    assert ledger["state"] in wr.LEDGER_TERMINAL_STATES
    assert load_solution(store, W1, S1) is not None                 # the new solve stands
    assert store.lock_holder(W1) is None


def test_low_a_pid_without_a_start_time_is_unknown_and_logged(caplog):
    with caplog.at_level(logging.WARNING, logger=wr.__name__):
        assert wr.refinish_process_alive({"stamp": "t", "process": {"pid": os.getpid()}}) is None
    assert any("start time" in r.getMessage() for r in caplog.records)
    from tower.world_builder.store import _lock_record

    assert wr.refinish_process_alive({"process": _lock_record(os.getpid())}) is True
