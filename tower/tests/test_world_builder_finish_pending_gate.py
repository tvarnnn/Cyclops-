"""The finisher with the evidence gate on: review V8, M1 and M3b.

Four ways the idle Tower's finisher misbehaved once a solve carries a gate record
(`scripts/world_finish_pending.py`), each pinned here by a test that failed before the
fix (the reviewer's probe: `RUN/baseline/review/V8/finisher/test_v8_finisher_probe.py`):

* M1a -- a re-gate that cannot start (the solution will not load, the solve's database is
  gone) was reported "waiting" but spent the run's one-world budget, and the run exited
  `EXIT_MORE_OWED`; the Tower respawns that with no backoff, so the refused world looped
  and every other owed world starved.
* M1b -- at the re-gate's attempt bound its verdict returned before the room and area
  checks, so owed room and area work was hidden for ever and the row kept promising a
  re-run.
* M1c -- a stop after the re-gate published left the room built from the OLD partition
  with nothing owed.
* M3b -- a live re-finish (`scripts/world_refinish.py`) marks the room `stopped` between
  its steps, and an idle Tower's finisher picked that up in the lock gaps.

And the constraint: a world with no gate record takes today's path, exactly.

The GPU stages and the gate itself are faked; the solutions are real (`write_solution`),
because the refusal is decided by whether the published solve loads.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import world_finish_pending as wfp  # noqa: E402
from scripts.world_build_session import StopRequest  # noqa: E402
from tests.test_world_builder_finish_pending import _stage, _world  # noqa: E402
from tower.world_builder import coherence_publish as CP  # noqa: E402
from tower.world_builder.global_solve import (  # noqa: E402
    Solution,
    workspace_for,
    write_solution,
)
from tower.world_builder.records import (  # noqa: E402
    STAGE_APPEARANCE,
    STAGE_STATE_OK,
    STAGE_STATE_RUNNING,
    STAGE_STATE_STOPPED,
    STAGE_SURFACE,
)
from tower.world_builder.store import WorldStore  # noqa: E402

DEPTH_LOST = {"state": CP.GATE_STATE_APPLIED, "masks_applied": True, "metric_available": False,
              "retryable": True, "cause": CP.CAUSE_DEPTH_UNAVAILABLE,
              "depth": {"state": "unavailable",
                        "detail": "DepthModelUnavailable: moge is not installed"}}
GATE_FAILED = {"state": CP.GATE_STATE_FAILED, "retryable": True, "cause": CP.CAUSE_GATE_FAILED,
               "detail": "ZeroDivisionError: a bug"}
ROOM_OK = {STAGE_SURFACE: _stage(STAGE_STATE_OK), STAGE_APPEARANCE: _stage(STAGE_STATE_OK)}
ROOM_INTERRUPTED = {STAGE_SURFACE: _stage(STAGE_STATE_RUNNING)}
N_KF = 4


@pytest.fixture(autouse=True)
def _no_native_warm(monkeypatch):
    monkeypatch.setattr(wfp, "prewarm_world_builder", lambda *a, **k: ())


@pytest.fixture
def stage_runner(monkeypatch):
    """The finisher's `final_surface_stages`, recorded; it writes `ok` like the real one."""
    calls = []

    def fake(store, world_id, session_id, **kwargs):
        holder = store.lock_holder(world_id)
        calls.append({"world": world_id, "session": session_id,
                      "lock_pid": None if holder is None else holder["pid"]})
        kwargs["record"](STAGE_SURFACE, state=STAGE_STATE_OK, detail=None)
        if kwargs.get("appearance"):
            kwargs["record"](STAGE_APPEARANCE, state=STAGE_STATE_OK, detail=None)
        return {"surface": {"attempted": True, "state": STAGE_STATE_OK}}

    monkeypatch.setattr(wfp, "final_surface_stages", fake)
    return calls


def _solution(session_id: str, gate: dict | None, *, transients=None, solved_at=5.0) -> Solution:
    kids = [f"{session_id}:{i:08d}" for i in range(1, N_KF + 1)]
    poses = {kid: {"component": 0, "rotation": np.eye(3).reshape(-1).tolist(),
                   "translation": [float(i), 0.0, 0.0], "observations": 10}
             for i, kid in enumerate(kids)}
    return Solution(
        solver="glomap", solved_at=solved_at, input_digest=f"digest-{session_id}",
        keyframe_ids=kids, poses=poses, components=[{"index": 0, "images": N_KF, "points": N_KF}],
        xyz=np.zeros((N_KF, 3), np.float32), rgb=np.zeros((N_KF, 3), np.uint8),
        component=np.zeros(N_KF, np.int32), first_keyframe=np.arange(N_KF, dtype=np.int32),
        track_length=np.full(N_KF, 2, np.int32), error=np.full(N_KF, 0.5, np.float32),
        observations=np.array([[i, 0, i] for i in range(N_KF)], np.int32),
        transients=transients if transients is not None else {"state": "applied"},
        gate=gate)


def _gated(root, world_id="w1", session_id="s1", *, stages, gate=DEPTH_LOST, database=True,
           loadable=True, transients=None, notice=True) -> WorldStore:
    """A stopped, finalized session whose published solve carries `gate`.

    `loadable=False` is the probe's shape: `solution.json` alone, which `load_solution`
    reads as absent. `notice` puts the row's sentence the publish wrote into the
    finalization detail, as `world_build_session` / `world_finalize` do."""
    store = _world(root, world_id=world_id, session_id=session_id, stages=stages)
    workspace = workspace_for(store, world_id, session_id)
    if loadable:
        write_solution(workspace, _solution(session_id, gate, transients=transients))
    else:
        workspace.root.mkdir(parents=True, exist_ok=True)
        workspace.solution_path.write_text(json.dumps(
            {"transients": transients or {"state": "applied"}, "gate": gate}))
    if database:
        workspace.database_path.write_bytes(b"the walk's features (not read: the gate is faked)")
    if notice and gate is not None:
        detail = CP.publish_notice({"gate": gate, "transients": transients or {"state": "applied"}})
        session = store.read_session(world_id, session_id)
        fin = dict(session.finalization, detail=detail)
        if detail:
            # And the phone's copy, as the builder writes it since contract v6 (§3.1).
            fin["notice"] = detail
        from dataclasses import replace

        store.write_session(replace(session, finalization=fin))
    return store


def _area_record(store, world_id="w1", session_id="s1") -> None:
    """A components record (hand-written: no solve identity, so it is taken as it is) with
    the room and one area, whose build nothing has settled yet."""
    from tower.world_builder import components as C

    kids = [f"{session_id}:{i:08d}" for i in range(1, N_KF + 1)]
    C.components_path(store, world_id, session_id).write_text(json.dumps({"components": [
        {"id": "0123456789abcdef", "state": "placed", "reason": None, "reasons": [],
         "shown_as": "room", "keyframes": 2, "capture_spans_s": [[0.0, 2.0]],
         "keyframe_ids": kids[:2]},
        {"id": "a1a1a1a1a1a1a1a1", "state": "unplaced", "reason": "scale-unavailable",
         "reasons": ["scale-unavailable"], "shown_as": "area", "keyframes": 2,
         "capture_spans_s": [[3.0, 9.0]], "keyframe_ids": kids[2:]}]}), encoding="utf-8")


def _ledger(store, world_id="w1") -> dict:
    path = store.world_dir(world_id) / wfp.ATTEMPTS_FILENAME
    return json.loads(path.read_text(encoding="utf-8"))["sessions"] if path.exists() else {}


def _run(root, **kwargs) -> int:
    return wfp.main(["--root", str(root), "--format", "json"], stop_request=StopRequest(),
                    **kwargs)


# ---------------------------------------------------------------------------
# M1a: a refused re-gate is decided read-only, waits, and starves nothing
# ---------------------------------------------------------------------------

REFUSALS = {
    "database-gone": {"database": False},
    "solution-will-not-load": {"loadable": False},
}


@pytest.mark.parametrize("refusal", sorted(REFUSALS))
def test_m1a_a_re_gate_that_cannot_start_is_decided_read_only(tmp_path, refusal):
    store = _gated(tmp_path, stages=ROOM_OK, **REFUSALS[refusal])
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    v = wfp.assess(store, "w1", "s1")
    assert not v.owed and v.code == "regate-refused" and v.stage == wfp.REGATE_STAGE, v
    assert v.code in wfp.WAITING_CODES
    # Read-only: no lock, no ledger, not a byte changed.
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("refusal", sorted(REFUSALS))
def test_m1a_a_refused_re_gate_does_not_starve_another_owed_world(tmp_path, stage_runner,
                                                                   refusal):
    """The reviewer's probe: before the fix every run exited `EXIT_MORE_OWED` (a respawn
    with no backoff) and b1 was never reached."""
    a0 = _gated(tmp_path, world_id="a0", stages=ROOM_OK, **REFUSALS[refusal])
    a0_before = a0.session_path("a0", "s1").read_bytes()
    _world(tmp_path, world_id="b1", stages=ROOM_INTERRUPTED)
    codes = [_run(tmp_path) for _ in range(5)]
    assert [c["world"] for c in stage_runner] == ["b1"], "b1 is finished, once, on the first run"
    assert wfp.EXIT_MORE_OWED not in codes, codes
    # Only waiting work is left: the Tower asks again after its backoff.
    assert codes == [wfp.EXIT_WAITING] * 5, codes
    # And the refused world was never worked on: no attempt, no lock, no record written.
    assert wfp.regate_ledger_key("s1") not in _ledger(a0, "a0")
    assert a0.session_path("a0", "s1").read_bytes() == a0_before
    assert a0.lock_holder("a0") is None


def test_m1a_a_re_gate_refused_under_the_lock_spends_no_budget(tmp_path, stage_runner,
                                                              monkeypatch):
    """The backstop: the solve looked fine to the survey and was refused under the lock
    (its database went between the two). Waiting, so the budget is not spent, the same run
    goes on to b1, and it exits WAITING -- not "more owed"."""
    a0 = _gated(tmp_path, world_id="a0", stages=ROOM_OK)
    _world(tmp_path, world_id="b1", stages=ROOM_INTERRUPTED)
    assert wfp.assess(a0, "a0", "s1").code == "owed-regate"
    stages_before = a0.read_session("a0", "s1").stages
    calls = []

    def refused(store, world_id, session_id, **kwargs):
        calls.append(world_id)
        raise CP.RegateRefused("the solve's database database.db is gone")

    monkeypatch.setattr(CP, "regate_published", refused)
    assert _run(tmp_path) == wfp.EXIT_WAITING
    assert calls == ["a0"]
    assert [c["world"] for c in stage_runner] == ["b1"]
    assert wfp.read_attempts(a0, "a0", wfp.regate_ledger_key("s1")) == 0
    # M1c's mark before the publish is undone: a refusal changed nothing.
    assert a0.read_session("a0", "s1").stages == stages_before


def _second_session(store, stages, world_id="w1", session_id="s2"):
    """Another stopped, finalized session of the same world."""
    from dataclasses import replace

    from tower.world_builder.records import Session

    world = store.read_world(world_id)
    store.write_world(replace(world, session_ids=tuple(world.session_ids) + (session_id,)))
    first = store.read_session(world_id, world.session_ids[0])
    store.write_session(Session(session_id=session_id, world_id=world_id, started_at=30.0,
                                ended_at=40.0, end_reason="stop",
                                finalization=dict(first.finalization, detail=None),
                                stages=stages))


def test_m1a_a_refused_session_does_not_hold_back_the_rest_of_its_world(
        tmp_path, stage_runner, monkeypatch):
    """A refusal is about ONE session's solve: that session is not retried in the run,
    and the world's other owed session is finished in the same run."""
    store = _gated(tmp_path, stages=ROOM_OK)
    _second_session(store, ROOM_INTERRUPTED)
    calls = []

    def refused(store_, world_id, session_id, **kwargs):
        calls.append(session_id)
        raise CP.RegateRefused("the solve's database database.db is gone")

    monkeypatch.setattr(CP, "regate_published", refused)
    assert _run(tmp_path) == wfp.EXIT_WAITING
    assert calls == ["s1"]
    assert [c["session"] for c in stage_runner] == ["s2"]


def test_m1a_a_world_whose_lock_was_taken_is_not_picked_again_in_the_run(
        tmp_path, stage_runner, monkeypatch):
    """Another writer took w1 between the survey and the re-gate: waiting, so the budget is
    not spent and w1's other owed session is not attempted in this run (it would only meet
    the same lock); b1 is finished, and the Tower asks again after its backoff."""
    from tower.world_builder.store import WorldLockedError

    store = _gated(tmp_path, stages=ROOM_OK)
    _second_session(store, ROOM_INTERRUPTED)
    _world(tmp_path, world_id="x9", stages=ROOM_INTERRUPTED)
    real = WorldStore.acquire_writer_lock
    tried = []

    def acquire(self, world_id):
        tried.append(world_id)
        if world_id == "w1":
            raise WorldLockedError("w1 is held by live pid 4242")
        return real(self, world_id)

    monkeypatch.setattr(WorldStore, "acquire_writer_lock", acquire)
    assert wfp.assess(store, "w1", "s1").code == "owed-regate"
    assert _run(tmp_path) == wfp.EXIT_WAITING
    assert tried == ["w1", "x9"], tried
    assert [c["world"] for c in stage_runner] == ["x9"]


@pytest.mark.parametrize("shape", ["database-gone", "solution-will-not-load", "accepted"])
def test_the_read_only_refusal_agrees_with_regate_published(tmp_path, shape):
    """The finisher's read-only check restates `regate_published`'s two refusals (it may
    not be the one that decides: coherence_publish.py belongs to another lane). This pins
    the two to one answer on every shape."""
    kwargs = REFUSALS.get(shape, {})
    store = _gated(tmp_path, stages=ROOM_OK, **kwargs)
    reason = wfp._regate_refusal(store, "w1", "s1")

    class Reached(Exception):
        pass

    def gate_runner(*a, **k):
        raise Reached()

    try:
        CP.regate_published(store, "w1", "s1", gate_runner=gate_runner)
    except CP.RegateRefused as exc:
        assert reason == str(exc)
    except Reached:
        assert reason is None
    else:  # pragma: no cover -- the stub always raises
        pytest.fail("regate_published neither refused nor reached the gate")


# ---------------------------------------------------------------------------
# M1b: at the attempt bound the re-gate is given up on the record, and what it hid is done
# ---------------------------------------------------------------------------


def _at_the_bound(store, world_id="w1", session_id="s1"):
    for _ in range(wfp.DEFAULT_MAX_ATTEMPTS):
        wfp.record_attempt(store, world_id, wfp.regate_ledger_key(session_id), detail="x")


def test_m1b_at_the_bound_the_interrupted_room_is_finished_in_the_same_run(tmp_path,
                                                                          stage_runner):
    store = _gated(tmp_path, stages=ROOM_INTERRUPTED)
    _at_the_bound(store)
    promise = store.read_session("w1", "s1").finalization["detail"]
    assert "re-runs" in promise
    assert _run(tmp_path) == 0
    assert [c["world"] for c in stage_runner] == ["w1"], "the room behind the bound is built"
    detail = store.read_session("w1", "s1").finalization["detail"]
    assert "re-runs" not in detail and "stopped trying" in detail
    assert "moge is not installed" in detail and "re-finish" in detail
    v = wfp.assess(store, "w1", "s1")
    assert not v.owed and v.code == "nothing-interrupted", v
    # Settled: a second run finds nothing and writes nothing.
    before = store.session_path("w1", "s1").read_bytes()
    assert _run(tmp_path) == 0
    assert len(stage_runner) == 1
    assert store.session_path("w1", "s1").read_bytes() == before


def test_m1b_the_bound_is_exhausted_until_the_give_up_is_on_record(tmp_path):
    """The probe's third case. Before: `attempt-bound`, not exhausted, not owed, for ever,
    over an interrupted room."""
    store = _gated(tmp_path, stages=ROOM_INTERRUPTED)
    _at_the_bound(store)
    v = wfp.assess(store, "w1", "s1")
    assert v.code == "attempt-bound" and v.stage == wfp.REGATE_STAGE
    assert v.exhausted and not v.owed
    out = wfp._retire(store, v, wfp.DEFAULT_MAX_ATTEMPTS)
    assert out["retired"] == wfp.REGATE_STAGE, out
    assert store.lock_holder("w1") is None
    after = wfp.assess(store, "w1", "s1")
    assert after.owed and after.stage == STAGE_SURFACE and after.code == "owed", after


def test_m1b_at_the_bound_owed_areas_are_settled(tmp_path, stage_runner, monkeypatch):
    """Areas the new record names are owed behind the room. With area builds off (the
    default) the finisher declines them -- a settled word -- in the same run."""
    monkeypatch.delenv("TOWER_WORLD_AREA_BUILDS", raising=False)
    store = _gated(tmp_path, stages=ROOM_OK)
    _area_record(store)
    _at_the_bound(store)
    assert wfp.assess(store, "w1", "s1").stage == wfp.REGATE_STAGE
    assert _run(tmp_path) == 0
    assert stage_runner == []                 # declined, not built: no GPU work
    v = wfp.assess(store, "w1", "s1")
    assert not v.owed and v.code == "nothing-interrupted", v
    assert "stopped trying" in store.read_session("w1", "s1").finalization["detail"]


def test_m1b_the_given_up_notice_keeps_the_masks_sentence(tmp_path):
    meta = {"transients": {"state": "unavailable", "cause": "gpu-oom", "retryable": True},
            "gate": GATE_FAILED}
    notice = wfp.regate_given_up_notice(meta, 3)
    assert notice.startswith(CP.NOTICE_MASKS_OOM)
    assert "the evidence gate failed (ZeroDivisionError: a bug)" in notice
    assert "re-runs" not in notice and "stopped trying" in notice and "3 times" in notice


def test_m1b_a_detail_rewritten_after_the_give_up_is_given_up_again(tmp_path, stage_runner):
    """The record IS the sentence: anything that puts the promise back (a hand-run
    `world_finalize.py` on the same solve) is corrected at the next run. Since contract v6
    the sentence the record is read from is `finalization.notice`, which such a publish
    rewrites together with `detail`."""
    store = _gated(tmp_path, stages=ROOM_OK)
    _at_the_bound(store)
    assert _run(tmp_path) == 0
    given_up = store.read_session("w1", "s1").finalization["detail"]
    from dataclasses import replace

    promise = CP.publish_notice({"gate": DEPTH_LOST})
    session = store.read_session("w1", "s1")
    store.write_session(replace(session, finalization=dict(
        session.finalization, detail=promise, notice=promise)))
    assert wfp.assess(store, "w1", "s1").exhausted
    assert _run(tmp_path) == 0
    assert store.read_session("w1", "s1").finalization["detail"] == given_up
    assert store.read_session("w1", "s1").finalization["notice"] == given_up


# ---------------------------------------------------------------------------
# M1c: the room is owed from before the publish, so no interruption leaves it stale
# ---------------------------------------------------------------------------


def _republishing(store, *, then=None):
    """A stand-in for `regate_published`: the gate ran with depth this time, the solution
    and its record are republished (no longer retryable) -- then `then()`."""
    calls = []

    def regate(store_, world_id, session_id, should_stop=None):
        calls.append(world_id)
        cleared = dict(DEPTH_LOST, retryable=False, cause=None, metric_available=True)
        write_solution(workspace_for(store_, world_id, session_id),
                       _solution(session_id, cleared, solved_at=6.0))
        if then is not None:
            then()
        return {"notice": None}

    regate.calls = calls
    return regate


def test_m1c_a_stop_after_the_re_gate_publishes_leaves_the_room_owed(tmp_path, monkeypatch):
    """The probe's second case. Before: `finished: False` and `nothing-interrupted` -- the
    room kept the old partition and nothing was owed."""
    store = _gated(tmp_path, stages=ROOM_OK)
    monkeypatch.setattr(wfp.WorldBuilderEngine, "build", lambda self, w, s: pytest.fail("built"))
    stop = StopRequest()
    regate = _republishing(store, then=lambda: stop.request(StopRequest.SOFT, "capture opened"))
    v = wfp.assess(store, "w1", "s1")
    assert v.code == "owed-regate"
    out = wfp.finish_regate(store, v, appearance=True, prune_depth_work=False,
                            stop_request=stop, regate=regate,
                            surface_stages=lambda *a, **k: pytest.fail("room rebuilt"))
    assert out["finished"] is False and regate.calls == ["w1"]
    after = wfp.assess(store, "w1", "s1")
    assert after.owed and after.stage == STAGE_SURFACE and after.code == "owed", after
    surface = store.read_session("w1", "s1").stages[STAGE_SURFACE]
    assert surface["state"] == STAGE_STATE_STOPPED and "re-gate" in surface["detail"]
    assert store.lock_holder("w1") is None


def test_m1c_a_process_killed_after_the_publish_leaves_the_room_owed(tmp_path, monkeypatch):
    """A kill has no `finally` that could mark anything: the room had to be owed BEFORE the
    publish. An interrupt raised right after it stands in for the kill."""
    store = _gated(tmp_path, stages=ROOM_OK)

    def killed():
        raise KeyboardInterrupt

    regate = _republishing(store, then=killed)
    v = wfp.assess(store, "w1", "s1")
    with pytest.raises(KeyboardInterrupt):
        wfp.finish_regate(store, v, appearance=True, prune_depth_work=False,
                          stop_request=StopRequest(), regate=regate,
                          surface_stages=lambda *a, **k: pytest.fail("room rebuilt"))
    after = wfp.assess(store, "w1", "s1")
    assert after.owed and after.stage == STAGE_SURFACE, after


def test_m1c_the_room_is_marked_before_the_publish(tmp_path):
    store = _gated(tmp_path, stages=ROOM_OK)
    seen = {}

    def regate(store_, world_id, session_id, should_stop=None):
        seen.update(store_.read_session(world_id, session_id).stages)
        raise CP.RegateRefused("the solve's database database.db is gone")

    v = wfp.assess(store, "w1", "s1")
    out = wfp.finish_regate(store, v, appearance=True, prune_depth_work=False,
                            stop_request=StopRequest(), regate=regate)
    assert {seen[s]["state"] for s in (STAGE_SURFACE, STAGE_APPEARANCE)} == {STAGE_STATE_STOPPED}
    # ... and a refusal, which did nothing, puts the room's record back as it was.
    assert out["waiting"] is True
    assert store.read_session("w1", "s1").stages == ROOM_OK


def test_m1c_a_finished_re_gate_rebuilds_the_room_and_owes_nothing(tmp_path, monkeypatch,
                                                                  stage_runner):
    store = _gated(tmp_path, stages=ROOM_OK)
    monkeypatch.setattr(wfp.WorldBuilderEngine, "build",
                        lambda self, w, s: SimpleNamespace(poses_solved=N_KF))
    v = wfp.assess(store, "w1", "s1")
    out = wfp.finish_regate(store, v, appearance=True, prune_depth_work=False,
                            stop_request=StopRequest(), regate=_republishing(store))
    assert out["finished"] is True, out
    assert len(stage_runner) == 1
    v = wfp.assess(store, "w1", "s1")
    assert not v.owed and v.code == "nothing-interrupted", v


def test_m1c_a_stop_before_the_re_gate_starts_publishes_nothing(tmp_path):
    store = _gated(tmp_path, stages=ROOM_OK)
    stop = StopRequest()
    stop.request(StopRequest.SOFT, "capture opened")
    v = wfp.assess(store, "w1", "s1")
    out = wfp.finish_regate(store, v, appearance=True, prune_depth_work=False,
                            stop_request=stop,
                            regate=lambda *a, **k: pytest.fail("published after a stop"))
    assert out["finished"] is False
    assert store.read_session("w1", "s1").stages == ROOM_OK
    assert wfp.read_attempts(store, "w1", wfp.regate_ledger_key("s1")) == 0


# ---------------------------------------------------------------------------
# M3b: the finisher stays out of a live re-finish
# ---------------------------------------------------------------------------


def _refinish_ledger(store, stamp, state, *, world_id="w1", session_id="s1", raw=None):
    from scripts import world_refinish as wr

    aside = store.world_dir(world_id) / wr.REFINISH_DIRNAME / stamp
    aside.mkdir(parents=True, exist_ok=True)
    text = raw if raw is not None else json.dumps(
        {"command": "scripts/world_refinish.py", "state": state, "world_id": world_id,
         "session_id": session_id, "stamp": stamp})
    (aside / wr.LEDGER_FILENAME).write_text(text, encoding="utf-8")


def _marked(stamp):
    from scripts import world_refinish as wr

    detail = wr.REFINISH_IN_PROGRESS.format(stamp=stamp)
    return {STAGE_SURFACE: _stage(STAGE_STATE_STOPPED, detail=detail),
            STAGE_APPEARANCE: _stage(STAGE_STATE_STOPPED, detail=detail)}


# (ledger state or a raw malformed text, room marked by that stamp) -> skipped?
LEDGER_CASES = {
    "setting-aside": ("setting-aside", False, True),
    "set-aside, room marked": ("set-aside", True, True),
    "set-aside, the room no longer carries its marker": ("set-aside", False, False),
    "restored after a failed solve": ("restored-after-a-failed-solve", False, False),
    "rolled back": ("rolled-back", False, False),
    "malformed, room marked": (None, True, True),
    "malformed, room not marked": (None, False, False),
    "an unknown state, room marked": ("a-state-from-the-future", True, True),
}


@pytest.mark.parametrize("case", sorted(LEDGER_CASES))
def test_m3b_the_ledger_and_the_room_decide_whether_a_refinish_is_live(tmp_path, case):
    state, marked, skipped = LEDGER_CASES[case]
    # An interrupted room either way, so "today's behaviour" is visibly `owed`.
    stages = _marked("t1") if marked else {STAGE_SURFACE: _stage(STAGE_STATE_STOPPED)}
    store = _world(tmp_path, stages=stages)
    _refinish_ledger(store, "t1", state, raw="{not json" if state is None else None)
    v = wfp.assess(store, "w1", "s1")
    if skipped:
        assert not v.owed and v.code == "refinish-in-progress", v
        assert v.code in wfp.WAITING_CODES
    else:
        assert v.owed and v.code == "owed", v


def test_m3b_another_stamps_marker_does_not_hold_a_finished_ledger(tmp_path):
    store = _world(tmp_path, stages=_marked("t2"))
    _refinish_ledger(store, "t1", "restored-after-an-error")
    assert wfp.assess(store, "w1", "s1").code == "owed"


def test_m3b_a_world_being_refinished_is_skipped_logged_and_waits(tmp_path, stage_runner,
                                                                  caplog):
    store = _world(tmp_path, stages=_marked("t1"))
    _refinish_ledger(store, "t1", "set-aside")
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with caplog.at_level(logging.INFO, logger=wfp.__name__):
        assert _run(tmp_path) == wfp.EXIT_WAITING
    assert stage_runner == []
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
    assert any("re-finish" in r.getMessage() and "w1" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("work", ["room", "regate", "areas"])
def test_m3b_a_verdict_older_than_the_refinish_is_asked_again_under_the_lock(
        tmp_path, stage_runner, monkeypatch, work):
    """The survey runs at the start of a run, and the world's turn can come minutes later.
    An owner who started a re-finish in between has marked the room under the world's
    lock; the finisher asks again once the lock is its own, and waits."""
    monkeypatch.setenv("TOWER_WORLD_AREA_BUILDS", "1")
    if work == "room":
        store = _gated(tmp_path, stages={STAGE_SURFACE: _stage(STAGE_STATE_STOPPED)}, gate=None)
    else:
        store = _gated(tmp_path, stages=ROOM_OK,
                       gate=DEPTH_LOST if work == "regate" else dict(DEPTH_LOST, retryable=False))
    if work == "areas":
        _area_record(store)
    verdict = wfp.assess(store, "w1", "s1")
    assert verdict.owed and verdict.stage == {"room": STAGE_SURFACE, "regate": wfp.REGATE_STAGE,
                                              "areas": wfp.AREAS_STAGE}[work], verdict
    # ... and now an owner's re-finish takes the world and marks its room.
    from dataclasses import replace

    _refinish_ledger(store, "t1", "set-aside")
    session = store.read_session("w1", "s1")
    store.write_session(replace(session, stages=_marked("t1")))
    monkeypatch.setattr(CP, "regate_published", lambda *a, **k: pytest.fail("re-gated"))
    before = _ledger(store)
    out = wfp.finish(store, verdict, appearance=True, prune_depth_work=False,
                     stop_request=StopRequest())
    assert out["finished"] is False and out["waiting"] is True and out["world_busy"] is True
    assert "t1" in out["reason"]
    assert stage_runner == []
    assert _ledger(store) == before, "no attempt was counted"
    assert store.lock_holder("w1") is None
    assert store.read_session("w1", "s1").stages == _marked("t1")


def test_m3b_the_finisher_stays_out_of_a_live_refinish_between_its_steps(tmp_path, monkeypatch,
                                                                         stage_runner):
    """The real `world_refinish.refinish`, with the idle Tower's finisher run in both lock
    gaps: after step 1 (the room marked `stopped`, the solve set aside) and after step 2
    (the new solve published, the child gone). Before the fix it rebuilt the room in the
    first gap, under the re-finish's feet."""
    from scripts import world_build_session as wbs
    from scripts import world_refinish as wr

    store = _world(tmp_path, stages=ROOM_OK)
    write_solution(workspace_for(store, "w1", "s1"), _solution("s1", None))
    workspace_for(store, "w1", "s1").database_path.write_bytes(b"old features")
    refinish_rooms = []

    def room(store_, world_id, session_id, **kwargs):
        refinish_rooms.append(session_id)
        kwargs["record"](STAGE_SURFACE, state=STAGE_STATE_OK, detail=None)
        kwargs["record"](STAGE_APPEARANCE, state=STAGE_STATE_OK, detail=None)
        return {"surface": {"attempted": True, "state": STAGE_STATE_OK}}

    monkeypatch.setattr(wbs, "final_surface_stages", room)
    in_the_gaps = []

    def solve(argv, env=None, capture_output=True, text=True):
        in_the_gaps.append(_run(tmp_path))                       # the gap 1 -> 2
        write_solution(workspace_for(store, "w1", "s1"), _solution("s1", None, solved_at=7.0))
        in_the_gaps.append(_run(tmp_path))                       # the gap 2 -> 3
        return SimpleNamespace(returncode=0, stdout=json.dumps({"finalized": True}), stderr="")

    report = wr.refinish(store, tmp_path, "w1", "s1", solve_runner=solve, stamp="t1")
    assert report["done"] is True, report
    assert in_the_gaps == [wfp.EXIT_WAITING, wfp.EXIT_WAITING]
    assert stage_runner == [], "the finisher built nothing while the re-finish was live"
    assert "s1" not in _ledger(store) or _ledger(store)["s1"]["attempts"] == 0
    assert refinish_rooms == ["s1"]
    # Finished: the ledger stays `set-aside`, but the room it marked was rebuilt, so the
    # world is back on today's path.
    v = wfp.assess(store, "w1", "s1")
    assert v.code == "nothing-interrupted", v
    assert _run(tmp_path) == 0


# ---------------------------------------------------------------------------
# The constraint: no gate, today's path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("solve", ["no gate record", "no solve at all"])
def test_a_world_without_a_gate_takes_todays_path_exactly(tmp_path, stage_runner, monkeypatch,
                                                           solve):
    store = _world(tmp_path, stages=ROOM_INTERRUPTED)
    if solve == "no gate record":
        write_solution(workspace_for(store, "w1", "s1"), _solution("s1", None))
    # None of the gate's machinery may even be consulted. (`raising=False`: this test runs,
    # and passes, on the code before the fix as well -- that is what "exactly" means.)
    monkeypatch.setattr(wfp, "_regate_refusal", lambda *a, **k: pytest.fail("refusal probed"),
                        raising=False)
    monkeypatch.setattr(wfp, "regate_given_up_notice",
                        lambda *a, **k: pytest.fail("notice computed"), raising=False)
    monkeypatch.setattr(wfp.WorldBuilderEngine, "mark_finalization",
                        lambda *a, **k: pytest.fail("finalization rewritten"))
    finalization = store.read_session("w1", "s1").finalization
    v = wfp.assess(store, "w1", "s1")
    assert v.owed and v.code == "owed" and v.stage == STAGE_SURFACE
    assert _run(tmp_path) == 0
    assert [c["world"] for c in stage_runner] == ["w1"]
    assert _ledger(store) == {"s1": {"attempts": 1, "forgiven": 0,
                                     "detail": "finishing the surface stage"}}
    assert store.read_session("w1", "s1").finalization == finalization
    assert not (store.world_dir("w1") / "refinish").exists()
