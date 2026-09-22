"""Photographic work that was interrupted is finished, automatically, later.

THE GAP THIS CLOSES. The surface and appearance stages run exactly once, in
the builder child, in the six to sixteen minutes AFTER Stop and AFTER the
world writer lock is released. Anything that ends that child first -- a Tower
shutdown, a machine sleep, a crash, the supervisor's thirty-second stop grace
-- discards the work, and until now NOTHING EVER RETRIED IT.
`scripts/world_finalize.py` rebuilds only the sparse derived tree; the serving
path in `tower/results/world_builder*.py` is strictly read-only and spawns
nothing; and no startup reconciliation existed anywhere. It happened for real
on 2026-09-22: a Tower shut down eight minutes into a build, and the next
start recovered nothing.

WHAT THESE TESTS ARE REALLY DEFENDING is the "owed" predicate, because the
predicate is where this feature can do harm. Rebuilding a world costs six to
sixteen minutes of GPU. There are 165 historical worlds on this machine whose
session records predate the stage record entirely, and rebuilding those would
be tens of hours of GPU nobody asked for. So "no stage record at all" must
mean NOT OWED -- that is the single most important test in this file -- and a
world some other process is still writing must mean NOT OWED too.

AND THE PREDICATE IS ALSO WHERE THIS FEATURE CAN BE POINTLESS. An adversarial
review measured the first version against the real root: 70 sessions, ZERO
owed -- including the very world the incident was about. The stage record it
selected on did not exist yet, and would not exist until a Tower carrying the
commit had itself been interrupted. A selector that is a no-op on the only
evidence there is has protected nothing. So there is a second signal, and the
tests below pin BOTH halves of it: what the surface stage's own `status.json`
already says, and the fact that reading it cannot widen the net -- ONE world
in 166 on this machine has a `surface/` directory at all.
"""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import world_finish_pending as wfp  # noqa: E402
from scripts.world_build_session import StopRequest  # noqa: E402
from tower.world_builder.records import (  # noqa: E402
    FINAL_SOLVE_FAILED,
    FINAL_SOLVE_SOLVED,
    FINALIZATION_COMPLETE,
    FINALIZATION_INTERRUPTED,
    STAGE_APPEARANCE,
    STAGE_STATE_FAILED,
    STAGE_STATE_OK,
    STAGE_STATE_RUNNING,
    STAGE_STATE_STOPPED,
    STAGE_STATE_UNAVAILABLE,
    STAGE_SURFACE,
    Session,
    World,
)
from tower.world_builder.store import WorldStore  # noqa: E402


# -- fixtures ---------------------------------------------------------


def _stage(state, *, attempted=True, detail=None):
    return {
        "attempted": attempted,
        "state": state,
        "started_at": 10.0,
        "updated_at": 11.0,
        "detail": detail,
    }


def _finalization(state=FINALIZATION_COMPLETE, final_solve=FINAL_SOLVE_SOLVED):
    return {
        "state": state,
        "final_solve": final_solve,
        "started_at": 9.0,
        "updated_at": 10.0,
        "detail": None,
    }


def _world(
    root: Path,
    *,
    world_id="w1",
    session_id="s1",
    stages,
    finalization=None,
    ended_at=20.0,
) -> WorldStore:
    """A stopped session on disk, in whatever state the test needs."""
    store = WorldStore(root)
    store.write_world(
        World(
            world_id=world_id,
            created_at=1.0,
            updated_at=2.0,
            session_ids=(session_id,),
        )
    )
    store.write_session(
        Session(
            session_id=session_id,
            world_id=world_id,
            started_at=10.0,
            ended_at=ended_at,
            end_reason="stop",
            finalization=_finalization() if finalization is None else finalization,
            stages=stages,
        )
    )
    return store


INTERRUPTED_SURFACE = {STAGE_SURFACE: _stage(STAGE_STATE_RUNNING)}


def _verdict(store, world_id="w1", session_id="s1", **kwargs):
    return wfp.assess(store, world_id, session_id, **kwargs)


# -- the "owed" predicate ---------------------------------------------


def test_a_surface_left_running_by_a_process_that_is_gone_is_owed(tmp_path):
    """The signature of an interrupted build.

    `mark_stage` writes `running` BEFORE the stage starts, deliberately: a
    builder the Job Object kills thirty seconds into a six-minute surface
    cannot say anything afterwards, so `running` under a pid that is gone is
    the only honest record of exactly that. It is also, now, a work order.
    """
    store = _world(tmp_path, stages=INTERRUPTED_SURFACE)
    verdict = _verdict(store)
    assert verdict.owed is True
    assert verdict.stage == STAGE_SURFACE


def test_a_surface_that_was_stopped_is_owed(tmp_path):
    """`stopped` is what `final_surface_stages` records when a hard stop
    arrives before the stage runs. Nothing else will ever pick it up."""
    store = _world(
        tmp_path,
        stages={
            STAGE_SURFACE: _stage(
                STAGE_STATE_STOPPED,
                attempted=False,
                detail="hard stop (SIGBREAK) during finalization",
            )
        },
    )
    assert _verdict(store).owed is True


def test_an_interrupted_appearance_on_a_finished_surface_is_owed(tmp_path):
    store = _world(
        tmp_path,
        stages={
            STAGE_SURFACE: _stage(STAGE_STATE_OK),
            STAGE_APPEARANCE: _stage(STAGE_STATE_RUNNING),
        },
    )
    verdict = _verdict(store)
    assert verdict.owed is True
    assert verdict.stage == STAGE_APPEARANCE


def test_a_finished_photographic_world_is_not_owed(tmp_path):
    store = _world(
        tmp_path,
        stages={
            STAGE_SURFACE: _stage(STAGE_STATE_OK),
            STAGE_APPEARANCE: _stage(STAGE_STATE_OK),
        },
    )
    assert _verdict(store).owed is False


def test_a_session_with_no_stages_record_and_no_surface_work_is_not_owed(tmp_path):
    """THE HISTORICAL-WORLD GUARD, and the most expensive test to get wrong.

    Absent is not a state in the vocabulary. It means "a Tower that never
    recorded this", which is every one of the 165 worlds on this machine
    that predate 2026-09-22. Treating absence as "interrupted" would put
    tens of hours of GPU work into a queue nobody asked for, on worlds that
    are finished and fine.

    "And no surface work": a session with no stage record AND no photographic
    artifact on disk is a world no Tower ever tried to make a picture of.
    That is the shape of 165 of the 166 worlds here, and it stays untouched.
    """
    store = _world(tmp_path, stages=None)
    verdict = _verdict(store)
    assert verdict.owed is False
    assert verdict.code == "no-stage-record"


def test_an_empty_stages_object_with_no_surface_work_is_also_not_owed(tmp_path):
    """`{}` says as little as absence does, and must be read as carefully."""
    store = _world(tmp_path, stages={})
    assert _verdict(store).owed is False


# -- the second signal: what the stage itself left on disk -------------
#
# Measured on the real root before this existed: 70 sessions, 0 owed, and the
# ONE genuinely interrupted world -- 2f447162 / cb308801, a Tower shut down
# eight minutes into a build on 2026-09-22 -- skipped as `no-stage-record`.
# Its `surface/cb308801.../status.json` reads {"state": "stopped", "stage":
# "depth", "pid": 7964}, written 82 seconds after its finalization completed.
# The signal was there, unambiguous, and unread.


def _surface_status(store, state, *, world_id="w1", session_id="s1",
                    stage=STAGE_SURFACE, pid=999999, age=0.0):
    path = store.world_dir(world_id) / stage / session_id / "status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "schema_version": 1,
            "pid": pid,
            "updated_at": time.time() - age,
            "state": state,
            "stage": "depth",
        }),
        encoding="utf-8",
    )
    return path


def test_a_stopped_surface_status_is_owed_when_no_stage_record_exists(tmp_path):
    """THE REAL INCIDENT, in the shape it is actually on disk.

    A `status.json` saying `stopped` cannot be written by a Tower that never
    ran a photographic stage, so reading it cannot discover a historical
    backlog -- which is the exact property the stage-record clause exists to
    guarantee. It is the same guarantee from a different file.
    """
    store = _world(tmp_path, stages=None)
    _surface_status(store, "stopped")
    verdict = _verdict(store)
    assert verdict.owed is True
    assert verdict.stage == STAGE_SURFACE
    assert verdict.code == "owed-by-status"


def test_a_running_surface_status_under_a_dead_pid_is_owed(tmp_path):
    """The other interrupted shape, decided by the SAME staleness helper the
    rest of the system uses (`surface_pipeline.status_is_stale`) rather than
    by a pid check invented here."""
    store = _world(tmp_path, stages=None)
    _surface_status(store, "running", pid=999999, age=600.0)
    assert _verdict(store).owed is True


def test_a_running_surface_status_under_a_live_pid_is_not_owed(tmp_path):
    """Something is building it right now. Nothing here may touch it."""
    store = _world(tmp_path, stages=None)
    _surface_status(store, "running", pid=os.getpid())
    verdict = _verdict(store)
    assert verdict.owed is False
    assert verdict.code == "building-now"


def test_a_finished_surface_status_is_not_owed(tmp_path):
    """`ok` is terminal from a status file exactly as it is from a record."""
    store = _world(tmp_path, stages=None)
    _surface_status(store, "ok")
    assert _verdict(store).owed is False


def test_a_failed_surface_status_is_not_owed(tmp_path):
    """And so is `failed`: it will fail the same way again."""
    store = _world(tmp_path, stages=None)
    _surface_status(store, "failed")
    assert _verdict(store).owed is False


def test_an_interrupted_appearance_status_is_owed(tmp_path):
    store = _world(tmp_path, stages=None)
    _surface_status(store, "ok")
    _surface_status(store, "stopped", stage=STAGE_APPEARANCE)
    verdict = _verdict(store)
    assert verdict.owed is True
    assert verdict.stage == STAGE_APPEARANCE


def test_the_stage_record_wins_wherever_there_is_one(tmp_path):
    """The record is the more precise signal and it is written LAST.

    A session whose record says the surface finished is finished, whatever a
    `status.json` from an earlier coarse build still says -- the live child
    writes one of those during every walk. Reading the file over the record
    would resurrect a world that is already done.
    """
    store = _world(tmp_path, stages={STAGE_SURFACE: _stage(STAGE_STATE_OK)})
    _surface_status(store, "stopped")
    verdict = _verdict(store)
    assert verdict.owed is False
    assert verdict.code == "nothing-interrupted"


def test_an_unreadable_surface_status_is_not_owed(tmp_path):
    """Unreadable is not "interrupted". It is nothing at all."""
    store = _world(tmp_path, stages=None)
    _surface_status(store, "stopped").write_text("{not json", encoding="utf-8")
    assert _verdict(store).owed is False


def test_the_status_signal_still_obeys_every_other_clause(tmp_path):
    """The second signal widens WHICH stage-state evidence counts. It does
    not relax the finalization, the solve, the lock or the bound."""
    store = _world(
        tmp_path,
        stages=None,
        finalization=_finalization(final_solve=FINAL_SOLVE_FAILED),
    )
    _surface_status(store, "stopped")
    assert _verdict(store).owed is False


def test_a_stage_that_was_never_requested_is_not_owed(tmp_path):
    """`unavailable` is terminal: nobody asked for a photographic world.

    A sparse world here is a configuration, not a failure, and re-deciding
    it on the operator's behalf is not this tool's business.
    """
    store = _world(
        tmp_path,
        stages={
            STAGE_SURFACE: _stage(
                STAGE_STATE_UNAVAILABLE,
                attempted=False,
                detail="not requested (--surface was not passed)",
            )
        },
    )
    assert _verdict(store).owed is False


def test_a_stage_that_failed_is_not_owed(tmp_path):
    """`failed` is terminal too. A depth network that is not installed will
    not be installed by running it again, and retrying it every boot is how
    an automatic repair becomes an automatic GPU leak."""
    store = _world(
        tmp_path,
        stages={STAGE_SURFACE: _stage(STAGE_STATE_FAILED, detail="OSError: no weights")},
    )
    assert _verdict(store).owed is False


def test_a_session_whose_finalization_never_completed_is_not_owed(tmp_path):
    """There is nothing to build a surface FROM. `world_finalize.py` is the
    tool for this shape, and it is a different tool on purpose."""
    store = _world(
        tmp_path,
        stages=INTERRUPTED_SURFACE,
        finalization=_finalization(state=FINALIZATION_INTERRUPTED),
    )
    assert _verdict(store).owed is False


def test_a_session_whose_final_solve_did_not_land_is_not_owed(tmp_path):
    """A surface needs a global solve; `final_surface_stages` refuses without
    one and records `unavailable`. Queuing it would only re-record that."""
    store = _world(
        tmp_path,
        stages=INTERRUPTED_SURFACE,
        finalization=_finalization(final_solve=FINAL_SOLVE_FAILED),
    )
    assert _verdict(store).owed is False


def test_a_session_that_never_stopped_is_not_owed(tmp_path):
    """`ended_at: null` is a walk in progress or a builder that died mid-walk.
    Either way the photographic stages are not the thing it is missing."""
    store = _world(tmp_path, stages=INTERRUPTED_SURFACE, ended_at=None)
    assert _verdict(store).owed is False


# -- the two liveness gates -------------------------------------------


@pytest.fixture
def a_live_process():
    """A real process that is not this one, for a lock that must be believed."""
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        stdin=subprocess.PIPE,
    )
    try:
        yield child
    finally:
        child.kill()
        child.wait(timeout=10)


def test_a_world_whose_writer_lock_names_a_live_process_is_not_owed(
    tmp_path, a_live_process
):
    """A lock naming a LIVE process is an error, not something to force.

    `world_finalize.py` calls its lock "the whole safety story" and this
    tool inherits that judgement unchanged: another writer owns this world,
    so this one waits for another day.
    """
    store = _world(tmp_path, stages=INTERRUPTED_SURFACE)
    record = {"pid": a_live_process.pid}
    try:
        import psutil

        record["created_at"] = float(psutil.Process(a_live_process.pid).create_time())
    except Exception:  # pragma: no cover - psutil is a dependency, belt and braces
        pass
    lock = store.lock_path("w1")
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps(record), encoding="utf-8")

    verdict = _verdict(store)
    assert verdict.owed is False
    assert "lock" in verdict.reason


def test_a_stage_that_is_genuinely_running_right_now_is_not_owed(tmp_path):
    """The record says `running` and so does a `status.json` whose process is
    alive. That is a build in flight, not an interrupted one.

    The distinction is made by `session_build_running`, the SAME probe
    `_lifecycle` uses to report `build_in_progress`, rather than by a pid
    check invented here -- two answers to one question is how they drift.
    """
    store = _world(tmp_path, stages=INTERRUPTED_SURFACE)
    status = store.world_dir("w1") / "surface" / "s1" / "status.json"
    status.parent.mkdir(parents=True, exist_ok=True)
    status.write_text(
        json.dumps({"state": "running", "pid": os.getpid(), "updated_at": time.time()}),
        encoding="utf-8",
    )
    verdict = _verdict(store)
    assert verdict.owed is False
    assert "running" in verdict.reason


# -- the attempt bound ------------------------------------------------


def test_the_attempt_bound_stops_a_retry_that_keeps_being_interrupted(tmp_path):
    """The loop this closes is the tool's own.

    This CLI marks the surface `running` before it starts, exactly as the
    builder does. So a finisher that is itself killed leaves the very
    signature it selects on -- and without a bound it would be selected
    again at every boot, forever, on a world it can never finish.
    """
    store = _world(tmp_path, stages=INTERRUPTED_SURFACE)
    assert _verdict(store, max_attempts=2).owed is True
    wfp.record_attempt(store, "w1", "s1", detail="first")
    assert _verdict(store, max_attempts=2).owed is True
    wfp.record_attempt(store, "w1", "s1", detail="second")
    verdict = _verdict(store, max_attempts=2)
    assert verdict.owed is False
    assert verdict.exhausted is True


def test_an_exhausted_session_is_retired_into_the_record(tmp_path, monkeypatch):
    """"Bounded" is not enough on its own: the reason has to be legible to
    whoever opens the world afterwards and finds it still sparse."""
    store = _world(tmp_path, stages=INTERRUPTED_SURFACE)
    wfp.record_attempt(store, "w1", "s1", detail="one")
    wfp.record_attempt(store, "w1", "s1", detail="two")

    code = wfp.main(
        ["--root", str(tmp_path), "--max-attempts", "2", "--format", "json"]
    )
    assert code == 0
    surface = store.read_session("w1", "s1").stages[STAGE_SURFACE]
    assert surface["state"] == STAGE_STATE_FAILED
    assert "2" in (surface["detail"] or "")
    # And it stays retired: `failed` is terminal, so the next boot does not
    # look at it again.
    assert _verdict(store, max_attempts=2).owed is False


def test_the_retirement_names_both_stages_of_the_way_back(tmp_path):
    """The escape hatch has to actually work when followed.

    `world_surface.py --force` alone rebuilds the geometry and no shading,
    so a user who did exactly what the record told them got a grey mesh and
    no way to know why. A saved world is the surface AND the appearance.
    """
    store = _world(tmp_path, stages=INTERRUPTED_SURFACE)
    wfp.record_attempt(store, "w1", "s1", detail="one")
    assert wfp.main(["--root", str(tmp_path), "--max-attempts", "1"]) == 0
    detail = store.read_session("w1", "s1").stages[STAGE_SURFACE]["detail"]
    assert "world_surface.py" in detail
    assert "world_appearance.py" in detail


def test_retiring_takes_the_world_lock_and_looks_again_underneath_it(
    tmp_path, a_live_process
):
    """A TOCTOU the survey opens and the lock closes.

    `assess` runs over every session before anything is written, so a
    builder can take the world between the check and `_retire`'s write --
    and `mark_stage` is an unlocked read-modify-write of `session.json`,
    so that builder's own stage write is what would be lost.
    """
    store = _world(tmp_path, stages=INTERRUPTED_SURFACE)
    wfp.record_attempt(store, "w1", "s1", detail="one")

    # The world is surveyed as exhausted, and only THEN does a live writer
    # arrive -- exactly the window between the survey and the write.
    real_survey = wfp.survey

    def survey_then_lock(store_, **kwargs):
        verdicts = real_survey(store_, **kwargs)
        record = {"pid": a_live_process.pid}
        try:
            import psutil

            record["created_at"] = float(
                psutil.Process(a_live_process.pid).create_time()
            )
        except Exception:  # pragma: no cover
            pass
        lock = store_.lock_path("w1")
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text(json.dumps(record), encoding="utf-8")
        return verdicts

    wfp.survey, saved = survey_then_lock, wfp.survey
    try:
        assert wfp.main(["--root", str(tmp_path), "--max-attempts", "1"]) == 0
    finally:
        wfp.survey = saved
    # Nothing was written: the record still says what the builder left.
    assert store.read_session("w1", "s1").stages[STAGE_SURFACE]["state"] == (
        STAGE_STATE_RUNNING
    )


def test_a_walk_starting_does_not_spend_an_attempt(tmp_path, monkeypatch):
    """THE EXPECTED EVENT ON EVERY BOOT MUST BE FREE.

    Boot the Tower, then go for a walk: the chore starts a six-minute
    surface, the wearer presses Start a minute later, and the chore is
    stopped. Counting that as a failed attempt means three ordinary days
    retire a perfectly recoverable world to `failed` for ever.

    The bound is for "something keeps killing me". An attempt is spent only
    when this tool was LEFT ALONE and still did not finish, so a stop that
    was asked for gives the attempt back -- and it does so from the stop
    watcher's own thread, milliseconds after the pipe closes, because the
    five-second grace ends in `terminate_tree` and nothing this process
    does afterwards would ever run.
    """
    store = _world(tmp_path, stages=INTERRUPTED_SURFACE)
    stop = StopRequest()

    def stopped_mid_surface(store_, world_id, session_id, **kwargs):
        # The pipe closes while the depth pass is running. This is the
        # watcher thread's call, verbatim.
        stop.request(StopRequest.SOFT, "stdin-closed")
        raise KeyboardInterrupt("terminate_tree got here first")

    monkeypatch.setattr(wfp, "final_surface_stages", stopped_mid_surface)
    with pytest.raises(KeyboardInterrupt):
        wfp.main(["--root", str(tmp_path)], stop_request=stop)

    assert wfp.read_attempts(store, "w1", "s1") == 0
    # So the next boot still owes it, rather than counting down to `failed`.
    assert _verdict(store).owed is True


def test_a_failure_nobody_asked_for_does_spend_an_attempt(tmp_path, monkeypatch):
    """The other half, or the bound would bound nothing.

    A stage that dies with no stop request is the shape the bound exists
    for: a machine that sleeps, an out-of-memory kill, a driver that falls
    over. That attempt stands.
    """
    store = _world(tmp_path, stages=INTERRUPTED_SURFACE)

    def die(*args, **kwargs):
        raise RuntimeError("the depth network fell over")

    monkeypatch.setattr(wfp, "final_surface_stages", die)
    assert wfp.main(["--root", str(tmp_path)]) == 1
    assert wfp.read_attempts(store, "w1", "s1") == 1


def test_an_attempt_that_cannot_be_counted_is_never_started(tmp_path, monkeypatch):
    """An uncounted attempt is an unbounded one.

    `record_attempt` writes before the work, so a read-only or full disk
    made it raise BEFORE anything was counted -- and the bound then never
    advanced, at every boot, for ever. That is precisely the loop the bound
    exists to prevent, arriving through the bound's own front door.
    """
    _world(tmp_path, stages=INTERRUPTED_SURFACE)
    ran = []
    monkeypatch.setattr(wfp, "final_surface_stages",
                        lambda *a, **kw: ran.append(1))

    def refuse(*args, **kwargs):
        raise OSError(30, "Read-only file system")

    monkeypatch.setattr(wfp, "record_attempt", refuse)
    assert wfp.main(["--root", str(tmp_path)]) == 1
    assert ran == []


# -- the bound on forgiveness -----------------------------------------
#
# WHY THE SECTION ABOVE WAS NOT ENOUGH. Every test above bounds the RETRIES.
# None of them bounds the FORGIVENESS, and on 2026-09-22 that was the whole
# defect: the finisher deadlocked in the Windows native loader, sat at 0% CPU
# making no progress at all, was killed by the stop grace -- and was forgiven,
# because a stop had been requested. At every Tower start. The ledger on disk
# said, after all of them:
#
#     {"attempts": 0, "detail": "attempt given back: stopped (stdin-closed)"}
#
# Zero. The counter that is the ONLY mechanism able to retire an unfinishable
# world never passed zero, so the world reported "Improving" across unlimited
# restarts with nothing on disk explaining it. The root cause is fixed
# elsewhere (`tower/native_prewarm.py`, and the prewarm call before the stop
# watcher is armed); these tests are the containment, so that the next thing
# that hangs or is killed instantly cannot imply progress for ever either.
#
# WHAT MUST NOT BREAK IS THE LEGITIMATE CASE. "Boot the Tower, then go for a
# walk" ends this process on an ordinary start, and counting those would
# retire a perfectly recoverable world in three ordinary days. So the first
# test below is the one holding the other end of the rope.


def _a_stopped_run(tmp_path, monkeypatch, *argv, source="stdin-closed"):
    """One whole Tower start that is interrupted, end to end.

    The finisher begins the surface, the wearer presses Start, the pipe
    closes, and `terminate_tree` arrives before the stage reaches a
    `should_stop` checkpoint. Verbatim what the watcher thread does -- a
    fresh `StopRequest` each time, because a real one belongs to a real
    process and each of these is a different boot.
    """
    stop = StopRequest()

    def stopped_mid_surface(store_, world_id, session_id, **kwargs):
        stop.request(StopRequest.SOFT, source)
        raise KeyboardInterrupt("terminate_tree got here first")

    monkeypatch.setattr(wfp, "final_surface_stages", stopped_mid_surface)
    with pytest.raises(KeyboardInterrupt):
        wfp.main(["--root", str(tmp_path), *argv], stop_request=stop)


def _ledger_entry(store, world_id="w1", session_id="s1") -> dict:
    path = store.world_dir(world_id) / wfp.ATTEMPTS_FILENAME
    return json.loads(path.read_text(encoding="utf-8"))["sessions"][session_id]


def test_one_ordinary_stop_during_a_start_still_forgives(tmp_path, monkeypatch):
    """THE ORDINARY CASE IS UNHARMED, and it is checked first on purpose.

    A bound on forgiveness that costs the common event its free pass has
    not fixed anything, it has only moved the harm: "boot the Tower, then go
    for a walk" is what the product EXPECTS, and three of them must not
    retire a recoverable world. One stop, one attempt given back, and the
    forgiveness recorded so the NEXT one can be counted.
    """
    store = _world(tmp_path, stages=INTERRUPTED_SURFACE)
    _a_stopped_run(tmp_path, monkeypatch)

    assert wfp.read_attempts(store, "w1", "s1") == 0
    assert wfp.read_forgiven(store, "w1", "s1") == 1
    assert "given back" in _ledger_entry(store)["detail"]
    assert _verdict(store).owed is True


def test_a_session_forgiven_up_to_the_bound_is_still_recoverable(
    tmp_path, monkeypatch
):
    """Five ordinary days of walks cost this world nothing.

    The bound is FIVE (`DEFAULT_MAX_FORGIVEN`) because the event it forgives
    is a Tower start followed by a walk, and a busy day is a handful of
    those, not fifty. Up to the bound the attempts must stay at zero and the
    world must stay owed -- one start that is left alone long enough to
    finish clears it out of the owed set entirely and none of this matters
    again.
    """
    # THE NUMBER ITSELF IS LOAD-BEARING and is pinned here rather than only
    # read: a bound below a day's ordinary Tower starts makes the free pass
    # not free, and the loop below would sail through a bound of zero without
    # noticing.
    assert wfp.DEFAULT_MAX_FORGIVEN >= 3

    store = _world(tmp_path, stages=INTERRUPTED_SURFACE)
    for _ in range(wfp.DEFAULT_MAX_FORGIVEN):
        _a_stopped_run(tmp_path, monkeypatch)

    assert wfp.read_attempts(store, "w1", "s1") == 0
    assert wfp.read_forgiven(store, "w1", "s1") == wfp.DEFAULT_MAX_FORGIVEN
    assert _verdict(store).owed is True


def test_past_the_bound_forgiveness_stops_and_the_attempts_accumulate(
    tmp_path, monkeypatch
):
    """The defect itself: a stop is no longer taken as proof of progress.

    Past the bound a stop request still ends the run, but it no longer hands
    the attempt back -- so `attempts` climbs by one at every start, which is
    the thing that could never happen on 2026-09-22. Three more starts and
    `--max-attempts` is reached.
    """
    store = _world(tmp_path, stages=INTERRUPTED_SURFACE)
    for _ in range(wfp.DEFAULT_MAX_FORGIVEN):
        _a_stopped_run(tmp_path, monkeypatch)
    assert wfp.read_attempts(store, "w1", "s1") == 0

    seen = []
    for _ in range(wfp.DEFAULT_MAX_ATTEMPTS):
        _a_stopped_run(tmp_path, monkeypatch)
        seen.append(wfp.read_attempts(store, "w1", "s1"))

    assert seen == [1, 2, 3]
    assert "forgiven" in _ledger_entry(store)["detail"]
    # Nine starts in all, which at a handful of Tower starts a day is two to
    # three ordinary days -- not never.
    verdict = _verdict(store)
    assert verdict.owed is False
    assert verdict.exhausted is True


def test_a_finisher_that_is_always_killed_is_finally_retired(
    tmp_path, monkeypatch
):
    """And the world says so, instead of "Improving" for ever.

    The point of reaching the bound at all is `_retire`: the stage recorded
    `failed`, with the same explanatory detail and the same way back by hand
    that an ordinary exhausted session gets. A photographic build that did
    not happen is an honest thing for a world to report; a perpetual
    "Improving" that nothing will ever advance is not.
    """
    store = _world(tmp_path, stages=INTERRUPTED_SURFACE)
    argv = ("--max-forgiven", "1", "--max-attempts", "2")
    for _ in range(3):
        _a_stopped_run(tmp_path, monkeypatch, *argv)
    assert wfp.read_attempts(store, "w1", "s1") == 2

    # A start that is left alone: nothing is owed any more, so the only work
    # is writing down why.
    assert wfp.main(["--root", str(tmp_path), *argv]) == 0
    surface = store.read_session("w1", "s1").stages[STAGE_SURFACE]
    assert surface["state"] == STAGE_STATE_FAILED
    assert "world_surface.py" in (surface["detail"] or "")
    assert _verdict(store).owed is False


def test_a_ledger_written_before_forgiveness_was_bounded_still_parses(
    tmp_path, monkeypatch
):
    """OLD LEDGERS ARE NOT A SPECIAL CASE, they are a zero.

    `forgiven` was added to a file that already sits beside every world this
    tool has touched, and those entries carry `attempts` and `detail` and
    nothing else. A session nobody has forgiven yet has been forgiven zero
    times, which is what the missing field would have said -- so the old
    entry parses, its attempt count survives, and the first stop after the
    upgrade behaves exactly as it would have on a new ledger.
    """
    store = _world(tmp_path, stages=INTERRUPTED_SURFACE)
    path = store.world_dir("w1") / wfp.ATTEMPTS_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema": 1, "sessions": {
            "s1": {"attempts": 1, "detail": "finishing the surface stage"},
            # A second old entry, forgiven DIRECTLY below. In the ordinary
            # flow `record_attempt` runs first and normalises the entry, so
            # the stop watcher never meets a missing field -- and
            # `forgive_attempt` may not raise, so if it ever did meet one the
            # cost would be a forgiveness silently not written rather than a
            # traceback. Asked of it here where the swallow cannot hide it.
            "s0": {"attempts": 2, "detail": "finishing the surface stage"},
        }}),
        encoding="utf-8",
    )

    assert wfp.read_attempts(store, "w1", "s1") == 1
    assert wfp.read_forgiven(store, "w1", "s1") == 0
    assert _verdict(store).owed is True

    wfp.forgive_attempt(store, "w1", "s0", detail="stopped (stdin-closed)")
    assert wfp.read_attempts(store, "w1", "s0") == 1
    assert wfp.read_forgiven(store, "w1", "s0") == 1

    _a_stopped_run(tmp_path, monkeypatch)
    # The attempt this run counted was given back, and the old count is
    # untouched underneath it.
    assert wfp.read_attempts(store, "w1", "s1") == 1
    assert wfp.read_forgiven(store, "w1", "s1") == 1


# -- the CLI ----------------------------------------------------------


@pytest.fixture
def stage_runner(monkeypatch):
    """`final_surface_stages`, replaced by something that records the call.

    The real one is six to sixteen minutes of GPU. What this file is pinning
    is WHICH sessions reach it, under what lock, and what is written down --
    not the reconstruction, which `test_world_builder_surface_pipeline.py`
    already owns.
    """
    calls = []

    def fake(store, world_id, session_id, **kwargs):
        calls.append({"world": world_id, "session": session_id, **kwargs})
        # The lock must be HELD while the stages run, or another writer can
        # start underneath them.
        holder = store.lock_holder(world_id)
        calls[-1]["lock_pid"] = None if holder is None else holder["pid"]
        kwargs["record"](STAGE_SURFACE, state=STAGE_STATE_OK, detail=None)
        if kwargs.get("appearance"):
            kwargs["record"](STAGE_APPEARANCE, state=STAGE_STATE_OK, detail=None)
        return {"surface": {"attempted": True, "state": STAGE_STATE_OK}}

    monkeypatch.setattr(wfp, "final_surface_stages", fake)
    return calls


def test_the_cli_finishes_an_owed_session_under_the_world_writer_lock(
    tmp_path, stage_runner
):
    store = _world(tmp_path, stages=INTERRUPTED_SURFACE)
    assert wfp.main(["--root", str(tmp_path), "--format", "json"]) == 0

    assert len(stage_runner) == 1
    assert stage_runner[0]["world"] == "w1"
    assert stage_runner[0]["session"] == "s1"
    # Held by US while it ran...
    assert stage_runner[0]["lock_pid"] == os.getpid()
    # ...and released afterwards, because the phone reads the world.
    assert store.lock_holder("w1") is None
    assert store.read_session("w1", "s1").stages[STAGE_SURFACE]["state"] == (
        STAGE_STATE_OK
    )


def test_the_cli_leaves_a_historical_world_completely_alone(tmp_path, stage_runner):
    """No stage record, no work, no lock taken, nothing written."""
    store = _world(tmp_path, stages=None)
    before = store.session_path("w1", "s1").read_bytes()
    assert wfp.main(["--root", str(tmp_path), "--format", "json"]) == 0
    assert stage_runner == []
    assert store.session_path("w1", "s1").read_bytes() == before
    assert not store.lock_path("w1").exists()


def test_running_the_cli_twice_finishes_the_work_once(tmp_path, stage_runner):
    """Idempotent, because a Tower restarts and this runs at every start."""
    _world(tmp_path, stages=INTERRUPTED_SURFACE)
    assert wfp.main(["--root", str(tmp_path)]) == 0
    assert wfp.main(["--root", str(tmp_path)]) == 0
    assert len(stage_runner) == 1


def test_the_cli_finishes_one_world_at_a_time(tmp_path, stage_runner):
    """Bounded. Two owed worlds are two boots, not twelve minutes of GPU in
    one -- and a walk that starts in between must find the machine free."""
    _world(tmp_path, world_id="w1", stages=INTERRUPTED_SURFACE)
    _world(tmp_path, world_id="w2", session_id="s2", stages=INTERRUPTED_SURFACE)
    assert wfp.main(["--root", str(tmp_path)]) == 0
    assert len(stage_runner) == 1


def test_a_stop_asked_for_before_the_work_starts_skips_it(tmp_path, stage_runner):
    """Stoppable the way the builder is. Nothing is in flight to lose: the
    stage record already carries the `running` this run would leave."""
    store = _world(tmp_path, stages=INTERRUPTED_SURFACE)
    stop = StopRequest()
    stop.request(StopRequest.SOFT, "stdin-closed")
    assert wfp.main(["--root", str(tmp_path)], stop_request=stop) == 0
    assert stage_runner == []
    # And it cost the world nothing: no lock, and no attempt.
    assert wfp.read_attempts(store, "w1", "s1") == 0
    assert not store.lock_path("w1").exists()


def test_the_dry_run_reports_what_is_owed_and_touches_nothing(
    tmp_path, stage_runner, capsys
):
    store = _world(tmp_path, stages=INTERRUPTED_SURFACE)
    assert wfp.main(["--root", str(tmp_path), "--dry-run", "--format", "json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert [(row["world_id"], row["session_id"]) for row in report["owed"]] == [
        ("w1", "s1")
    ]
    assert stage_runner == []
    assert not store.lock_path("w1").exists()


def test_the_report_it_prints_at_every_boot_is_counts_not_sentences(
    tmp_path, stage_runner, capsys
):
    """This child inherits the Tower's stdout and runs at every start, over a
    root with seventy sessions on the machine it was written for. A sentence
    per skipped session is five hundred lines of console in front of the one
    line that matters; the sentences live behind `--dry-run`.
    """
    for index in range(12):
        _world(tmp_path, world_id=f"w{index}", session_id=f"s{index}", stages=None)
    assert wfp.main(["--root", str(tmp_path), "--format", "json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["skipped"] == {"no-stage-record": 12}
    assert "skipped_detail" not in report


def test_the_appearance_is_run_only_when_it_is_asked_for(tmp_path, stage_runner):
    """The same Settings object decides for the builder and for this, through
    the same flag on both argvs. There is no second default to drift."""
    _world(tmp_path, stages=INTERRUPTED_SURFACE)
    assert wfp.main(["--root", str(tmp_path), "--no-appearance"]) == 0
    assert stage_runner[0]["appearance"] is False


def test_the_call_into_the_builders_stage_path_still_binds(tmp_path):
    """The tests above replace `final_surface_stages`, so nothing else here
    would notice the day the real one's signature moves. This binds the call
    against the REAL function without running it.

    It is the whole safety of reusing the builder's path by import: the
    parameters, the ordering and the stop behaviour are the builder's, and a
    silent drift into a copy is exactly what that was meant to prevent.
    """
    import inspect

    from scripts.world_build_session import final_surface_stages

    inspect.signature(final_surface_stages).bind(
        object(), "w1", "s1",
        solved=True,
        appearance=True,
        prune_depth_work=True,
        should_stop=lambda: False,
        stop_source=lambda: None,
        record=lambda *a, **kw: None,
    )


def test_a_stop_asked_for_mid_run_leaves_the_world_lock_released(
    tmp_path, monkeypatch
):
    """A stage that raises must not take the writer lock with it.

    The lock is per world and `acquire_writer_lock` refuses a live holder
    rather than forcing it, so a lock this tool leaked would refuse the
    wearer's NEXT walk of that world -- for as long as this process lived.
    """
    store = _world(tmp_path, stages=INTERRUPTED_SURFACE)

    def explode(*args, **kwargs):
        raise RuntimeError("the depth network fell over")

    monkeypatch.setattr(wfp, "final_surface_stages", explode)
    assert wfp.main(["--root", str(tmp_path), "--format", "json"]) == 1
    assert store.lock_holder("w1") is None


def test_the_cli_root_flag_routes_through_the_artifact_root_guard(tmp_path):
    """`tests/test_artifact_paths.py` scans for this too; this is the
    behavioural half -- a root that would land at the drive root is refused
    rather than silently written to."""
    with pytest.raises(SystemExit):
        wfp.main(["--root", "C:\\" if os.name == "nt" else "/"])


# -- the Tower spawns it ----------------------------------------------


def _finish_argv(**overrides):
    from dataclasses import replace

    from tower import main as tower_main
    from tower.config import get_settings

    settings = replace(get_settings(), world_root="data/world_builder", **overrides)
    spec = tower_main._world_finish_spec(settings)
    return None if spec is None else spec.argv


def test_the_tower_spawns_the_finisher_as_an_argv_never_an_import():
    """`main.py:117-133` explains why this is load-bearing: a command line
    inherits nothing, and `test_shared_code_does_not_import_a_cartridge`
    forbids the web process from importing the cartridge at all."""
    argv = _finish_argv()
    assert argv is not None
    assert argv[0] == sys.executable
    assert argv[1].endswith("world_finish_pending.py")
    assert "--root" in argv
    assert "--stop-on-stdin-close" in argv
    assert "--max-worlds" in argv


def test_the_finisher_can_be_switched_off():
    assert _finish_argv(world_finish_pending=False) is None


def test_the_finisher_needs_the_stages_it_would_finish():
    """It builds a surface and an appearance. Neither is possible without the
    solve they are anchored to, and asking for one anyway is a configuration
    mistake this refuses once rather than per session."""
    assert _finish_argv(world_surface=False) is None
    assert _finish_argv(world_solve=False) is None
    assert "--no-appearance" in _finish_argv(world_appearance=False)


def test_a_capture_opening_stops_the_finisher(tmp_path):
    """THE SAFETY GATE. A walk needs the GPU and the frame path must never be
    disturbed, so the first sign of one ends this chore immediately.

    The finisher is started once, at Tower startup, when nothing is streaming
    and every cartridge session is stopped. `capture_opened` and `attach` are
    the two funnels through which anything begins following a capture, so
    stopping on both means a walk can never find this running.
    """
    from tower import main as tower_main

    stopped = []

    class _Chore:
        def stop(self, reason, **kwargs):
            stopped.append(reason)

    supervisor = tower_main._build_capture_worker_supervisor(
        tower_main.get_settings(), {}, yields_the_gpu_to=_Chore()
    )
    supervisor.capture_opened("cap-1", tmp_path)
    assert stopped and "capture" in stopped[0]

    stopped.clear()
    supervisor.attach("anything", "cap-1", tmp_path)
    assert stopped


def test_a_stream_with_no_capture_recorder_still_stops_the_finisher(
    tmp_path, monkeypatch
):
    """THE HOLE IN THE CAPTURE FUNNELS, and it is a supported configuration.

    `TOWER_CAPTURE_ROOT` unset means no capture id is ever minted, so
    `capture_opened` never fires and `attach` returns early -- while frames
    stream and the live cartridges are told the stream opened and take the
    GPU. The chore would have kept it, and a world writer lock, for the rest
    of that stream. A STREAM exists whenever a phone is sending frames; a
    CAPTURE exists only when a recorder is armed, and only the first of those
    is the event that matters here.
    """
    import base64
    import io

    from fastapi.testclient import TestClient
    from PIL import Image

    from tower.main import create_app

    monkeypatch.delenv("TOWER_CAPTURE_ROOT", raising=False)
    monkeypatch.setenv("TOWER_WORLD_ROOT", str(tmp_path))

    stopped = []

    class _Chore:
        def start(self):
            return True

        def stop(self, reason, **kwargs):
            stopped.append(reason)

    app = create_app()
    assert app.state.capture_root is None, "this test needs the recorder OFF"
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

    assert stopped, "a stream opened and the chore kept the GPU"
    assert "stream" in stopped[0]


def test_a_second_stop_waits_for_the_child_to_actually_be_gone(tmp_path):
    """The claim "a builder is spawned strictly after the finisher is dead"
    has to be true for BOTH callers.

    `capture_opened` comes off ws.py's threadpool and `attach` off the HTTP
    one, and the phone streaming while the wearer presses Start interleaves
    them. Measured on the first version: the second caller returned in 0.05 s
    with the child still alive and running a GPU op, while the first was
    still inside its grace -- so the guarantee held for one caller and not
    the other.
    """
    from tower import main as tower_main
    from tower.capture_workers import WorkerSpec

    chore = tower_main._BackgroundChore(
        WorkerSpec(
            # A child that ignores the closed pipe, which is exactly what a
            # finisher inside a depth pass looks like from out here.
            argv=(sys.executable, "-c", "import time; time.sleep(60)"),
            name="probe",
            stop_via_stdin=True,
            stop_grace_seconds=0.5,
        )
    )
    assert chore.start() is True
    process = chore._process
    returned = {}

    def stopper(tag):
        def run():
            chore.stop(f"{tag}")
            returned[tag] = process.poll() is not None

        return run

    first = threading.Thread(target=stopper("first"))
    first.start()
    time.sleep(0.05)
    second = threading.Thread(target=stopper("second"))
    second.start()
    first.join(timeout=30)
    second.join(timeout=30)

    assert returned == {"first": True, "second": True}
