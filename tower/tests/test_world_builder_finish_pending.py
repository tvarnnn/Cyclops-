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
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import world_finish_pending as wfp  # noqa: E402
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


def test_a_session_with_no_stages_record_at_all_is_not_owed(tmp_path):
    """THE HISTORICAL-WORLD GUARD, and the most expensive test to get wrong.

    Absent is not a state in the vocabulary. It means "a Tower that never
    recorded this", which is every one of the 165 worlds on this machine
    that predate 2026-09-22. Treating absence as "interrupted" would put
    tens of hours of GPU work into a queue nobody asked for, on worlds that
    are finished and fine.
    """
    store = _world(tmp_path, stages=None)
    verdict = _verdict(store)
    assert verdict.owed is False
    assert "never recorded" in verdict.reason


def test_an_empty_stages_object_is_also_not_owed(tmp_path):
    """`{}` says as little as absence does, and must be read as carefully."""
    store = _world(tmp_path, stages={})
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
    _world(tmp_path, stages=INTERRUPTED_SURFACE)
    assert (
        wfp.main(["--root", str(tmp_path)], should_stop=lambda: True) == 0
    )
    assert stage_runner == []


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
