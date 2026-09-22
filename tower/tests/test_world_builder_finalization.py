"""The finalization record, the lock that names a process, and the engine
calls the builder uses to keep them truthful.

Written after the 2026-09-06 physical walk (world fcbca9e9), where a builder
died mid-walk and the only facts on disk were a lock with a dead pid, a
session record with `ended_at: null`, and 463 keyframes of perfectly good
geometry. Nothing could say "interrupted" because nothing recorded the
difference between a builder that was killed, one that finished, and one
that was still finishing.
"""

import json
import os
import time

import pytest

from tests import synthetic_scene as ss
from tower.world_builder.engine import WorldBuilderEngine
from tower.world_builder.records import (
    FINALIZATION_COMPLETE,
    FINALIZATION_INTERRUPTED,
    FINALIZATION_PENDING,
    STAGE_APPEARANCE,
    STAGE_DENSE,
    STAGE_STATE_FAILED,
    STAGE_STATE_OK,
    STAGE_STATE_RUNNING,
    STAGE_STATE_STOPPED,
    STAGE_STATE_UNAVAILABLE,
    STAGE_STATES,
    STAGE_SURFACE,
    Session,
    session_from_json_dict,
)
from tower.world_builder.store import WorldStore


# -- the record -------------------------------------------------------


def test_a_session_record_round_trips_its_finalization():
    session = Session(
        session_id="s", world_id="w", started_at=1.0, ended_at=2.0, end_reason="stop",
        finalization={
            "state": FINALIZATION_PENDING,
            "final_solve": "pending",
            "started_at": 2.0,
            "updated_at": 2.0,
            "detail": None,
        },
    )
    data = session.to_json_dict()
    assert data["finalization"]["state"] == FINALIZATION_PENDING
    assert session_from_json_dict(data).finalization == session.finalization


def test_a_record_written_before_finalization_existed_still_parses():
    """Every session on every Tower before today has no `finalization` key.
    Absent means "never recorded", which the readers treat as None."""
    data = Session(session_id="s", world_id="w", started_at=1.0).to_json_dict()
    del data["finalization"]
    assert session_from_json_dict(data).finalization is None


# -- the lock ---------------------------------------------------------


def test_the_lock_names_the_process_not_just_its_pid(tmp_path):
    store = WorldStore(tmp_path)
    store.acquire_writer_lock("w")
    holder = json.loads(store.lock_path("w").read_text(encoding="utf-8"))
    assert holder["pid"] == os.getpid()
    assert isinstance(holder["created_at"], float)
    answer = store.lock_holder("w")
    assert answer["pid"] == os.getpid()
    assert answer["alive"] is True
    assert answer["unreadable"] is False


def test_a_reused_pid_does_not_resurrect_a_dead_builder(tmp_path):
    """psutil.pid_exists(pid) says a RECYCLED pid is alive. The create time
    tells the two processes apart."""
    store = WorldStore(tmp_path)
    store.lock_path("w").parent.mkdir(parents=True)
    store.lock_path("w").write_text(
        json.dumps({"pid": os.getpid(), "created_at": 12345.0}), encoding="utf-8"
    )
    assert store.lock_holder("w")["alive"] is False
    # A lock written by an older builder carries only a pid; liveness then
    # falls back to the pid alone, as it always did.
    store.lock_path("w").write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
    assert store.lock_holder("w")["alive"] is True


def test_lock_holder_answers_for_every_lock_shape(tmp_path):
    import psutil

    store = WorldStore(tmp_path)
    assert store.lock_holder("w") is None
    store.lock_path("w").parent.mkdir(parents=True)
    store.lock_path("w").write_text("{not json", encoding="utf-8")
    assert store.lock_holder("w") is None
    store.lock_path("w").write_text(json.dumps({"pid": "x"}), encoding="utf-8")
    assert store.lock_holder("w") == {"pid": None, "alive": False, "unreadable": True}
    dead = next(pid for pid in range(100_000, 200_000) if not psutil.pid_exists(pid))
    store.lock_path("w").write_text(json.dumps({"pid": dead}), encoding="utf-8")
    assert store.lock_holder("w") == {"pid": dead, "alive": False, "unreadable": False}


def test_a_stale_lock_with_a_reused_pid_is_reclaimed(tmp_path):
    """`acquire_writer_lock` refuses a live holder. A dead holder whose pid
    happens to be running again must not block the next session forever."""
    store = WorldStore(tmp_path)
    store.lock_path("w").parent.mkdir(parents=True)
    store.lock_path("w").write_text(
        json.dumps({"pid": os.getppid(), "created_at": 1.0}), encoding="utf-8"
    )
    store.acquire_writer_lock("w")
    assert store.lock_holder("w")["pid"] == os.getpid()


# -- the engine -------------------------------------------------------


def _engine_with_session(tmp_path):
    store = WorldStore(tmp_path)
    engine = WorldBuilderEngine(store)
    world_id = engine.create_world("Room")
    session_id = engine.start_session(world_id, frame_source="synthetic")
    return store, engine, world_id, session_id


def test_stop_session_can_hold_the_lock_through_finalization(tmp_path):
    store, engine, world_id, session_id = _engine_with_session(tmp_path)
    engine.stop_session(hold_lock=True)
    assert store.lock_path(world_id).exists(), "the lock must survive stop_session"
    session = store.read_session(world_id, session_id)
    assert session.ended_at is not None
    assert session.finalization["state"] == FINALIZATION_PENDING
    assert session.finalization["final_solve"] == "pending"
    engine.release_world(world_id)
    assert not store.lock_path(world_id).exists()


def test_stop_session_without_hold_lock_is_unchanged(tmp_path):
    store, engine, world_id, session_id = _engine_with_session(tmp_path)
    engine.stop_session()
    assert not store.lock_path(world_id).exists()
    assert store.read_session(world_id, session_id).finalization is None


def test_mark_finalization_rewrites_the_record_without_touching_counts(tmp_path):
    store, engine, world_id, session_id = _engine_with_session(tmp_path)
    engine.stop_session(hold_lock=True)
    before = store.read_session(world_id, session_id)
    engine.mark_finalization(
        world_id, session_id, state=FINALIZATION_COMPLETE, final_solve="solved",
    )
    after = store.read_session(world_id, session_id)
    assert after.finalization["state"] == FINALIZATION_COMPLETE
    assert after.finalization["final_solve"] == "solved"
    assert after.finalization["updated_at"] >= before.finalization["updated_at"]
    assert after.finalization["started_at"] == before.finalization["started_at"]
    assert after.keyframes_accepted == before.keyframes_accepted
    assert after.ended_at == before.ended_at

    engine.mark_finalization(
        world_id, session_id, state=FINALIZATION_INTERRUPTED,
        final_solve="skipped", detail="stop requested while observing",
    )
    record = store.read_session(world_id, session_id).finalization
    assert record["state"] == FINALIZATION_INTERRUPTED
    assert record["detail"] == "stop requested while observing"


def test_mark_finalization_on_a_record_that_never_stopped_still_writes(tmp_path):
    """The error path: an exception before stop_session. The builder stops
    the session with `error` and marks the finalization interrupted; the
    record must accept that even though nothing set it to pending first."""
    store, engine, world_id, session_id = _engine_with_session(tmp_path)
    engine.stop_session(reason="error", hold_lock=True)
    engine.mark_finalization(
        world_id, session_id, state=FINALIZATION_INTERRUPTED,
        final_solve=None, detail="RuntimeError: boom",
    )
    record = store.read_session(world_id, session_id)
    assert record.end_reason == "error"
    assert record.finalization["state"] == FINALIZATION_INTERRUPTED
    assert record.finalization["detail"] == "RuntimeError: boom"


# -- the post-finalization stages -------------------------------------
#
# `finalization` answers "did the builder finish?", and until 2026-09-22 the
# builder answered YES several minutes before the thing the wearer walked for
# existed. `mark_finalization(complete)` and `release_world()` both run in
# `world_build_session.main`'s `finally`, which is deliberately BEFORE the
# surface, the appearance and the dense stages -- the lock is released so the
# phone can read the world while they run. That sequencing is right; the
# record was not. A world could be `complete / solved` with no photographic
# representation, and nothing on disk said whether one was attempted, was
# skipped, or raised. These pin the second record that says so.


def test_a_session_record_round_trips_its_post_finalization_stages():
    stages = {
        STAGE_SURFACE: {
            "attempted": True,
            "state": STAGE_STATE_OK,
            "started_at": 3.0,
            "updated_at": 9.0,
            "detail": None,
        },
        STAGE_APPEARANCE: {
            "attempted": True,
            "state": STAGE_STATE_FAILED,
            "started_at": 9.0,
            "updated_at": 11.0,
            "detail": "RuntimeError: boom",
        },
    }
    session = Session(
        session_id="s", world_id="w", started_at=1.0, ended_at=2.0, end_reason="stop",
        stages=stages,
    )
    data = session.to_json_dict()
    assert data["stages"][STAGE_SURFACE]["state"] == STAGE_STATE_OK
    assert session_from_json_dict(data).stages == stages
    # A copy per stage, not the caller's nested dicts: the record is frozen
    # and must not change under whoever still holds what it was handed.
    data["stages"][STAGE_SURFACE]["state"] = "tampered"
    assert session.stages[STAGE_SURFACE]["state"] == STAGE_STATE_OK


def test_a_record_written_before_the_stages_existed_still_parses():
    """Exactly as `finalization` did it: every session written before today
    has no `stages` key, and absent means "never recorded", not "skipped"."""
    data = Session(session_id="s", world_id="w", started_at=1.0).to_json_dict()
    del data["stages"]
    assert session_from_json_dict(data).stages is None
    # And a key that is not an object is not a stage record either.
    data["stages"] = "complete"
    assert session_from_json_dict(data).stages is None


def test_the_stage_vocabulary_is_the_surface_pipeline_vocabulary():
    """One vocabulary, one definition. `surface_pipeline` writes these words
    into `status.json`, `dense_pipeline` imports them from there, and the
    session record now stores them -- so a `SurfaceResult.state` can be
    recorded verbatim, with no translation table to fall out of step. Two
    sets of five strings that happened to agree on the day they were written
    is the silent-divergence failure this module already refuses to repeat
    for `Confidence`."""
    from tower.world_builder import surface_pipeline as SP

    assert (SP.STATE_OK, SP.STATE_RUNNING, SP.STATE_FAILED, SP.STATE_STOPPED,
            SP.STATE_UNAVAILABLE) == (
        STAGE_STATE_OK, STAGE_STATE_RUNNING, STAGE_STATE_FAILED,
        STAGE_STATE_STOPPED, STAGE_STATE_UNAVAILABLE,
    )
    assert set(STAGE_STATES) == {
        SP.STATE_OK, SP.STATE_RUNNING, SP.STATE_FAILED, SP.STATE_STOPPED,
        SP.STATE_UNAVAILABLE,
    }


def test_mark_stage_records_running_then_the_outcome(tmp_path):
    """`running` is written BEFORE the stage starts, on purpose: the Job
    Object kills this process tree on a 30-second grace, and a builder killed
    inside a six-minute surface gets no chance to record anything afterwards.
    A stage left saying `running` by a process that is gone is the truth."""
    store, engine, world_id, session_id = _engine_with_session(tmp_path)
    engine.stop_session(hold_lock=True)
    engine.mark_finalization(
        world_id, session_id, state=FINALIZATION_COMPLETE, final_solve="solved",
    )
    before = store.read_session(world_id, session_id)

    engine.mark_stage(world_id, session_id, STAGE_SURFACE, state=STAGE_STATE_RUNNING)
    running = store.read_session(world_id, session_id).stages[STAGE_SURFACE]
    assert running == {
        "attempted": True,
        "state": STAGE_STATE_RUNNING,
        "started_at": running["started_at"],
        "updated_at": running["updated_at"],
        "detail": None,
    }

    engine.mark_stage(
        world_id, session_id, STAGE_SURFACE, state=STAGE_STATE_OK, detail=None,
    )
    after = store.read_session(world_id, session_id)
    done = after.stages[STAGE_SURFACE]
    assert done["state"] == STAGE_STATE_OK
    # `started_at` survives the second write, so "how long did the surface
    # take" is answerable from the record alone -- the same guarantee
    # `mark_finalization` gives finalization.
    assert done["started_at"] == running["started_at"]
    assert done["updated_at"] >= running["updated_at"]
    # And nothing else on the record moved.
    assert after.finalization == before.finalization
    assert after.ended_at == before.ended_at
    assert after.keyframes_accepted == before.keyframes_accepted


def test_mark_stage_leaves_the_other_stages_alone(tmp_path):
    store, engine, world_id, session_id = _engine_with_session(tmp_path)
    engine.stop_session(hold_lock=True)
    engine.mark_stage(world_id, session_id, STAGE_SURFACE, state=STAGE_STATE_OK)
    engine.mark_stage(
        world_id, session_id, STAGE_APPEARANCE, state=STAGE_STATE_FAILED,
        detail="ValueError: no redactor",
    )
    engine.mark_stage(
        world_id, session_id, STAGE_DENSE, state=STAGE_STATE_UNAVAILABLE,
        attempted=False, detail="not requested",
    )
    stages = store.read_session(world_id, session_id).stages
    assert stages[STAGE_SURFACE]["state"] == STAGE_STATE_OK
    assert stages[STAGE_APPEARANCE]["detail"] == "ValueError: no redactor"
    assert stages[STAGE_DENSE]["attempted"] is False
    assert set(stages) == {STAGE_SURFACE, STAGE_APPEARANCE, STAGE_DENSE}


def test_mark_stage_refuses_a_word_nobody_switches_on(tmp_path):
    """Closed sets, like FINALIZATION_STATES: consumers switch on these."""
    store, engine, world_id, session_id = _engine_with_session(tmp_path)
    engine.stop_session(hold_lock=True)
    with pytest.raises(ValueError):
        engine.mark_stage(world_id, session_id, "surfacey", state=STAGE_STATE_OK)
    with pytest.raises(ValueError):
        engine.mark_stage(world_id, session_id, STAGE_SURFACE, state="finished")
    assert store.read_session(world_id, session_id).stages is None


def test_mark_stage_works_after_the_world_lock_is_released(tmp_path):
    """THE WHOLE POINT. The surface and appearance stages run unlocked, by
    design -- `world_builder_render.py` documents the release as what lets the
    phone read the world while they build -- so the record of them is written
    unlocked too. A version of this that needed the lock would have to hold it
    across six minutes of surface work and would take the live world away from
    the wearer, which is the re-architecture this fix exists not to do."""
    store, engine, world_id, session_id = _engine_with_session(tmp_path)
    engine.stop_session(hold_lock=True)
    engine.mark_finalization(
        world_id, session_id, state=FINALIZATION_COMPLETE, final_solve="solved",
    )
    engine.release_world(world_id)
    assert not store.lock_path(world_id).exists()
    engine.mark_stage(world_id, session_id, STAGE_SURFACE, state=STAGE_STATE_RUNNING)
    stages = store.read_session(world_id, session_id).stages
    assert stages[STAGE_SURFACE]["state"] == STAGE_STATE_RUNNING


def test_a_builder_that_was_not_asked_for_a_surface_says_so_on_the_record(tmp_path):
    """End to end, cold start, as a user runs it: a synthetic walk with no
    `--surface` must leave a record saying the photographic stages were never
    requested -- a different fact from "nobody has looked yet", and the one an
    operator needs to tell a sparse world apart from a broken one."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "scripts/world_build_session.py", "--synthetic",
         "--synthetic-frames", "8", "--root", str(tmp_path), "--format", "json"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    store = WorldStore(tmp_path)
    stages = store.read_session(report["world_id"], report["session_id"]).stages
    assert set(stages) == {STAGE_SURFACE, STAGE_APPEARANCE, STAGE_DENSE}
    for stage in (STAGE_SURFACE, STAGE_APPEARANCE, STAGE_DENSE):
        assert stages[stage]["attempted"] is False
        assert stages[stage]["state"] == STAGE_STATE_UNAVAILABLE
        assert "not requested" in stages[stage]["detail"]
