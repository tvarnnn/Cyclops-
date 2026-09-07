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
