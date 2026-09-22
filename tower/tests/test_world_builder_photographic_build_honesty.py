"""The gap between "the lock was released" and "the world is finished".

A wearer walked a bedroom on 2026-09-22, pressed Stop, and within ninety
seconds the phone said **Saved**. The photographic reconstruction had not
been built and was six to sixteen minutes away. They reasonably concluded
it was done, looked at the sparse points, and shut the Tower down -- which
destroyed the build that was running.

The ordering that produces this is DELIBERATE and must not be changed:
`scripts/world_build_session.py` marks finalization complete and releases
the world writer lock in its `finally`, and only THEN runs the surface,
appearance and dense stages, so the phone can read the world while the
better picture is being made (`world_builder_render.py`'s own docstring
names this gap). What was wrong is not the ordering; it is that
`_lifecycle` inferred "a build is running" from the LOCK alone, and the
lock is exactly the thing that ordering gives up.

So these tests pin one fact: while a photographic stage is *verifiably*
running for this session -- its `status.json` says `running` and the pid
that wrote it is alive -- the status says `finalizing` /
`build_in_progress: true`, which the installed iOS build renders as
"Improving ... it is worth waiting for Saved". And they pin the four ways
that must NOT happen: a stale status from a dead process, a stage that has
already published its manifest, a session with no stage status at all, and
the two arms that read the lock directly.
"""

import json
import os
import time

import psutil
import pytest

from tower.results.world_builder import (
    LIFECYCLE_FINALIZING,
    LIFECYCLE_READY,
    LIFECYCLE_RECEIVING,
    MODEL_STATE_FINALIZING,
    MODEL_STATE_FINALIZED,
    WorldBuilderStatusProducer,
)
from tower.world_builder.events import WorldEvent
from tower.world_builder.records import (
    FINAL_SOLVE_SOLVED,
    FINALIZATION_COMPLETE,
    Session,
)


def _dead_pid() -> int:
    return next(pid for pid in range(100_000, 200_000) if not psutil.pid_exists(pid))


def _lifecycle_of(store, world_id, session_id) -> dict:
    payload = (
        WorldBuilderStatusProducer(store.root, time.time)
        .snapshot(world_id, session_id)
        .payload
    )
    return payload["lifecycle"], payload


def _write_status(store, world_id, session_id, stage, **fields) -> None:
    root = store.world_dir(world_id) / stage / session_id
    root.mkdir(parents=True, exist_ok=True)
    (root / "status.json").write_text(json.dumps(fields), encoding="utf-8")


def _finalized(store, world_id, session_id) -> None:
    """The record the shipped builder leaves just before it lets the lock go."""
    store.append_event(
        world_id, session_id,
        WorldEvent(event_id=1, kind="session_stopped", at=2.0, payload={}),
    )
    session = store.read_session(world_id, session_id)
    store.write_session(
        Session(
            session_id=session_id,
            world_id=world_id,
            started_at=session.started_at,
            ended_at=session.ended_at,
            end_reason="stop",
            finalization={
                "state": FINALIZATION_COMPLETE,
                "final_solve": FINAL_SOLVE_SOLVED,
                "started_at": 2.0,
                "updated_at": 3.0,
                "detail": None,
            },
        )
    )


@pytest.fixture
def finalized_world(derived_world):
    """The exact disk state ninety seconds after Stop: finalization complete,
    the writer lock released, the derived tree there -- and the surface stage
    not started yet."""
    store, world_id, session_id = derived_world
    _finalized(store, world_id, session_id)
    return store, world_id, session_id


# -- the world is still being built -----------------------------------------


def test_a_running_surface_after_the_lock_is_released_is_not_saved(finalized_world):
    store, world_id, session_id = finalized_world
    before, _ = _lifecycle_of(store, world_id, session_id)
    assert before["state"] == LIFECYCLE_READY, (
        "the fixture must start from the state that produced the bug"
    )

    _write_status(store, world_id, session_id, "surface",
                  state="running", pid=os.getpid(), updated_at=time.time())

    lifecycle, payload = _lifecycle_of(store, world_id, session_id)
    assert lifecycle["state"] == LIFECYCLE_FINALIZING
    assert lifecycle["build_in_progress"] is True
    assert lifecycle["build_in_progress_unavailable_reason"] is None
    assert payload["model_state"] == MODEL_STATE_FINALIZING, (
        "the phone reads model_state; 'finalized' here is the word that said Saved"
    )


def test_the_evidence_names_the_stage_and_the_pid_and_not_the_lock(finalized_world):
    """The lock is NOT held here. Evidence that says it is would be a second
    untruth told to cover the first."""
    store, world_id, session_id = finalized_world
    _write_status(store, world_id, session_id, "surface",
                  state="running", pid=os.getpid(), updated_at=time.time())

    lifecycle, _ = _lifecycle_of(store, world_id, session_id)
    evidence = lifecycle["evidence"]
    assert "surface" in evidence
    assert "status.json" in evidence
    assert f"pid {os.getpid()}" in evidence
    # It may SAY the lock was released -- that is true and worth saying. What
    # it must never do is claim the lock is held, which is the lock-held arm's
    # sentence and the one thing that is false here.
    assert "holds the writer lock" not in evidence, evidence
    assert "released" in evidence, evidence
    assert "photographic" in (lifecycle["reason"] or "")


def test_the_finalization_record_travels_unchanged(finalized_world):
    """`finalization` describes the SOLVE and it really did complete. The
    photographic stage is a different fact and must not overwrite it."""
    store, world_id, session_id = finalized_world
    _write_status(store, world_id, session_id, "surface",
                  state="running", pid=os.getpid(), updated_at=time.time())

    lifecycle, _ = _lifecycle_of(store, world_id, session_id)
    assert lifecycle["finalization"]["state"] == FINALIZATION_COMPLETE
    assert lifecycle["finalization"]["final_solve"] == FINAL_SOLVE_SOLVED


@pytest.mark.parametrize("stage", ["surface", "appearance", "dense"])
def test_every_photographic_stage_counts(finalized_world, stage):
    store, world_id, session_id = finalized_world
    _write_status(store, world_id, session_id, stage,
                  state="running", pid=os.getpid(), updated_at=time.time())

    lifecycle, _ = _lifecycle_of(store, world_id, session_id)
    assert lifecycle["state"] == LIFECYCLE_FINALIZING, stage
    assert lifecycle["build_in_progress"] is True, stage
    assert stage in lifecycle["evidence"], lifecycle["evidence"]


# -- and the four ways it must not happen -----------------------------------


def test_a_status_left_behind_by_a_dead_process_is_not_a_build(finalized_world):
    """`status_is_stale` exists for this. A Tower killed mid-surface leaves
    `running` on disk forever, and a permanent "Improving" is the same lie
    pointing the other way."""
    store, world_id, session_id = finalized_world
    _write_status(store, world_id, session_id, "surface",
                  state="running", pid=_dead_pid(), updated_at=time.time())

    lifecycle, payload = _lifecycle_of(store, world_id, session_id)
    assert lifecycle["state"] == LIFECYCLE_READY
    assert lifecycle["build_in_progress"] is False
    assert payload["model_state"] == MODEL_STATE_FINALIZED


def test_a_stage_that_has_already_published_its_manifest_is_not_a_build(
    finalized_world,
):
    """Review 3's R5, inherited from `_stage_running`: a build writes its
    manifest, prunes, and only then `ok`. A poll in that gap must not claim
    the world is still changing."""
    store, world_id, session_id = finalized_world
    _write_status(store, world_id, session_id, "surface",
                  state="running", pid=os.getpid(), updated_at=time.time())
    root = store.world_dir(world_id) / "surface" / session_id
    status_ns = (root / "status.json").stat().st_mtime_ns
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"built_at": 1.0}), encoding="utf-8")
    os.utime(manifest, ns=(status_ns + 10**9, status_ns + 10**9))

    lifecycle, _ = _lifecycle_of(store, world_id, session_id)
    assert lifecycle["state"] == LIFECYCLE_READY


def test_a_world_with_no_stage_status_reports_what_it_always_did(finalized_world):
    store, world_id, session_id = finalized_world
    lifecycle, payload = _lifecycle_of(store, world_id, session_id)
    assert lifecycle["state"] == LIFECYCLE_READY
    assert lifecycle["build_in_progress"] is False
    assert lifecycle["reason"] is None
    assert payload["model_state"] == MODEL_STATE_FINALIZED


def test_a_finished_stage_status_is_not_a_build(finalized_world):
    store, world_id, session_id = finalized_world
    for stage in ("surface", "appearance", "dense"):
        _write_status(store, world_id, session_id, stage,
                      state="ok", pid=os.getpid(), updated_at=time.time())

    lifecycle, _ = _lifecycle_of(store, world_id, session_id)
    assert lifecycle["state"] == LIFECYCLE_READY


def test_the_status_poll_and_the_revision_poll_agree(finalized_world):
    """One predicate, two surfaces. `/render/revision` already answers `live`
    with `session_build_running`; the status page must not answer the same
    question differently, or a follower swaps in a world the status page is
    still telling the wearer to wait for."""
    from tower.results import world_builder_render as R

    store, world_id, session_id = finalized_world
    for stage_state, pid in (("running", os.getpid()),
                             ("running", _dead_pid()),
                             ("ok", os.getpid())):
        _write_status(store, world_id, session_id, "surface",
                      state=stage_state, pid=pid, updated_at=time.time())
        lifecycle, _ = _lifecycle_of(store, world_id, session_id)
        assert lifecycle["build_in_progress"] is R.session_build_running(
            store, world_id, session_id
        ), (stage_state, pid, lifecycle["evidence"])


# -- the two arms that read the lock directly are untouched ------------------


def test_the_receiving_arm_is_unchanged(tmp_path):
    from tests.result_channel_fixtures import start_live_world

    root = tmp_path / "worlds"
    world_id, session_id, engine = start_live_world(root, frames=6)
    try:
        store = engine._store
        # Even with a stage status lying around, an OPEN session is receiving.
        _write_status(store, world_id, session_id, "surface",
                      state="running", pid=os.getpid(), updated_at=time.time())
        lifecycle, _ = _lifecycle_of(store, world_id, session_id)
        assert lifecycle["state"] == LIFECYCLE_RECEIVING
        assert lifecycle["build_in_progress"] is False
        assert "writer lock" in lifecycle["evidence"]
    finally:
        engine.release_world(world_id)


def test_the_lock_held_finalizing_arm_still_cites_the_lock(tmp_path):
    from tests.result_channel_fixtures import start_live_world

    root = tmp_path / "worlds"
    world_id, session_id, engine = start_live_world(root, frames=6)
    try:
        engine.stop_session(hold_lock=True)
        store = engine._store
        _write_status(store, world_id, session_id, "surface",
                      state="running", pid=os.getpid(), updated_at=time.time())
        lifecycle, _ = _lifecycle_of(store, world_id, session_id)
        assert lifecycle["state"] == LIFECYCLE_FINALIZING
        assert lifecycle["build_in_progress"] is True
        assert "writer lock" in lifecycle["evidence"], (
            "the lock really is held here; the evidence must say the true thing"
        )
    finally:
        engine.release_world(world_id)
