"""Every lifecycle the disk can now tell apart, on the wire, truthfully.

After the 2026-09-06 physical walk the Tower could say `receiving`, `failed`,
`stopped_unbuilt` and `ready` -- and the walk that produced 463 keyframes of
usable geometry was `failed`, because a builder that died mid-walk and a
builder that never built anything looked identical on disk. The builder
now keeps its lock through finalization and records how finalization went,
so the producer can say `finalizing` when a live process is finishing and
`interrupted` when one is not, while still reporting the geometry that
exists.

The `selection` block is new for a different bug: with nothing live, an
unpinned subscription was answered with the newest world on disk and the
phone drew it as the live world. The block says which of live, finalizing,
latest and pinned the answer is, so the client can decide instead of guess.
"""

import json
import os
import time

import psutil
import pytest

from tests.result_channel_fixtures import build_world, start_live_world
from tower.results.world_builder import (
    LIFECYCLE_FINALIZING,
    LIFECYCLE_INTERRUPTED,
    LIFECYCLE_READY,
    LIFECYCLE_RECEIVING,
    MODEL_STATE_FINALIZING,
    MODEL_STATE_FINALIZED,
    MODEL_STATE_INTERRUPTED,
    MODEL_STATE_RECEIVING,
    SELECTION_FINALIZING,
    SELECTION_LATEST,
    SELECTION_LIVE,
    SELECTION_NONE,
    SELECTION_PINNED,
    WorldBuilderStatusProducer,
)
from tower.results.world_builder_library import build_world_listing
from tower.world_builder.records import (
    FINAL_SOLVE_SKIPPED,
    FINAL_SOLVE_SOLVED,
    FINALIZATION_COMPLETE,
    FINALIZATION_INTERRUPTED,
)
from tower.world_builder.store import WorldStore


def _dead_pid() -> int:
    return next(pid for pid in range(100_000, 200_000) if not psutil.pid_exists(pid))


def _payload(root, world_id=None, session_id=None) -> dict:
    return WorldBuilderStatusProducer(root, time.time).snapshot(world_id, session_id).payload


def _write_dead_lock(store, world_id):
    store.lock_path(world_id).write_text(json.dumps({"pid": _dead_pid()}), encoding="utf-8")


# -- finalizing: a live process is finishing --------------------------------


def test_a_stopped_session_whose_builder_still_holds_the_lock_is_finalizing(tmp_path):
    root = tmp_path / "worlds"
    world_id, session_id, engine = start_live_world(root, frames=6)
    try:
        engine.stop_session(hold_lock=True)
        payload = _payload(root)
        lifecycle = payload["lifecycle"]
        assert lifecycle["state"] == LIFECYCLE_FINALIZING
        assert lifecycle["build_in_progress"] is True
        assert f"pid {os.getpid()}" in lifecycle["evidence"]
        assert lifecycle["finalization"]["state"] == "pending"
        assert payload["model_state"] == MODEL_STATE_FINALIZING
        assert payload["selection"]["mode"] == SELECTION_FINALIZING
        assert payload["selection"]["world_id"] == world_id
        assert payload["selection"]["session_id"] == session_id
    finally:
        engine.release_world(world_id)


def test_a_finalized_session_is_ready_and_says_how_the_final_solve_went(tmp_path):
    root = tmp_path / "worlds"
    world_id, session_id, engine = start_live_world(root, frames=6)
    engine.stop_session(hold_lock=True)
    engine.build(world_id, session_id)
    engine.mark_finalization(
        world_id, session_id, state=FINALIZATION_COMPLETE, final_solve=FINAL_SOLVE_SOLVED,
    )
    engine.release_world(world_id)
    payload = _payload(root)
    assert payload["lifecycle"]["state"] == LIFECYCLE_READY
    assert payload["lifecycle"]["finalization"]["final_solve"] == FINAL_SOLVE_SOLVED
    assert payload["model_state"] == MODEL_STATE_FINALIZED
    assert payload["selection"]["mode"] == SELECTION_LATEST


# -- interrupted: nobody is finishing, and the record says why --------------


def test_a_dead_builder_mid_walk_is_interrupted_and_its_geometry_is_still_reported(tmp_path):
    """The 2026-09-06 shape. Before: `failed`, no picture. Now: `interrupted`,
    with the geometry the interim builds left behind."""
    root = tmp_path / "worlds"
    world_id, session_id, engine = start_live_world(root, frames=8)
    # An interim rebuild happened before the death, as it did on the walk.
    engine.build(world_id, session_id)
    store = WorldStore(root)
    _write_dead_lock(store, world_id)
    try:
        payload = _payload(root)
    finally:
        engine.stop_session()
    lifecycle = payload["lifecycle"]
    assert lifecycle["state"] == LIFECYCLE_INTERRUPTED
    assert "no longer running" in lifecycle["evidence"]
    assert "without stopping" in lifecycle["reason"]
    assert lifecycle["finalization"] is None
    assert payload["geometry"]["available"] is True
    assert payload["model_state"] == MODEL_STATE_INTERRUPTED
    assert payload["model_state_reason"] == lifecycle["reason"]
    assert payload["world_snapshot"]["keyframe_count"] >= 2
    # Nothing live: the unpinned answer is the newest world, and says so.
    assert payload["selection"]["mode"] == SELECTION_LATEST


def test_a_dead_builder_with_no_build_is_interrupted_with_no_geometry(tmp_path):
    root = tmp_path / "worlds"
    world_id, session_id, engine = start_live_world(root, frames=4)
    _write_dead_lock(WorldStore(root), world_id)
    try:
        payload = _payload(root)
    finally:
        engine.stop_session()
    assert payload["lifecycle"]["state"] == LIFECYCLE_INTERRUPTED
    assert payload["geometry"]["available"] is False
    assert payload["model_state"] == MODEL_STATE_INTERRUPTED


def test_a_builder_that_died_while_finalizing_is_interrupted_not_finalizing(tmp_path):
    root = tmp_path / "worlds"
    world_id, session_id, engine = start_live_world(root, frames=6)
    engine.stop_session(hold_lock=True)
    engine.build(world_id, session_id)
    # The process is gone; the lock and the pending finalization remain.
    _write_dead_lock(WorldStore(root), world_id)
    payload = _payload(root)
    lifecycle = payload["lifecycle"]
    assert lifecycle["state"] == LIFECYCLE_INTERRUPTED
    assert "finaliz" in lifecycle["reason"]
    assert lifecycle["finalization"]["state"] == "pending"
    assert payload["geometry"]["available"] is True
    assert payload["model_state"] == MODEL_STATE_INTERRUPTED


def test_a_session_the_builder_was_asked_to_stop_is_interrupted_but_complete(tmp_path):
    """Soft stop mid-walk: the session ended as `interrupted`, the final
    solve was skipped, the final build ran. The wire says interrupted --
    the walk did not end by the wearer's Stop -- and carries the record."""
    root = tmp_path / "worlds"
    world_id, session_id, engine = start_live_world(root, frames=6)
    engine.stop_session("interrupted", hold_lock=True)
    engine.build(world_id, session_id)
    engine.mark_finalization(
        world_id, session_id, state=FINALIZATION_COMPLETE, final_solve=FINAL_SOLVE_SKIPPED,
        detail="final solve skipped: stop requested (stdin-closed) while observing",
    )
    engine.release_world(world_id)
    payload = _payload(root)
    lifecycle = payload["lifecycle"]
    assert lifecycle["state"] == LIFECYCLE_INTERRUPTED
    assert "interrupted" in lifecycle["reason"]
    assert lifecycle["finalization"]["state"] == FINALIZATION_COMPLETE
    assert lifecycle["finalization"]["final_solve"] == FINAL_SOLVE_SKIPPED
    assert payload["geometry"]["available"] is True
    assert payload["geometry"]["current"] is True
    assert payload["model_state"] == MODEL_STATE_INTERRUPTED


def test_an_errored_session_is_interrupted_with_the_error(tmp_path):
    root = tmp_path / "worlds"
    world_id, session_id, engine = start_live_world(root, frames=6)
    engine.stop_session("error", hold_lock=True)
    engine.mark_finalization(
        world_id, session_id, state=FINALIZATION_INTERRUPTED, final_solve=None,
        detail="RuntimeError: boom",
    )
    engine.release_world(world_id)
    payload = _payload(root)
    assert payload["lifecycle"]["state"] == LIFECYCLE_INTERRUPTED
    assert payload["lifecycle"]["finalization"]["detail"] == "RuntimeError: boom"
    assert payload["model_state"] == MODEL_STATE_INTERRUPTED


# -- the legacy records still read as they did -----------------------------


def test_a_record_without_a_finalization_block_keeps_its_old_states(tmp_path):
    root = tmp_path / "worlds"
    world_id, session_id, engine = start_live_world(root, frames=6)
    engine.stop_session()
    assert _payload(root)["lifecycle"]["state"] == "stopped_unbuilt"
    engine.build(world_id, session_id)
    payload = _payload(root)
    assert payload["lifecycle"]["state"] == LIFECYCLE_READY
    assert payload["lifecycle"]["finalization"] is None


# -- selection: why this world is on the wire --------------------------------


def test_the_selection_names_a_live_world_as_live(tmp_path):
    root = tmp_path / "worlds"
    build_world(root, frames=6, name="older")
    world_id, session_id, engine = start_live_world(root, frames=6, name="live")
    try:
        payload = _payload(root)
        assert payload["lifecycle"]["state"] == LIFECYCLE_RECEIVING
        assert payload["model_state"] == MODEL_STATE_RECEIVING
        assert payload["selection"] == {
            "mode": SELECTION_LIVE,
            "world_id": world_id,
            "session_id": session_id,
            "reason": "a live builder holds this world's writer lock",
        }
    finally:
        engine.stop_session()


def test_the_selection_names_a_pinned_world_as_pinned_even_when_another_is_live(tmp_path):
    root = tmp_path / "worlds"
    old_world, old_session = build_world(root, frames=6, name="older")
    world_id, session_id, engine = start_live_world(root, frames=6, name="live")
    try:
        payload = _payload(root, old_world, old_session)
        assert payload["selection"]["mode"] == SELECTION_PINNED
        assert payload["selection"]["world_id"] == old_world
        assert payload["selection"]["session_id"] == old_session
        payload = _payload(root, old_world)
        assert payload["selection"]["mode"] == SELECTION_PINNED
    finally:
        engine.stop_session()


def test_the_selection_with_nothing_live_is_latest_and_names_the_newest_world(tmp_path):
    root = tmp_path / "worlds"
    build_world(root, frames=6, name="first")
    time.sleep(0.01)
    newest, newest_session = build_world(root, frames=6, name="second")
    payload = _payload(root)
    assert payload["selection"]["mode"] == SELECTION_LATEST
    assert payload["selection"]["world_id"] == newest
    assert payload["selection"]["session_id"] == newest_session
    assert "most recently updated" in payload["selection"]["reason"]


def test_an_unavailable_payload_carries_an_empty_selection(tmp_path):
    payload = _payload(tmp_path / "empty")
    assert payload["selection"] == {
        "mode": SELECTION_NONE, "world_id": None, "session_id": None,
        "reason": payload["model_state_reason"],
    }


def test_the_selection_is_not_part_of_the_revision(tmp_path):
    """Two subscriptions -- one pinned to the newest world, one unpinned --
    describe the same bytes on disk; the revision must agree so a client
    switching between them sees "same data", not a phantom change."""
    root = tmp_path / "worlds"
    world_id, session_id = build_world(root, frames=6)
    producer = WorldBuilderStatusProducer(root, time.time)
    pinned = producer.snapshot(world_id, session_id)
    unpinned = producer.snapshot(None, None)
    assert pinned.payload["selection"]["mode"] != unpinned.payload["selection"]["mode"]
    assert pinned.revision == unpinned.revision


# -- the listing says the same thing ----------------------------------------


def test_the_listing_states_each_session_the_way_the_status_channel_does(tmp_path):
    root = tmp_path / "worlds"
    store = WorldStore(root)
    done_world, done_session = build_world(root, frames=6, name="done")
    dead_world, dead_session, engine = start_live_world(root, frames=6, name="dead")
    engine.build(dead_world, dead_session)
    _write_dead_lock(store, dead_world)
    fin_world, fin_session, fin_engine = start_live_world(root, frames=6, name="finalizing")
    fin_engine.stop_session(hold_lock=True)
    # The listing does not call a lock held by the asking process "live"
    # (a viewer inside the writer is not a builder), so the finalizing
    # builder is stood in for by the parent process, which is running by
    # definition -- with its start time, as a real lock carries.
    store.lock_path(fin_world).write_text(json.dumps({
        "pid": os.getppid(), "created_at": psutil.Process(os.getppid()).create_time(),
    }), encoding="utf-8")
    try:
        listing = build_world_listing(store)
    finally:
        fin_engine.release_world(fin_world)
        engine.stop_session()
    by_id = {w["world_id"]: w for w in listing["worlds"]}
    done = by_id[done_world]["sessions"][0]
    dead = by_id[dead_world]["sessions"][0]
    fin = by_id[fin_world]["sessions"][0]
    assert done["state"] == "complete" and done["abandoned"] is False
    assert dead["state"] == "interrupted" and dead["abandoned"] is True
    assert dead["has_geometry"] is True
    # The record still says 0 (it was never rewritten); the journal says
    # how many keyframes actually landed, and the listing carries both.
    assert dead["keyframes_accepted"] == 0
    assert dead["keyframes_journaled"] >= 2
    assert fin["state"] == "finalizing" and fin["abandoned"] is False
    assert fin["finalization"]["state"] == "pending"
    assert by_id[fin_world]["live"] is True
    assert by_id[dead_world]["live"] is False
