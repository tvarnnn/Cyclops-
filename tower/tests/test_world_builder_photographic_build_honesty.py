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
    _STAGE_STATUS_MAX_AGE_S,
    LIFECYCLE_FINALIZING,
    LIFECYCLE_INTERRUPTED,
    LIFECYCLE_READY,
    LIFECYCLE_RECEIVING,
    MODEL_STATE_FINALIZING,
    MODEL_STATE_FINALIZED,
    SELECTION_FINALIZING,
    SELECTION_LATEST,
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


_COMPLETE_SOLVED = {
    "state": FINALIZATION_COMPLETE,
    "final_solve": FINAL_SOLVE_SOLVED,
    "started_at": 2.0,
    "updated_at": 3.0,
    "detail": None,
}


def _finalized(store, world_id, session_id, *, end_reason="stop",
               finalization=None) -> None:
    """The record the shipped builder leaves just before it lets the lock go.

    `end_reason` is a parameter because `"stop"` is NOT the ordinary shape.
    `world_builder_library.session_state` says so in its own words: leaving
    the World Builder screen sends `session/stop` while the capture is still
    open, so the record reads `end_reason: interrupted` and then finalises
    complete, solved, with geometry. A walk that trips the recorder's
    40-minute bound ends `interrupted` too, by design.
    """
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
            end_reason=end_reason,
            finalization=(
                dict(_COMPLETE_SOLVED) if finalization is None else finalization
            ),
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
    # THE ARM UNDER TEST MUST ACTUALLY BE THE ONE REACHED. Without these two
    # lines the test passed with the whole fix reverted, because `ready`
    # carries the same record -- caught by a reviewer who set `building` to
    # None and counted which of these tests still passed.
    assert lifecycle["state"] == LIFECYCLE_FINALIZING
    assert lifecycle["build_in_progress"] is True
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


@pytest.mark.parametrize("end_reason", ["stop", "interrupted", "error"])
def test_the_status_poll_and_the_revision_poll_agree_once_the_lock_is_released(
    derived_world, end_reason,
):
    """One predicate, two surfaces -- IN THE WINDOW THIS FIX IS ABOUT, which
    is a stopped session with the writer lock released.

    `/render/revision` answers `live` with `session_build_running`; the status
    page must not answer the same question differently there, or a follower
    swaps in a world the status page is still telling the wearer to wait for.

    DELIBERATELY NOT AN UNCONDITIONAL CLAIM, and the name now says so. The two
    legitimately differ while the world lock is held: `session_build_running`
    is true for an OPEN session, and `lifecycle.build_in_progress` is
    deliberately false under `receiving`, because "frames are arriving" is not
    "a build is running". They also differ once this module's own staleness
    ceiling fires -- see
    `test_a_running_status_that_has_not_moved_for_hours_is_not_a_build`.
    """
    from tower.results import world_builder_render as R

    store, world_id, session_id = derived_world
    _finalized(store, world_id, session_id, end_reason=end_reason)
    for stage_state, pid in (("running", os.getpid()),
                             ("running", _dead_pid()),
                             ("ok", os.getpid())):
        _write_status(store, world_id, session_id, "surface",
                      state=stage_state, pid=pid, updated_at=time.time())
        lifecycle, _ = _lifecycle_of(store, world_id, session_id)
        assert lifecycle["build_in_progress"] is R.session_build_running(
            store, world_id, session_id
        ), (end_reason, stage_state, pid, lifecycle["evidence"])


# -- H1: the ordinary iOS record shape is `interrupted`, not `stop` ---------


@pytest.mark.parametrize("end_reason", ["interrupted", "error"])
def test_a_capture_that_ended_badly_still_says_a_build_is_running(
    derived_world, end_reason,
):
    """THE BLOCKER. The first version of this fix put its arm BELOW the two
    `end_reason in ("error", "interrupted")` branches, so those won and the
    payload read `ready` / `finalized` / `build_in_progress: False` with a
    live surface process on disk. `finalized` is the word iOS maps to
    **Saved**: the exact ninety-second lie this file exists to kill, still
    fully reachable -- and by the COMMON path, because leaving the World
    Builder screen sends `session/stop` while the capture is open, which
    records `end_reason: interrupted` and then finalises complete and solved
    (`world_builder_library.session_state` says so in its own comments).
    """
    store, world_id, session_id = derived_world
    _finalized(store, world_id, session_id, end_reason=end_reason)
    _write_status(store, world_id, session_id, "surface",
                  state="running", pid=os.getpid(), updated_at=time.time())

    lifecycle, payload = _lifecycle_of(store, world_id, session_id)
    assert lifecycle["state"] == LIFECYCLE_FINALIZING, end_reason
    assert lifecycle["build_in_progress"] is True, end_reason
    assert payload["model_state"] == MODEL_STATE_FINALIZING, end_reason


def test_a_failed_capture_still_says_what_happened_to_the_capture(derived_world):
    """`finalizing` is a claim about a PROCESS, not a claim that the walk went
    well. A capture that ended in an error and is nonetheless being built --
    `stop_session(END_REASON_ERROR)` still runs a best-effort final build, so
    `--surface` still runs -- must say both things: wait, and this one ended
    badly. Erasing the second would be a new lie in place of the old one.
    """
    store, world_id, session_id = derived_world
    _finalized(store, world_id, session_id, end_reason="error",
               finalization={"state": "pending", "final_solve": "pending",
                             "started_at": 2.0, "updated_at": 3.0,
                             "detail": "the solver died"})
    _write_status(store, world_id, session_id, "surface",
                  state="running", pid=os.getpid(), updated_at=time.time())

    lifecycle, _ = _lifecycle_of(store, world_id, session_id)
    assert lifecycle["state"] == LIFECYCLE_FINALIZING
    assert lifecycle["build_in_progress"] is True
    assert "'error'" in lifecycle["reason"], lifecycle["reason"]
    assert "the solver died" in lifecycle["reason"], lifecycle["reason"]
    # And the builder's own finalization record travels untouched.
    assert lifecycle["finalization"]["state"] == "pending"


@pytest.mark.parametrize("end_reason", ["interrupted", "error"])
def test_a_capture_that_ended_badly_with_nothing_running_is_unchanged(
    derived_world, end_reason,
):
    """The other half. No stage status, so nothing is running, so the two
    `end_reason` arms must answer exactly as they always did."""
    store, world_id, session_id = derived_world
    _finalized(store, world_id, session_id, end_reason=end_reason)

    lifecycle, payload = _lifecycle_of(store, world_id, session_id)
    # A completed finalization outranks how the capture ended: `ready`.
    assert lifecycle["state"] == LIFECYCLE_READY, end_reason
    assert lifecycle["build_in_progress"] is False, end_reason
    assert payload["model_state"] == MODEL_STATE_FINALIZED, end_reason
    assert repr(end_reason) in lifecycle["reason"], lifecycle["reason"]


def test_a_dead_lock_over_a_running_stage_still_says_wait(derived_world):
    """A lock naming a dead pid AND a live surface: the builder died and the
    surface stage was run since. `interrupted` renders as "Needs retry", which
    invites the wearer to redo the walk -- and the retry is what kills the
    build that is running. The present-tense fact wins, and the dead lock is
    still in the evidence."""
    store, world_id, session_id = derived_world
    _finalized(store, world_id, session_id)
    store.lock_path(world_id).write_text(
        json.dumps({"pid": _dead_pid()}), encoding="utf-8"
    )
    before, _ = _lifecycle_of(store, world_id, session_id)
    assert before["state"] == LIFECYCLE_INTERRUPTED

    _write_status(store, world_id, session_id, "surface",
                  state="running", pid=os.getpid(), updated_at=time.time())
    lifecycle, _ = _lifecycle_of(store, world_id, session_id)
    assert lifecycle["state"] == LIFECYCLE_FINALIZING
    assert lifecycle["build_in_progress"] is True
    assert "no longer running" in lifecycle["evidence"], lifecycle["evidence"]


# -- M5: a status that cannot go stale must still be bounded ----------------


@pytest.fixture
def pid_start_time_hidden(monkeypatch):
    """Windows' `AccessDenied` on `create_time()`, which is the whole of M5.

    `store._holder_is_running` answers that with "CANNOT JUDGE IS NOT DEAD"
    and returns True -- the right default for a writer lock, and the reason
    the pid probe underneath `status_is_stale` can never call such a status
    stale. With this in place the pid probe says ALIVE no matter how old the
    status is, so only an age ceiling can end it.
    """

    def _denied(self):
        raise psutil.AccessDenied(self.pid)

    monkeypatch.setattr(psutil.Process, "create_time", _denied)


def test_a_hidden_start_time_really_does_read_alive(
    finalized_world, pid_start_time_hidden,
):
    """The premise of the two tests below, pinned before they lean on it: with
    `create_time()` denied, a fresh `running` status is believed."""
    store, world_id, session_id = finalized_world
    _write_status(store, world_id, session_id, "surface", state="running",
                  pid=os.getpid(), updated_at=time.time())

    lifecycle, _ = _lifecycle_of(store, world_id, session_id)
    assert lifecycle["state"] == LIFECYCLE_FINALIZING


def test_a_running_status_that_has_not_moved_for_hours_is_not_a_build(
    finalized_world, pid_start_time_hidden,
):
    """A killed builder leaves `status.json` saying `running` under pid P. If
    the OS later recycles P onto a process this Tower may not inspect, the pid
    probe reads alive forever (see the fixture). Nothing else clears it: only
    another `surfacify()` rewrites the file, and `_stage_running`'s "manifest
    newer than the status" escape hatch needs a manifest a killed build never
    wrote. Without a bound the phone would sit on "Improving" permanently over
    a finished world -- this fix's own lie, pointing the other way.
    """
    store, world_id, session_id = finalized_world
    _write_status(store, world_id, session_id, "surface", state="running",
                  pid=os.getpid(),  # alive, and its start time is hidden
                  updated_at=time.time() - 2 * _STAGE_STATUS_MAX_AGE_S)

    lifecycle, payload = _lifecycle_of(store, world_id, session_id)
    assert lifecycle["state"] == LIFECYCLE_READY
    assert lifecycle["build_in_progress"] is False
    assert payload["model_state"] == MODEL_STATE_FINALIZED


def test_the_ceiling_is_loose_enough_for_a_real_build(
    finalized_world, pid_start_time_hidden,
):
    """The bound must not fire on a build that is merely slow. `_status()`
    rewrites `running` only at sub-stage boundaries -- six writes across a
    6-16 minute build -- so `updated_at` is minutes-coarse BY DESIGN and the
    gap between two writes can be most of a stage. A ceiling tuned tight would
    reintroduce "Saved" mid-build, which is the failure this file exists for.

    Measured on this machine: surface 374 s and appearance 85-91 s for ~385
    keyframes, and a 690-keyframe walk scales about 1.75x, so ~655 s for the
    longest single stage. Twenty minutes of silence must still count as
    building.
    """
    store, world_id, session_id = finalized_world
    _write_status(store, world_id, session_id, "surface", state="running",
                  pid=os.getpid(), updated_at=time.time() - 20 * 60)

    lifecycle, _ = _lifecycle_of(store, world_id, session_id)
    assert lifecycle["state"] == LIFECYCLE_FINALIZING
    assert _STAGE_STATUS_MAX_AGE_S >= 2 * 655, (
        "the ceiling must clear the longest measured stage with real headroom"
    )


def test_a_status_with_no_usable_timestamp_falls_back_to_the_file(
    finalized_world, pid_start_time_hidden,
):
    """`updated_at` is written by the pipeline on every state, but a status
    that lost it must still be bounded rather than trusted forever. The file's
    own mtime is always there, written by the same atomic replace."""
    store, world_id, session_id = finalized_world
    _write_status(store, world_id, session_id, "surface",
                  state="running", pid=os.getpid())  # no updated_at at all
    # Freshly written, so its mtime is now: still a build.
    lifecycle, _ = _lifecycle_of(store, world_id, session_id)
    assert lifecycle["state"] == LIFECYCLE_FINALIZING

    path = store.world_dir(world_id) / "surface" / session_id / "status.json"
    old = time.time() - 2 * _STAGE_STATUS_MAX_AGE_S
    os.utime(path, (old, old))
    lifecycle, _ = _lifecycle_of(store, world_id, session_id)
    assert lifecycle["state"] == LIFECYCLE_READY


# -- M6: one broken module must not blind the other two ---------------------


def test_a_dense_module_that_will_not_import_does_not_blind_the_surface(
    finalized_world, monkeypatch,
):
    """`session_build_running` deliberately puts the surface and dense probes
    in separate `try` blocks. The first version of this fix imported both
    staleness functions inside ONE `try`, so a dense import failure made the
    probe answer "nothing is building" for surface and appearance too -- the
    fix silently disabling itself, and the phone back to "Saved"."""
    import sys

    class _Broken:  # no `status_is_stale`, so `from ... import` raises
        pass

    monkeypatch.setitem(sys.modules, "tower.world_builder.dense_pipeline", _Broken())
    store, world_id, session_id = finalized_world
    _write_status(store, world_id, session_id, "surface",
                  state="running", pid=os.getpid(), updated_at=time.time())

    lifecycle, _ = _lifecycle_of(store, world_id, session_id)
    assert lifecycle["state"] == LIFECYCLE_FINALIZING
    assert "surface" in lifecycle["evidence"]


def test_a_surface_module_that_will_not_import_does_not_blind_the_dense(
    finalized_world, monkeypatch,
):
    import sys

    class _Broken:
        pass

    monkeypatch.setitem(sys.modules, "tower.world_builder.surface_pipeline", _Broken())
    store, world_id, session_id = finalized_world
    _write_status(store, world_id, session_id, "dense",
                  state="running", pid=os.getpid(), updated_at=time.time())

    lifecycle, _ = _lifecycle_of(store, world_id, session_id)
    assert lifecycle["state"] == LIFECYCLE_FINALIZING
    assert "dense" in lifecycle["evidence"]


# -- M3: the picker row and the panel are read by the same wearer -----------


def _row(store, world_id, session_id) -> dict:
    from tower.results.world_builder_library import build_world_listing

    listing = build_world_listing(store)
    world = next(w for w in listing["worlds"] if w["world_id"] == world_id)
    return next(s for s in world["sessions"] if s["session_id"] == session_id)


@pytest.mark.parametrize("end_reason", ["stop", "interrupted"])
def test_the_saved_worlds_row_agrees_with_the_panel(derived_world, end_reason):
    """`session_state` derives `live` from the writer lock alone, and the lock
    is exactly what the surface stage runs without. Measured before this fix:
    the status channel said `finalizing` / "Improving" while the Saved Worlds
    picker row for the same session said `complete`. A wearer who opens the
    picker reads "Complete" and shuts the Tower down -- the precise failure
    this whole change exists to prevent, on the surface a person actually
    chooses a walk from.
    """
    store, world_id, session_id = derived_world
    _finalized(store, world_id, session_id, end_reason=end_reason)
    assert _row(store, world_id, session_id)["state"] == "complete"

    _write_status(store, world_id, session_id, "surface",
                  state="running", pid=os.getpid(), updated_at=time.time())
    lifecycle, _ = _lifecycle_of(store, world_id, session_id)
    assert lifecycle["state"] == LIFECYCLE_FINALIZING
    assert _row(store, world_id, session_id)["state"] == "finalizing", (
        "the picker row and the panel disagree about the same session"
    )


def test_the_saved_worlds_row_is_unchanged_when_nothing_is_running(
    finalized_world,
):
    store, world_id, session_id = finalized_world
    assert _row(store, world_id, session_id)["state"] == "complete"
    _write_status(store, world_id, session_id, "surface",
                  state="running", pid=_dead_pid(), updated_at=time.time())
    assert _row(store, world_id, session_id)["state"] == "complete"


def test_the_saved_worlds_row_still_says_receiving_while_the_lock_is_held(
    derived_world, monkeypatch,
):
    """The listing's own `live` arm is untouched: an OPEN session under a live
    lock is `receiving`, whatever a stage status says.

    The lock is simulated at `lock_holder`, as `test_world_builder_render_
    revision_live` does, because the listing's probe excludes the calling
    process (a Tower cannot be its own builder) and an in-process engine
    therefore never reads as live.
    """
    from tower.world_builder.store import WorldStore

    store, world_id, session_id = derived_world
    session = store.read_session(world_id, session_id)
    store.write_session(Session(session_id=session_id, world_id=world_id,
                                started_at=session.started_at, ended_at=None))
    monkeypatch.setattr(
        WorldStore, "lock_holder",
        lambda self, wid: {"pid": 1, "alive": True, "unreadable": False},
    )
    _write_status(store, world_id, session_id, "surface",
                  state="running", pid=os.getpid(), updated_at=time.time())
    assert _row(store, world_id, session_id)["state"] == "receiving"

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


# -- the selection block and the lifecycle block are one payload ------------


def _unpinned(store) -> dict:
    """What an UNPINNED subscription is answered with -- the live screen's."""
    return WorldBuilderStatusProducer(store.root, time.time).snapshot(None, None).payload


def test_the_selection_does_not_call_a_building_world_history(finalized_world):
    """`_most_relevant` derived `live` from the writer lock alone, so during
    the six to sixteen minutes of the photographic build it answered
    `latest` / "nothing is live" in the SAME payload whose lifecycle said
    `finalizing` with `build_in_progress: true`.

    On iOS `WorldSelection.isHistoryOfferedAsLive` is `mode == .latest`, and
    `TowerWorldBuilderClient` then presents the report as `.idle` -- "No world
    yet" -- unless `followedWalk` still names the world; `followedWalk` stops
    being refreshed the moment the selection turns `latest`. So a relaunch or
    a jetsam kill mid-build dropped the live screen to "No world yet" while
    the Tower was reporting a build in progress.
    """
    store, world_id, session_id = finalized_world
    before = _unpinned(store)
    assert before["selection"]["mode"] == SELECTION_LATEST
    assert before["lifecycle"]["build_in_progress"] is False

    _write_status(store, world_id, session_id, "surface",
                  state="running", pid=os.getpid(), updated_at=time.time())

    payload = _unpinned(store)
    assert payload["lifecycle"]["state"] == LIFECYCLE_FINALIZING
    assert payload["selection"]["mode"] == SELECTION_FINALIZING, (
        "one payload said a build was running and offered the world as history"
    )
    assert payload["selection"]["world_id"] == world_id
    # The existing `finalizing` selection reason cites a LIVE builder holding
    # the lock. There is none here, and reusing that sentence would be the
    # same untruth the lifecycle evidence was careful not to tell.
    assert "a live builder holds" not in payload["selection"]["reason"], (
        payload["selection"]["reason"]
    )
    assert "photographic" in payload["selection"]["reason"]


def test_the_selection_is_unchanged_when_nothing_is_running(finalized_world):
    store, world_id, session_id = finalized_world
    _write_status(store, world_id, session_id, "surface",
                  state="running", pid=_dead_pid(), updated_at=time.time())
    payload = _unpinned(store)
    assert payload["selection"]["mode"] == SELECTION_LATEST
    assert payload["lifecycle"]["state"] == LIFECYCLE_READY


# -- a broken probe must say so, not quietly answer "Saved" -----------------


def _break_the_probe(monkeypatch):
    """What another lane's in-flight edit did for real: a module that raises
    at import. `world_builder_render` is imported inside the probe, so this
    reproduces the failure the reviewer actually hit."""
    import sys

    class _Exploding:
        def __getattr__(self, name):
            raise RuntimeError("world_builder_render is broken right now")

    monkeypatch.setitem(sys.modules, "tower.results.world_builder_render",
                        _Exploding())


def test_a_broken_probe_is_reported_not_swallowed(finalized_world, monkeypatch):
    """THE FAILURE MODE OF THIS FIX MUST NOT BE THE BUG IT FIXES.

    Every probe used to be wrapped in `except Exception: logger.debug(...)`
    returning None, so any import or read error silently restored the pre-fix
    behaviour -- `ready`, "Saved" over a world that is still building, and
    `build_in_progress_unavailable_reason` left null, which is the one field
    that exists to carry exactly this.
    """
    store, world_id, session_id = finalized_world
    _write_status(store, world_id, session_id, "surface",
                  state="running", pid=os.getpid(), updated_at=time.time())
    _break_the_probe(monkeypatch)

    lifecycle, _ = _lifecycle_of(store, world_id, session_id)
    assert lifecycle["build_in_progress"] is None, (
        "False here is the pre-fix claim asserted on no evidence"
    )
    assert lifecycle["build_in_progress_unavailable_reason"], lifecycle
    assert "probe" in lifecycle["build_in_progress_unavailable_reason"]


def test_a_broken_probe_warns_once_and_then_stays_quiet(
    finalized_world, monkeypatch, caplog,
):
    """A 2 Hz poll must not write two warnings a second for the life of the
    Tower. The payload carries the reason on every poll; the log only has to
    make it visible."""
    import logging

    from tower.results import world_builder as W

    monkeypatch.setattr(W, "_PROBE_FAILURES_SEEN", set())
    store, world_id, session_id = finalized_world
    _break_the_probe(monkeypatch)
    with caplog.at_level(logging.DEBUG, logger=W.logger.name):
        for _ in range(5):
            _lifecycle_of(store, world_id, session_id)
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1, [r.getMessage() for r in warnings]


def test_a_probe_that_works_says_nothing_about_being_unable_to_tell(
    finalized_world,
):
    store, world_id, session_id = finalized_world
    lifecycle, _ = _lifecycle_of(store, world_id, session_id)
    assert lifecycle["build_in_progress"] is False
    assert lifecycle["build_in_progress_unavailable_reason"] is None
