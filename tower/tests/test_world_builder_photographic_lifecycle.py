"""Saved means the photographic room is there. It must mean nothing else.

THE THREE WAYS "SAVED" WAS A LIE, from the Mac/iOS validation of
2026-09-22 (§7 of `WORLD-BUILDER-LIVE-WORLD-VISUALIZATION-MAC-VALIDATION.md`).

The serving path asked one present-tense question -- *is a photographic
stage running right now?* -- and answered the wearer's question from it.
Running meant "Improving"; not running meant **Saved**. Three different
false Saveds came out of that one mistake:

  * **T2, the flicker.** "Running" is false in the gaps BETWEEN the stages:
    between the lock being released and the surface's first `running`
    status, and between the surface's `ok` and the appearance's first
    `running` status -- a label-policy pass and a SHA-1 pass over every
    keyframe, tens of seconds on a long walk. The phone said Saved for
    those polls and then went back to Improving.

  * **T3, the false success.** A stage that FAILED is also not running, and
    so is a stage that was never attempted. Both read Saved, so "the
    photographic build failed" was indistinguishable from "the photographic
    room is ready".

  * **T4, failing open.** When the probe itself broke, the code computed a
    reason, put it in `build_in_progress_unavailable_reason` -- a field
    `WORLD-BUILDER-IOS.md` §2.4 lists as **not consumed** -- and left the
    state at ready. A broken probe rendered as Saved.

`tower/world_builder/photographic.py` answers the settled question instead
(*does this world still owe a photographic room?*), and these tests pin
each row of the resulting matrix. They also pin the thing that must NOT
change: the 165 worlds on this machine that were built before the
photographic stages existed are finished, and they still say so.

The sibling file `test_world_builder_photographic_build_honesty.py` pins
the present-tense half -- that a verifiably RUNNING stage says Improving
and names its pid. This file pins everything either side of it.
"""

import json
import time

import psutil
import pytest

from tower.results.world_builder import (
    LIFECYCLE_FINALIZING,
    LIFECYCLE_READY,
    MODEL_STATE_FINALIZED,
    MODEL_STATE_FINALIZING,
    WorldBuilderStatusProducer,
)
from tower.world_builder.events import WorldEvent
from tower.world_builder.photographic import (
    PHOTOGRAPHIC_COMPLETE,
    PHOTOGRAPHIC_FAILED,
    PHOTOGRAPHIC_NEVER_RECORDED,
    PHOTOGRAPHIC_OWED,
    PHOTOGRAPHIC_RUNNING,
    PHOTOGRAPHIC_UNOBSERVABLE,
    photographic_state,
)
from tower.world_builder.records import (
    FINAL_SOLVE_SOLVED,
    FINALIZATION_COMPLETE,
    STAGE_APPEARANCE,
    STAGE_STATE_FAILED,
    STAGE_STATE_OK,
    STAGE_STATE_RUNNING,
    STAGE_STATE_STOPPED,
    STAGE_SURFACE,
    Session,
)

_COMPLETE_SOLVED = {
    "state": FINALIZATION_COMPLETE,
    "final_solve": FINAL_SOLVE_SOLVED,
    "started_at": 2.0,
    "updated_at": 3.0,
    "detail": None,
}


def _dead_pid() -> int:
    return next(pid for pid in range(100_000, 200_000) if not psutil.pid_exists(pid))


def _finalized(store, world_id, session_id, *, stages=None):
    """The record the builder leaves as it lets the writer lock go."""
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
            finalization=dict(_COMPLETE_SOLVED),
            stages=stages,
        )
    )


def _stage(state, **extra):
    entry = {"attempted": True, "state": state, "detail": None}
    entry.update(extra)
    return entry


def _write_status(store, world_id, session_id, stage_dir, **fields):
    """The stage's own `status.json`.

    `stage_dir` rather than `stage` because `stage` is also a FIELD inside
    the file (`"stage": "depth"`), and the collision is not hypothetical --
    it cost this file a round of red.
    """
    root = store.world_dir(world_id) / stage_dir / session_id
    root.mkdir(parents=True, exist_ok=True)
    (root / "status.json").write_text(json.dumps(fields), encoding="utf-8")


def _clear_statuses(store, world_id, session_id):
    """Every stage file gone -- the disk state of a session that has not
    started a stage since its statuses were pruned."""
    for stage_dir in (STAGE_SURFACE, STAGE_APPEARANCE):
        path = store.world_dir(world_id) / stage_dir / session_id / "status.json"
        if path.exists():
            path.unlink()


def _payload(store, world_id, session_id):
    return (
        WorldBuilderStatusProducer(store.root, time.time)
        .snapshot(world_id, session_id)
        .payload
    )


def _state_of(store, world_id, session_id):
    session = store.read_session(world_id, session_id)
    return photographic_state(store, world_id, session_id, session)


# -- the settled judgement itself -------------------------------------------


class TestTheSettledVocabulary:
    """`photographic_state` on each shape, without the lifecycle around it."""

    def test_a_finished_appearance_is_complete(self, derived_world):
        store, world_id, session_id = derived_world
        _finalized(store, world_id, session_id, stages={
            STAGE_SURFACE: _stage(STAGE_STATE_OK),
            STAGE_APPEARANCE: _stage(STAGE_STATE_OK),
        })
        assert _state_of(store, world_id, session_id)["state"] == (
            PHOTOGRAPHIC_COMPLETE
        )

    def test_a_finished_surface_alone_is_not_complete(self, derived_world):
        """A surface with no shading on it is a grey mesh, not the room.

        The product invariant is the photographic room; `world_finish_
        pending._retire` says the same in its own words ("a saved world is
        the surface AND the shading on it"). A surface that finished while
        the appearance is still owed is owed.
        """
        store, world_id, session_id = derived_world
        _finalized(store, world_id, session_id, stages={
            STAGE_SURFACE: _stage(STAGE_STATE_OK),
            STAGE_APPEARANCE: _stage(STAGE_STATE_RUNNING),
        })
        verdict = _state_of(store, world_id, session_id)
        assert verdict["state"] == PHOTOGRAPHIC_OWED
        assert verdict["stage"] == STAGE_APPEARANCE

    def test_a_failed_stage_beats_an_owed_one(self, derived_world):
        """Waiting does not fix a stage that raised, so it must not say wait."""
        store, world_id, session_id = derived_world
        _finalized(store, world_id, session_id, stages={
            STAGE_SURFACE: _stage(STAGE_STATE_FAILED, detail="cuda oom"),
            STAGE_APPEARANCE: _stage(STAGE_STATE_STOPPED),
        })
        verdict = _state_of(store, world_id, session_id)
        assert verdict["state"] == PHOTOGRAPHIC_FAILED
        assert verdict["stage"] == STAGE_SURFACE
        assert "cuda oom" in verdict["detail"]

    def test_a_historical_world_is_never_recorded_not_owed(self, derived_world):
        """THE 165-WORLD CASE, and the one that must not move.

        No `stages` block, no stage artifact on disk: a Tower that never ran
        a photographic stage. That is not the same fact as one that tried
        and failed, and reading it as "owed" would queue tens of hours of
        GPU against worlds that are finished and fine.
        """
        store, world_id, session_id = derived_world
        _finalized(store, world_id, session_id, stages=None)
        assert _state_of(store, world_id, session_id)["state"] == (
            PHOTOGRAPHIC_NEVER_RECORDED
        )

    def test_an_interrupted_artifact_with_no_record_is_owed(self, derived_world):
        """World 2f447162's exact shape: a status.json and no record.

        The Tower that wrote it was shut down before it could write a stage
        record. `scripts/world_finish_pending.py` was built for this world
        specifically, and the serving path has to agree with it about which
        worlds are unfinished.
        """
        store, world_id, session_id = derived_world
        _finalized(store, world_id, session_id, stages=None)
        _write_status(store, world_id, session_id, STAGE_SURFACE,
                      state=STAGE_STATE_STOPPED, stage="depth",
                      updated_at=time.time())
        verdict = _state_of(store, world_id, session_id)
        assert verdict["state"] == PHOTOGRAPHIC_OWED
        assert verdict["stage"] == STAGE_SURFACE

    def test_a_live_stage_is_running(self, derived_world):
        store, world_id, session_id = derived_world
        import os

        _finalized(store, world_id, session_id, stages={
            STAGE_SURFACE: _stage(STAGE_STATE_RUNNING),
        })
        _write_status(store, world_id, session_id, STAGE_SURFACE,
                      state=STAGE_STATE_RUNNING, stage="depth",
                      pid=os.getpid(), updated_at=time.time())
        assert _state_of(store, world_id, session_id)["state"] == (
            PHOTOGRAPHIC_RUNNING
        )

    def test_a_record_saying_running_under_a_dead_pid_is_owed(
        self, derived_world
    ):
        """`running` is two facts and only liveness tells them apart."""
        store, world_id, session_id = derived_world
        _finalized(store, world_id, session_id, stages={
            STAGE_SURFACE: _stage(STAGE_STATE_RUNNING),
        })
        _write_status(store, world_id, session_id, STAGE_SURFACE,
                      state=STAGE_STATE_RUNNING, stage="depth",
                      pid=_dead_pid(), updated_at=time.time())
        assert _state_of(store, world_id, session_id)["state"] == (
            PHOTOGRAPHIC_OWED
        )


# -- T2: no Saved in the gaps between the stages ----------------------------


class TestTheStageBoundariesNeverSaySaved:
    """The flicker, pinned at both gaps.

    Neither gap has a `status.json` saying `running`, which is why the
    present-tense probe could not see them and the record can.
    """

    def test_the_gap_before_the_surface_starts_is_not_saved(
        self, derived_world
    ):
        """GAP ONE. The builder now records the surface BEFORE it releases
        the writer lock (`world_build_session.py`, in the `finally`), so
        this window has a record even though no stage has written a status
        file yet."""
        store, world_id, session_id = derived_world
        _finalized(store, world_id, session_id, stages={
            STAGE_SURFACE: _stage(
                STAGE_STATE_RUNNING,
                detail="the photographic stages are about to run",
            ),
        })
        payload = _payload(store, world_id, session_id)
        lifecycle = payload["lifecycle"]
        assert lifecycle["state"] == LIFECYCLE_FINALIZING
        assert payload["model_state"] == MODEL_STATE_FINALIZING
        assert lifecycle["photographic"]["state"] == PHOTOGRAPHIC_OWED

    def test_the_gap_between_surface_and_appearance_is_not_saved(
        self, derived_world
    ):
        """GAP TWO, the big one: the surface has published `ok` and the
        appearance has not written its first `running` status yet. Measured
        at tens of seconds on a 400-keyframe walk."""
        store, world_id, session_id = derived_world
        _finalized(store, world_id, session_id, stages={
            STAGE_SURFACE: _stage(STAGE_STATE_OK),
            STAGE_APPEARANCE: _stage(STAGE_STATE_RUNNING),
        })
        # The surface's own file says `ok` -- finished, not running. This is
        # exactly the disk state the old probe read as "nothing is building".
        _write_status(store, world_id, session_id, STAGE_SURFACE,
                      state=STAGE_STATE_OK, updated_at=time.time())
        payload = _payload(store, world_id, session_id)
        assert payload["lifecycle"]["state"] == LIFECYCLE_FINALIZING
        assert payload["model_state"] == MODEL_STATE_FINALIZING
        assert payload["lifecycle"]["photographic"]["stage"] == (
            STAGE_APPEARANCE
        )

    def test_the_whole_sequence_never_reports_ready_until_the_end(
        self, derived_world
    ):
        """THE SEQUENCE, not the snapshots. Every intermediate disk state a
        real build passes through, in order, asserted to be non-ready -- and
        then the final one, asserted to be ready. A future edit that fixes
        one gap by opening another fails here."""
        store, world_id, session_id = derived_world
        import os

        live = os.getpid()
        during = [
            # lock released, nothing started
            ({STAGE_SURFACE: _stage(STAGE_STATE_RUNNING)}, None),
            # the surface is genuinely running
            ({STAGE_SURFACE: _stage(STAGE_STATE_RUNNING)},
             (STAGE_SURFACE, STAGE_STATE_RUNNING, live)),
            # the surface finished; the appearance has not begun
            ({STAGE_SURFACE: _stage(STAGE_STATE_OK),
              STAGE_APPEARANCE: _stage(STAGE_STATE_RUNNING)},
             (STAGE_SURFACE, STAGE_STATE_OK, live)),
            # the appearance is running
            ({STAGE_SURFACE: _stage(STAGE_STATE_OK),
              STAGE_APPEARANCE: _stage(STAGE_STATE_RUNNING)},
             (STAGE_APPEARANCE, STAGE_STATE_RUNNING, live)),
        ]
        for index, (stages, status) in enumerate(during):
            _finalized(store, world_id, session_id, stages=stages)
            # Each step IS a disk state, not an accumulation of the ones
            # before it: a stale `running` file left behind by step 2 would
            # make step 3 pass for the wrong reason.
            _clear_statuses(store, world_id, session_id)
            if status is not None:
                stage, state, pid = status
                _write_status(store, world_id, session_id, stage,
                              state=state, pid=pid, updated_at=time.time())
            payload = _payload(store, world_id, session_id)
            assert payload["lifecycle"]["state"] == LIFECYCLE_FINALIZING, (
                f"step {index} reported ready mid-build: {payload['lifecycle']}"
            )
            assert payload["model_state"] != MODEL_STATE_FINALIZED, index

        # ...and only now.
        _finalized(store, world_id, session_id, stages={
            STAGE_SURFACE: _stage(STAGE_STATE_OK),
            STAGE_APPEARANCE: _stage(STAGE_STATE_OK),
        })
        _clear_statuses(store, world_id, session_id)
        payload = _payload(store, world_id, session_id)
        assert payload["lifecycle"]["state"] == LIFECYCLE_READY
        assert payload["model_state"] == MODEL_STATE_FINALIZED
        assert payload["lifecycle"]["photographic"]["state"] == (
            PHOTOGRAPHIC_COMPLETE
        )


# -- T3: a failed or owed build must not claim success ----------------------


class TestAFailedBuildDoesNotClaimSuccess:

    def test_a_failed_appearance_is_carried_on_the_wire(self, derived_world):
        """The whole of T3: `session.stages` was read by
        `world_finish_pending.py` and by NOTHING on the wire, so the phone
        could not tell a failed photographic build from a finished one."""
        store, world_id, session_id = derived_world
        _finalized(store, world_id, session_id, stages={
            STAGE_SURFACE: _stage(STAGE_STATE_OK),
            STAGE_APPEARANCE: _stage(
                STAGE_STATE_FAILED, detail="the appearance stage raised"
            ),
        })
        lifecycle = _payload(store, world_id, session_id)["lifecycle"]
        assert lifecycle["photographic"]["state"] == PHOTOGRAPHIC_FAILED
        assert lifecycle["photographic"]["stage"] == STAGE_APPEARANCE
        assert "raised" in lifecycle["photographic"]["detail"]

    def test_a_failed_build_is_not_left_saying_improving_forever(
        self, derived_world
    ):
        """The other half of the bar, and it pulls the opposite way.

        A failed photographic build is TERMINAL: the world is saved, it has
        whatever rung it reached, and nothing more is coming. Reporting it
        as "Improving" would strand the wearer waiting for a build that
        will never arrive -- which the brief forbids as plainly as it
        forbids the false Saved. The truth travels in the `photographic`
        block instead.
        """
        store, world_id, session_id = derived_world
        _finalized(store, world_id, session_id, stages={
            STAGE_SURFACE: _stage(STAGE_STATE_FAILED, detail="no solve"),
        })
        payload = _payload(store, world_id, session_id)
        assert payload["lifecycle"]["state"] == LIFECYCLE_READY
        assert payload["model_state"] == MODEL_STATE_FINALIZED
        assert payload["lifecycle"]["photographic"]["state"] == (
            PHOTOGRAPHIC_FAILED
        )


# -- T4: a probe that breaks must not answer "Saved" ------------------------


def _break_the_render_probe(monkeypatch):
    """A module that raises at import -- what another lane's in-flight edit
    did for real. `world_builder_render` is imported INSIDE the probe, so
    this reproduces the failure a reviewer actually hit."""
    import sys

    monkeypatch.delitem(sys.modules, "tower.results.world_builder_render",
                        raising=False)
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "tower.results.world_builder_render":
            raise RuntimeError("the render module is broken")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)


class TestABrokenProbeDoesNotSaySaved:

    def test_an_unobservable_probe_on_an_unfinished_world_is_not_ready(
        self, derived_world, monkeypatch
    ):
        """T4. The pre-fix code nulled `build_in_progress` -- a field iOS
        does not read -- and left the state at ready, so the caveat it
        carefully computed rendered as **Saved**."""
        store, world_id, session_id = derived_world
        _finalized(store, world_id, session_id, stages={
            STAGE_SURFACE: _stage(STAGE_STATE_RUNNING),
        })
        _break_the_render_probe(monkeypatch)

        payload = _payload(store, world_id, session_id)
        lifecycle = payload["lifecycle"]
        assert lifecycle["state"] == LIFECYCLE_FINALIZING
        assert payload["model_state"] != MODEL_STATE_FINALIZED
        assert lifecycle["photographic"]["state"] == PHOTOGRAPHIC_UNOBSERVABLE
        # "I do not know" is not "no", and the boolean has to say so.
        assert lifecycle["build_in_progress"] is None
        assert lifecycle["build_in_progress_unavailable_reason"]

    def test_a_broken_probe_does_not_relabel_a_historical_world(
        self, derived_world, monkeypatch
    ):
        """THE OTHER FAILURE, and the reason the probe is asked last.

        An earlier draft asked liveness FIRST, for every session. A broken
        probe would then have turned all 165 historical worlds on this
        machine `unobservable` at once, and a conservative mapping would
        have parked every one of them on "Improving" forever -- the
        permanent false Improving the brief forbids. A world with no stage
        record needs no probe to be judged, so it is judged without one.
        """
        store, world_id, session_id = derived_world
        _finalized(store, world_id, session_id, stages=None)
        _break_the_render_probe(monkeypatch)

        payload = _payload(store, world_id, session_id)
        assert payload["lifecycle"]["state"] == LIFECYCLE_READY
        assert payload["model_state"] == MODEL_STATE_FINALIZED
        assert payload["lifecycle"]["photographic"]["state"] == (
            PHOTOGRAPHIC_NEVER_RECORDED
        )


# -- the compatibility bar --------------------------------------------------


class TestTheHistoricalWorldsAreUnchanged:

    def test_a_sparse_only_world_still_reads_saved(self, derived_world):
        """165 of the 166 worlds on this machine. They are finished."""
        store, world_id, session_id = derived_world
        _finalized(store, world_id, session_id, stages=None)
        payload = _payload(store, world_id, session_id)
        assert payload["lifecycle"]["state"] == LIFECYCLE_READY
        assert payload["model_state"] == MODEL_STATE_FINALIZED

    def test_the_photographic_block_is_additive(self, derived_world):
        """iOS decodes with `JSONSerialization` dictionaries and ignores
        unknown keys (Mac validation §5 item 10), so the block may be added
        without a decoder change -- but the fields it sits beside must not
        move."""
        store, world_id, session_id = derived_world
        _finalized(store, world_id, session_id, stages=None)
        lifecycle = _payload(store, world_id, session_id)["lifecycle"]
        for key in ("state", "evidence", "build_in_progress"):
            assert key in lifecycle, key
        assert set(lifecycle["photographic"]) == {"state", "stage", "detail"}


# -- the two judges must agree, or a world waits forever --------------------


class TestOwedMeansSomethingWillActuallyFinishIt:
    """`owed` is a PROMISE, and only one thing in the Tower keeps it.

    `photographic_state` says a world is unfinished; the phone then says
    "Improving"; and the only process that ever makes that stop being true
    is `scripts/world_finish_pending.py`, which decides what to finish with
    its own `assess()`. If this module can say `owed` about a session
    `assess()` refuses to pick up, that world says Improving **forever** --
    the exact failure the brief forbids, arrived at from the opposite
    direction to the false Saved.

    `assess()` refuses, among other things, a session whose finalization is
    not complete (`not-finalized`) or whose final solve did not succeed
    (`no-final-solve`) -- a surface needs a global solve and there is none.
    So a stage record on such a session must NOT be read as owed, however
    interrupted it looks.
    """

    @pytest.mark.parametrize("finalization", [
        # finalized, but the global solve did not succeed
        {"state": FINALIZATION_COMPLETE, "final_solve": "failed",
         "started_at": 2.0, "updated_at": 3.0, "detail": None},
        {"state": FINALIZATION_COMPLETE, "final_solve": "skipped",
         "started_at": 2.0, "updated_at": 3.0, "detail": None},
        # never finalized at all
        {"state": "interrupted", "final_solve": None,
         "started_at": 2.0, "updated_at": 3.0, "detail": None},
    ])
    def test_a_stage_record_without_a_solve_is_not_owed(
        self, derived_world, finalization
    ):
        store, world_id, session_id = derived_world
        store.append_event(
            world_id, session_id,
            WorldEvent(event_id=1, kind="session_stopped", at=2.0, payload={}),
        )
        session = store.read_session(world_id, session_id)
        store.write_session(Session(
            session_id=session_id, world_id=world_id,
            started_at=session.started_at, ended_at=session.ended_at,
            end_reason="stop", finalization=finalization,
            stages={STAGE_SURFACE: _stage(STAGE_STATE_STOPPED)},
        ))

        verdict = _state_of(store, world_id, session_id)
        assert verdict["state"] != PHOTOGRAPHIC_OWED, (
            "this session would say Improving forever: "
            "world_finish_pending.assess() refuses it for want of a solve, "
            f"so nothing will ever finish it. {verdict}"
        )
        # And the lifecycle must not park on finalizing either.
        payload = _payload(store, world_id, session_id)
        assert payload["lifecycle"]["state"] != LIFECYCLE_FINALIZING, (
            payload["lifecycle"]
        )

    def test_a_solved_finalized_session_is_still_owed(self, derived_world):
        """The guard above must not swallow the case that IS recoverable."""
        store, world_id, session_id = derived_world
        _finalized(store, world_id, session_id, stages={
            STAGE_SURFACE: _stage(STAGE_STATE_STOPPED),
        })
        assert _state_of(store, world_id, session_id)["state"] == (
            PHOTOGRAPHIC_OWED
        )
