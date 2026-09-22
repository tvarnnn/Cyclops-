"""Does this session still OWE a photographic room, and did it get one?

WHY A SECOND QUESTION EXISTS BESIDE "IS A BUILD RUNNING".

The serving path used to ask exactly one present-tense question -- *is a
photographic stage running right now?* -- and answer the wearer's question
from it: running means "Improving", not running means "Saved". That is the
wrong question, and the Mac/iOS validation of 2026-09-22 caught it being
wrong in three different ways at once (§7, T2/T3/T4):

  * **T2, the flicker.** "Running" is false in the gaps BETWEEN stages. The
    builder releases the world lock, assembles a report, imports the
    reconstruction stack, and only then writes the surface's first
    `running` status; and between the surface's `ok` and the appearance's
    first `running` status. Both gaps are SHORT -- measured, the second is sub-millisecond,
    not the "tens of seconds" an earlier draft of this claimed -- and
    that changes nothing: a 2 Hz poll lands in them, and a builder that
    dies in either window leaves a world that reads finished for ever. For those polls the
    Tower said ready and the phone said **Saved**, then went back to
    Improving. A wearer told to wait for "Saved" can be told it twice.

  * **T3, the false success.** A stage that FAILED, or was stopped, or was
    never attempted, is also not running. All of them read "Saved" too, so
    "the photographic build failed" was indistinguishable from "the
    photographic room is ready".

  * **T4, failing open.** When the probe itself broke, the code carefully
    computed a reason, put it in `build_in_progress_unavailable_reason`,
    and left `state` at ready -- and the contract says iOS does not read
    that field (`WORLD-BUILDER-IOS.md` §2.4, "Not consumed"). A broken
    probe rendered as **Saved**.

All three are the same mistake. "A process is working this millisecond" is
a fact about a PROCESS; "this world is finished" is a fact about the WORLD.
The second is what the wearer is asking, it is settled rather than
instantaneous, and it is already written down.

WHAT THIS MODULE ANSWERS INSTEAD

`photographic_state()` returns one word from a closed vocabulary describing
where the photographic representation has got to, computed from the
session's own stage record first and the stages' own artifacts second --
the SAME precedence, and the same two signals, that
`scripts/world_finish_pending.assess()` uses to decide what to finish. That
is deliberate: the tool that BUILDS the missing work and the channel that
REPORTS it must not be able to disagree about which worlds are missing it.

THE HISTORICAL WORLDS ARE THE HARD PART, and the reason this is not simply
"anything without an appearance is unfinished". There are 166 worlds on the
machine this was written for, and 69 of their 70 sessions were built by a
Tower that had no photographic stages at all. They are not broken, they are
not owed anything, and relabelling them would turn one honest bug report
into sixty-nine false ones.

MEASURED, because an earlier version of this paragraph guessed and was
wrong: of those 70 sessions, **67 reach `UNATTEMPTED`** (they predate the
finalization record, so they have no completed global solve to build from)
and **2 reach `NEVER_RECORDED`**; the 1 remaining is the recovered world,
`COMPLETE`. Both of the first two words are settled and map to ready exactly
as before -- which is why no old world changes a word -- but they are not
the same word, and the next person to edit either mapping should be
reasoning from the real numbers rather than from this comment's first
draft.

The distinction is sound in both directions because of an ordering the
builder now guarantees: `world_build_session.py` records the surface stage
BEFORE it releases the world lock, so a world built by current code has a
`stages` block from the first instant there is anything to be wrong about.
Absent means old, not "new and early".
"""

from __future__ import annotations

import logging

from tower.world_builder.records import (
    FINAL_SOLVE_SOLVED,
    FINALIZATION_COMPLETE,
    STAGE_APPEARANCE,
    STAGE_STATE_FAILED,
    STAGE_STATE_OK,
    STAGE_STATE_RUNNING,
    STAGE_STATE_STOPPED,
    STAGE_STATE_UNAVAILABLE,
    STAGE_SURFACE,
)

logger = logging.getLogger(__name__)

# The stages that make the photographic room. `dense` is deliberately not
# here: it is an optional enrichment, and a Tower with it switched off is
# not a Tower that owes anything. Same tuple, same reason, as
# `scripts/world_finish_pending.PHOTOGRAPHIC_STAGES`.
PHOTOGRAPHIC_STAGES = (STAGE_SURFACE, STAGE_APPEARANCE)

# The closed vocabulary. Every word is a DIFFERENT THING TO SAY TO A WEARER,
# which is the test a candidate word has to pass to be in here.
#
# The photographic room exists and is servable.
PHOTOGRAPHIC_COMPLETE = "complete"
# A process is working on it right now, on evidence of a live pid.
PHOTOGRAPHIC_RUNNING = "running"
# It is unfinished and nothing is working on it. Recoverable: this is what
# `scripts/world_finish_pending.py` picks up at the next Tower start.
PHOTOGRAPHIC_OWED = "owed"
# A stage ran and failed. NOT recoverable by simply waiting -- the wearer
# should be told, because waiting is what they would otherwise do.
PHOTOGRAPHIC_FAILED = "failed"
# The stages ran and declined: no solve to build from, the Tower's
# appearance setting is off. Nothing is wrong and nothing is coming.
PHOTOGRAPHIC_UNATTEMPTED = "unattempted"
# A world from before the photographic stages existed. Saved, and honest.
PHOTOGRAPHIC_NEVER_RECORDED = "never_recorded"
# The probe broke. NOT the same as "nothing is owed", and the entire point
# of T4 is that it must never be collapsed into one.
PHOTOGRAPHIC_UNOBSERVABLE = "unobservable"

PHOTOGRAPHIC_STATES = (
    PHOTOGRAPHIC_COMPLETE,
    PHOTOGRAPHIC_RUNNING,
    PHOTOGRAPHIC_OWED,
    PHOTOGRAPHIC_FAILED,
    PHOTOGRAPHIC_UNATTEMPTED,
    PHOTOGRAPHIC_NEVER_RECORDED,
    PHOTOGRAPHIC_UNOBSERVABLE,
)

# The words that mean "this session is not finished with the photographic
# stages, and something should still happen". A world in one of these is
# never `ready`, whatever any present-tense probe says -- that is the T2
# fix, and it is a property of the SET rather than of any one call site.
UNSETTLED_STATES = (
    PHOTOGRAPHIC_RUNNING,
    PHOTOGRAPHIC_OWED,
    PHOTOGRAPHIC_UNOBSERVABLE,
)

# The states a stage record can be in that mean the stage did not reach an
# end of its own. Same two words `world_finish_pending.INTERRUPTED_STATES`
# selects on, for the same reason.
_INTERRUPTED_RECORD_STATES = (STAGE_STATE_RUNNING, STAGE_STATE_STOPPED)


class _Unobservable(Exception):
    """A probe could not answer. Carries the sentence a client may read."""


def _stage_entry(session, stage):
    stages = getattr(session, "stages", None) or {}
    return stages.get(stage) or {}


def _appearance_is_expected(session) -> bool:
    """Whether a photographic room is a thing this session should have.

    THESE ARE `assess()`'S OWN REFUSALS, and they are here so the two cannot
    disagree about which worlds are missing work. Saying `owed` puts the
    phone on "Improving", and the only process that ever makes that stop
    being true is `scripts/world_finish_pending.py`; a shape it refuses is a
    world that would wait for ever.

    * `never-stopped` -- a walk still in progress, or a builder that died
      mid-walk. The photographic stages are not what it is missing, and
      `world_finalize.py` is the tool for the second shape. Added after a
      reviewer found a session with a `session_stopped` journal event and
      `ended_at: null` -- a crash between the two writes -- reading `owed`
      on one surface and `interrupted` on the other, with the finisher
      silent.
    * `not-finalized` / `no-final-solve` -- `final_surface_stages` refuses
      without a solve and records `unavailable`. Queuing it would be asking
      for work that is defined not to happen.
    """
    if getattr(session, "ended_at", None) is None:
        return False
    finalization = getattr(session, "finalization", None) or {}
    return (
        finalization.get("state") == FINALIZATION_COMPLETE
        and finalization.get("final_solve") == FINAL_SOLVE_SOLVED
    )


def _live_stage(store, world_id, session_id):
    """The photographic stage a LIVE process is working on, or None.

    Delegates to the render module's `_stage_running`, which is the same
    per-stage predicate `session_build_running` is made of -- staleness,
    the dead-pid rule and the published-manifest rule together. Two answers
    to one question is how they come to disagree, so there is one.

    Raises `_Unobservable` rather than returning False when it cannot tell.
    Returning False would be the T4 bug: a broken probe asserting that
    nothing is running.
    """
    try:
        # Imported at call time, like every other consumer in the serving
        # path: the render module reaches into the surface and dense
        # pipelines, and putting those in the import graph of every status
        # poll would cost every poll for a probe most polls never make.
        from tower.results.world_builder_render import (  # noqa: PLC0415
            _stage_running,
        )
        from tower.results.world_builder import (  # noqa: PLC0415
            _stage_staleness_probe,
        )
    except Exception as exc:  # noqa: BLE001 -- reported, never swallowed
        raise _Unobservable(
            f"the photographic build probe could not be loaded: "
            f"{type(exc).__name__}"
        ) from exc

    try:
        world_dir = store.world_dir(world_id)
    except Exception as exc:  # noqa: BLE001
        raise _Unobservable(
            f"the world directory could not be read: {type(exc).__name__}"
        ) from exc

    for stage in PHOTOGRAPHIC_STAGES:
        status_path = world_dir / stage / session_id / "status.json"
        try:
            if _stage_running(status_path, _stage_staleness_probe(stage)):
                return stage
        except Exception as exc:  # noqa: BLE001 -- this stage only
            raise _Unobservable(
                f"the {stage} liveness probe failed: {type(exc).__name__}"
            ) from exc
    return None


def _interrupted_stage_on_disk(store, world_id, session_id):
    """A photographic stage whose own artifact says it was interrupted.

    The second signal, for sessions with no stage record. It cannot widen
    the net onto historical worlds: `<world>/<stage>/<session>/status.json`
    is written by the pipelines and by nothing else, so a Tower that never
    ran a photographic stage cannot have left one behind. Measured when
    this was written: 166 worlds on the machine, ONE with a `surface/`
    directory, and it was the interrupted one.
    """
    from tower.storage import read_json_closed  # noqa: PLC0415
    from tower.world_builder.surface_pipeline import (  # noqa: PLC0415
        status_is_stale,
    )

    world_dir = store.world_dir(world_id)
    for stage in PHOTOGRAPHIC_STAGES:
        path = world_dir / stage / session_id / "status.json"
        try:
            status = read_json_closed(path)
        except Exception:  # noqa: BLE001 -- an unreadable status is not a claim
            continue
        if not status:
            continue
        state = status.get("state")
        if state == STAGE_STATE_STOPPED:
            return stage
        if state == STAGE_STATE_RUNNING and status_is_stale(status):
            return stage
    return None


def _liveness(store, world_id, session_id, stage_hint):
    """`RUNNING`, `OWED` or `UNOBSERVABLE` for a stage that has not settled.

    Consulted ONLY where it can change the answer -- which is the whole
    reason this is a separate function. An earlier draft asked liveness
    first, for every session, and a reviewer pointed out what that does
    when the probe itself is broken: all 165 historical worlds on this
    machine would have gone `UNOBSERVABLE` at once, and a conservative
    mapping would then have parked every one of them on "Improving"
    forever. The brief forbids that as explicitly as it forbids the false
    "Saved", and both failures come from asking a question whose answer
    does not matter here.

    So the probe is asked exactly when the RECORD is ambiguous: a stage
    that is `running` or `stopped` is either being worked on right now or
    abandoned, and nothing but liveness tells those apart.
    """
    try:
        live = _live_stage(store, world_id, session_id)
    except _Unobservable as exc:
        # THE PROBE BROKE, on a session that demonstrably has an unfinished
        # photographic stage. "Nothing is running" would be the T4 claim,
        # asserted on no evidence, and it is the one answer that renders as
        # "Saved".
        return {
            "state": PHOTOGRAPHIC_UNOBSERVABLE,
            "stage": stage_hint,
            "detail": str(exc),
        }
    if live is not None:
        return {
            "state": PHOTOGRAPHIC_RUNNING,
            "stage": live,
            "detail": f"the {live} stage is running under a live process",
        }
    return {
        "state": PHOTOGRAPHIC_OWED,
        "stage": stage_hint,
        "detail": (
            f"the {stage_hint} stage is unfinished and no process is "
            "working on it"
        ),
    }


def photographic_state(store, world_id: str, session_id: str, session) -> dict:
    """Where this session's photographic representation has got to.

    Returns `{"state": <one of PHOTOGRAPHIC_STATES>, "stage": str | None,
    "detail": str}`. Never raises: a probe that fails becomes
    `UNOBSERVABLE` with its reason, because the one outcome this function
    must not have is an exception on the status path quietly becoming a
    reassuring word.

    PRECEDENCE. The RECORD first and the ARTIFACT second -- the same two
    signals in the same order as `scripts/world_finish_pending.assess()`,
    so the tool that BUILDS the missing work and the channel that REPORTS
    it cannot disagree about which worlds are missing it. Liveness is asked
    only where the record is ambiguous; see `_liveness`.
    """
    stages = getattr(session, "stages", None) or {}

    if stages:
        appearance = _stage_entry(session, STAGE_APPEARANCE)

        # COMPLETE is the APPEARANCE's word, never the surface's. A surface
        # with no shading on it is a grey mesh, and the product invariant is
        # the photographic room: "a saved world is the surface AND the
        # shading on it", as `world_finish_pending._retire` puts it.
        if appearance.get("state") == STAGE_STATE_OK:
            return {
                "state": PHOTOGRAPHIC_COMPLETE,
                "stage": STAGE_APPEARANCE,
                "detail": "the appearance stage finished",
            }

        # FAILED BEATS OWED. A stage that ran and raised is not going to be
        # fixed by waiting, and telling a wearer to wait is what "Improving"
        # does. This is the half of T3 that matters most: before it, a
        # failed photographic build was indistinguishable from a finished
        # one.
        for stage in PHOTOGRAPHIC_STAGES:
            entry = _stage_entry(session, stage)
            if entry.get("state") == STAGE_STATE_FAILED:
                return {
                    "state": PHOTOGRAPHIC_FAILED,
                    "stage": stage,
                    "detail": (
                        entry.get("detail")
                        or f"the {stage} stage is recorded failed"
                    ),
                }

        for stage in PHOTOGRAPHIC_STAGES:
            entry = _stage_entry(session, stage)
            if entry.get("state") in _INTERRUPTED_RECORD_STATES:
                # OWED IS A PROMISE, AND ONLY ONE THING KEEPS IT. Saying
                # `owed` puts the phone on "Improving", and the only process
                # that ever makes that stop being true is
                # `scripts/world_finish_pending.py` -- which decides what to
                # pick up with its own `assess()`, and refuses a session
                # whose finalization is incomplete (`not-finalized`) or
                # whose global solve did not succeed (`no-final-solve`): a
                # surface needs a solve and there is none.
                #
                # So an interrupted stage record on such a session must not
                # be read as owed. It would be a world that says "Improving"
                # FOREVER, which is the same failure as the false "Saved"
                # arrived at from the other side, and the brief forbids both.
                # The two judgements are kept in agreement here, by
                # construction, rather than by both being edited the same
                # way on some later day.
                if not _appearance_is_expected(session):
                    return {
                        "state": PHOTOGRAPHIC_UNATTEMPTED,
                        "stage": stage,
                        "detail": (
                            f"the {stage} stage did not finish, and this "
                            "session has no completed global solve to build a "
                            "photographic room from, so nothing will retry it"
                        ),
                    }
                # The ambiguous case, and the ONLY one that needs a probe.
                # `running` in the record with nothing alive is an
                # interrupted stage, not a running one.
                return _liveness(store, world_id, session_id, stage)

        # `unavailable` IS TWO DIFFERENT FACTS AND `attempted` TELLS THEM
        # APART. An adversarial review caught the first version of this
        # module filing both under "nothing is wrong and nothing is coming",
        # which is T3 re-entering through a word the vocabulary did not
        # enumerate. The pipelines write `unavailable` for a stage that was
        # never asked for -- the appearance switched off, no solve to build
        # from, the surface not `ok` -- with `attempted: False`; and for a
        # stage that RAN and could not produce -- an ASTC encode that
        # returned the wrong byte count, a WebP encode that failed, open3d
        # missing, a surface artifact shorter than its header -- with
        # `attempted: True`. The second is a photographic build that broke,
        # and telling the wearer "Saved" over it is exactly the lie this
        # module exists to stop.
        #
        # Both are terminal either way: `world_finish_pending` retries only
        # `running` and `stopped`, so neither is `owed` and neither must say
        # "Improving". They differ in what is TRUE, and so in what is said.
        for stage in PHOTOGRAPHIC_STAGES:
            entry = _stage_entry(session, stage)
            if (
                entry.get("state") == STAGE_STATE_UNAVAILABLE
                and entry.get("attempted")
            ):
                return {
                    "state": PHOTOGRAPHIC_FAILED,
                    "stage": stage,
                    "detail": (
                        entry.get("detail")
                        or f"the {stage} stage ran and could not produce a "
                           "photographic representation"
                    ),
                }

        # Every photographic stage reached an end of its own, none is `ok`,
        # and none of them was even attempted. Nothing is wrong and nothing
        # is coming.
        return {
            "state": PHOTOGRAPHIC_UNATTEMPTED,
            "stage": None,
            "detail": (
                appearance.get("detail")
                or "no photographic stage was attempted for this session"
            ),
        }

    # NO RECORD. Either a world from before the stages existed, or the
    # 2026-09-22 shape: a session interrupted by a Tower that wrote a
    # `status.json` and was shut down before it could write a record. The
    # second signal exists for that one world, and cannot widen the net --
    # `<world>/<stage>/<session>/status.json` is written by the pipelines
    # and by nothing else.
    try:
        stage = _interrupted_stage_on_disk(store, world_id, session_id)
    except Exception as exc:  # noqa: BLE001
        return {
            "state": PHOTOGRAPHIC_UNOBSERVABLE,
            "stage": None,
            "detail": (
                f"the photographic stage artifacts could not be read: "
                f"{type(exc).__name__}"
            ),
        }
    if stage is not None:
        # Same obligation as the record branch above: an interrupted
        # artifact on a session `assess()` will not pick up is not owed.
        if not _appearance_is_expected(session):
            return {
                "state": PHOTOGRAPHIC_UNATTEMPTED,
                "stage": stage,
                "detail": (
                    f"the {stage} stage left an interrupted status.json, but "
                    "this session has no completed global solve to build a "
                    "photographic room from, so nothing will retry it"
                ),
            }
        return _liveness(store, world_id, session_id, stage)

    if not _appearance_is_expected(session):
        return {
            "state": PHOTOGRAPHIC_UNATTEMPTED,
            "stage": None,
            "detail": (
                "this session has no finished global solve to build a "
                "photographic room from"
            ),
        }

    # THE HISTORICAL WORLD, and the one that must not move. 165 of the 166
    # worlds on the machine this was written for land here, and they reach
    # it without the probe being asked at all.
    return {
        "state": PHOTOGRAPHIC_NEVER_RECORDED,
        "stage": None,
        "detail": (
            "no Tower ever ran a photographic stage for this session, which "
            "is not the same as one that tried and failed"
        ),
    }


def is_unsettled(state: str) -> bool:
    """Whether this word means the world is not finished being made.

    The single place the T2 rule lives: a world in an unsettled state is
    never reported ready, whether or not a process happens to be running in
    the instant the poll arrives.
    """
    return state in UNSETTLED_STATES
