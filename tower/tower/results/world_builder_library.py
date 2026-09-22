"""The list of saved worlds, for a viewer that wants to open an old one.

Contract: `world_builder.worlds/2026-09-10` (docs/contracts/WORLD-BUILDER-WORLDS.md).

Read-only, and deliberately thin: it is the index a phone needs to choose
a `(world_id, session_id)` pair, which the status subscription
(`result_subscribe` with `world_id`/`session_id`) and the geometry routes
already accept. It carries no geometry, no imagery and no per-keyframe
data -- those stay where they are.

`live` is answered from the world's writer lock: a lock file whose pid is
still running means a builder is writing that world right now. That is
the same liveness question `WorldStore.acquire_writer_lock` asks, and it
is asked of the OS rather than of a timestamp for the same reason.
"""

from __future__ import annotations

import logging
import math
import os

from tower.world_builder.records import FINAL_SOLVE_SOLVED, FINALIZATION_COMPLETE
from tower.world_builder.store import (
    WorldStore,
    WorldStoreError,
    manifest_describing,
    session_has_drawable_geometry,
)

logger = logging.getLogger(__name__)

WORLDS_CONTRACT = "world_builder.worlds/2026-09-10"

# Per-session `state`, the same vocabulary the status channel's lifecycle
# uses (`tower/results/world_builder.py`), minus the two words that only
# make sense against a live subscription (`idle`, `unavailable`). A row in
# the picker and the panel it opens must not disagree about what a
# session is.
SESSION_RECEIVING = "receiving"
SESSION_FINALIZING = "finalizing"
SESSION_COMPLETE = "complete"
SESSION_INTERRUPTED = "interrupted"
SESSION_UNBUILT = "unbuilt"


def _world_is_live(store: WorldStore, world_id: str) -> bool:
    holder = store.lock_holder(world_id)
    return (
        holder is not None
        and holder["alive"]
        and holder["pid"] != os.getpid()
    )


def _has_geometry(store: WorldStore, world_id: str, session_id: str, manifest) -> bool:
    """`WorldStore.session_has_drawable_geometry`, under this module's name.

    The rule itself lives in the store because the render page needs the
    same answer and the two modules must not import each other -- see
    `test_the_three_surfaces_ask_the_same_question_of_the_same_files`,
    which was written after a reviewer counted three copies of an earlier
    version of this and found them free to drift.
    """
    return session_has_drawable_geometry(store, world_id, session_id, manifest)


def _keyframes_journaled(store: WorldStore, world_id: str, session_id: str) -> int:
    """How many keyframes the journal holds, whatever the record says.

    `session.keyframes_accepted` is written at start (zero) and rewritten
    at stop; a session that never stopped keeps the zero forever. The
    journal is one line per keyframe, so counting lines is the honest
    figure and costs one sequential read.
    """
    path = store.keyframes_path(world_id, session_id)
    try:
        with path.open("rb") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


def session_state(session, *, live: bool, has_geometry: bool, manifest=None) -> str:
    """One word for what a session IS, from the record, the lock and the tree.

    Mirrors `_lifecycle` in the status producer for the facts a listing
    has (it does not compute geometry currency, so `complete` on a record
    that predates finalization means "stopped and built", not "current").

    `manifest` is the fact that made the sentence above true. Without it
    the last line read `complete if has_geometry else unbuilt`, and
    `unbuilt` is defined as "an older record that stopped and never
    built" -- so a session whose manifest PROVES a build ran, and whose
    derived tree was then deleted, was described to the wearer as one that
    never built. `_lifecycle` grew a whole `interrupted` branch to stop
    saying exactly that; this surface, which the docstring claims to
    mirror and which is the one a person chooses a walk from, kept saying
    it. Found by a reviewer building the state and reading both.

    It carries the FIGURES, not just presence, because presence alone
    cannot tell "a build lost its output" from "a build found nothing" --
    see `_nothing_to_open`.
    """
    finalization = session.finalization
    stopped = session.ended_at is not None
    if live and not stopped:
        return SESSION_RECEIVING
    if live and stopped:
        return SESSION_FINALIZING
    if not stopped:
        # Open record, nobody writing: killed mid-walk.
        return SESSION_INTERRUPTED
    if (
        finalization is not None
        and finalization.get("state") == FINALIZATION_COMPLETE
        and finalization.get("final_solve") == FINAL_SOLVE_SOLVED
        and has_geometry
    ):
        # `final_solve == solved` AS WELL, because that is the panel's
        # rule and the first version of this branch omitted it: a Tower
        # shut down mid-walk writes `complete` with `final_solve:
        # skipped`, and the picker said "Complete" over a crashed walk the
        # panel called "Interrupted". Five record shapes, all built by a
        # reviewer, all disagreeing.
        # BEFORE the end-reason check, because a completed finalization
        # outranks how the capture ended -- the status producer's own rule
        # (`world_builder.py`, "a completed finalization outranks how the
        # capture ended"), which this function claims to mirror. Leaving
        # the World Builder screen sends `session/stop` while the capture
        # is open, so the record reads `end_reason: interrupted` and then
        # finalises `complete`, solved, with geometry: the picker said
        # "Interrupted" over a session the panel called "Saved". Built by
        # a dress-rehearsal reviewer.
        return SESSION_COMPLETE
    if session.end_reason in ("error", "interrupted"):
        return SESSION_INTERRUPTED
    if finalization is not None:
        # `has_geometry` HERE TOO, not only on the line below.
        #
        # This is the surface a person chooses a walk FROM, and it said
        # "Complete" over a session whose derived tree was gone -- the
        # same hole the status producer's READY branch had, found by a
        # reviewer in the same pass, on the more damaging of the two
        # surfaces. The docstring above says this "mirrors `_lifecycle`";
        # it did not, and a claim like that is only worth what the code
        # behind it does.
        if finalization.get("state") == FINALIZATION_COMPLETE and has_geometry:
            return SESSION_COMPLETE
        if finalization.get("state") != FINALIZATION_COMPLETE:
            # The finalization itself did not finish. That IS an
            # interruption, whatever is or is not on disk.
            return SESSION_INTERRUPTED
        return _nothing_to_open(manifest)
    if has_geometry:
        return SESSION_COMPLETE
    return _nothing_to_open(manifest)


def _nothing_to_open(manifest) -> str:
    """One word for a session with no geometry, and WHICH kind of none.

    Two situations reach here and they are not the same thing, which is
    the whole reason this is a function.

    **A build recorded real figures and they are not on disk now.** Poses
    and points were deleted, or a write was lost. Something happened to
    this session, and `interrupted` says so. Reporting it as `unbuilt` --
    documented as "stopped and never built" -- describes a session whose
    own manifest proves the opposite, which is what a reviewer caught
    `session_state` doing while its docstring claimed to mirror
    `_lifecycle`.

    **A build ran and found nothing.** `engine.build` writes a derived
    tree unconditionally, so this is a real and ordinary shape: a dark
    corridor, a blank wall, a lens cap, a calibration that never arrived.
    `points.json` holds `{"points": []}`, 14 bytes, and eleven sessions on
    the real 163-world root are exactly this. NOTHING WAS INTERRUPTED
    here, and saying so tells a wearer the walk failed when it merely
    found nothing to reconstruct. `unbuilt` is the word, and iOS renders
    it **"No geometry"** -- which is the true sentence, and the one the
    panel behind the row agrees with.

    A manifest that cannot be read, or carries no figures, says nothing
    either way; the tree is empty or absent regardless, so `unbuilt`.
    """
    if isinstance(manifest, dict):
        points = manifest.get("points")
        positioned = manifest.get("poses_positioned")
        if (isinstance(points, int) and not isinstance(points, bool) and points > 0) or (
            isinstance(positioned, int)
            and not isinstance(positioned, bool)
            and positioned > 0
        ):
            return SESSION_INTERRUPTED
    return SESSION_UNBUILT


def _sortable(value):
    """A key that cannot raise, and that keeps garbage out of the way.

    Returns `(rank, comparable)`. Comparison reaches the second element
    only when the ranks are equal, and within a rank the second elements
    are always the same type, so no comparison can raise.

    **RANK 1 FOR A REAL TIMESTAMP, RANK 0 FOR EVERYTHING ELSE, and that
    ordering is the point.** The sort is `reverse=True` over
    `updated_at`, whose contract is "newest first". A first version of
    this ranked by TYPE NAME -- `("str", ...)` sorts above `("num", ...)`
    -- so a world carrying `updated_at: "2020-01-01T00:00:00Z"` was
    placed **ahead of every real world in the picker**. A reviewer built
    it against a copy of the real 163-world root and watched it take the
    top row. A malformed value is not evidence of recency and must sort
    last; under `reverse=True` that means lowest.

    `float()` IS NOT CALLED, and calling it was the other half of the
    same bug: `float(10**400)` raises `OverflowError`, out of a function
    whose docstring said it never raises and into `GET /worlds`, which
    has no handler -- a 500 losing all 163 worlds, on the same trigger
    (a tie sending Python to the second key) as the case this was written
    to fix. Python compares `int` and `float` without converting either.
    """
    if isinstance(value, bool):
        # Before the int check: `bool` IS an int, and `True` sorting among
        # timestamps as 1.0 is a silent wrong answer.
        return (0, ("bool", value))
    if isinstance(value, int) or (
        isinstance(value, float) and math.isfinite(value)
    ):
        # `int` FIRST AND WITHOUT `isfinite`. An int is finite by
        # construction, and `math.isfinite` converts to float first --
        # `math.isfinite(10**400)` raises `OverflowError`, which is the
        # very escape this function exists to prevent, reintroduced by
        # the fix for the infinities. Caught by the test written for the
        # first version of it. Python compares a big int against a float
        # exactly, without converting either.
        #
        # `math.isfinite` for the float, not `value == value`. The latter excludes NaN
        # -- which is not orderable, and which silently makes `sort`
        # return an arbitrary permutation rather than raise -- and ADMITS
        # the infinities, so `float("inf")` sorted ahead of every real
        # world under `reverse=True`: the exact outcome this function was
        # written to stop a string producing. Reachable from the Tower's
        # own writer: `json.dumps` emits the bare `Infinity` token by
        # default and `json.loads` accepts it back.
        return (1, value)
    if value is None:
        return (0, ("", ""))
    return (0, ("str", str(value)))


def _is_timestamp(value) -> bool:
    """A finite int or float, and nothing else.

    The contract (WORLD-BUILDER-WORLDS.md §2) types `started_at`,
    `ended_at`, `created_at` and `updated_at` as numbers, and the phone's
    decoder holds it to that: `WorldLibrary.swift` decodes the listing
    as ONE value, so a single row carrying a string where a number was
    promised fails the whole decode and every world vanishes from Saved
    Worlds. Not one row -- all of them. The record readers do not coerce
    (`session_from_json_dict` is `started_at=data["started_at"]`, raw),
    so what a `session.json` says is what would be served.

    `bool` is refused before the int check because `bool` IS an int. An
    int too big for a double is refused too: `float(10**400)` raises
    `OverflowError`, and a Double decoder cannot take it either.
    """
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        try:
            return math.isfinite(float(value))
        except OverflowError:
            return False
    return isinstance(value, float) and math.isfinite(value)


def build_world_listing(store: WorldStore) -> dict:
    """Every world with a readable `world.json`, newest first, with its
    sessions oldest first. A world whose sessions cannot be read is listed
    with what could be read; a world that cannot be read at all is left out
    rather than invented.

    A record whose timestamps are not the numbers the contract promises
    counts as unreadable (see `_is_timestamp`): it is omitted, with a
    warning naming it, rather than served raw for the phone to choke on."""
    worlds = []
    for world_id in store.list_world_ids():
        try:
            world = store.read_world(world_id)
        except (WorldStoreError, OSError, ValueError, KeyError):
            continue
        if not (_is_timestamp(world.created_at) and _is_timestamp(world.updated_at)):
            logger.warning(
                "[Tower][Worlds] world %s has created_at=%r updated_at=%r; the "
                "listing contract types both as numbers and the phone rejects "
                "the whole listing on one such row, so this world is omitted "
                "like an unreadable one",
                world_id, world.created_at, world.updated_at,
            )
            continue
        live = _world_is_live(store, world_id)
        sessions = []
        for session_id in store.list_session_ids(world_id):
            try:
                session = store.read_session(world_id, session_id)
            except (WorldStoreError, OSError, ValueError, KeyError):
                continue
            if not (
                _is_timestamp(session.started_at)
                and (session.ended_at is None or _is_timestamp(session.ended_at))
                # The other field the phone's decoder REQUIRES from a row;
                # `has_geometry` is computed here and always a bool.
                and isinstance(session.frame_source, str)
            ):
                logger.warning(
                    "[Tower][Worlds] world %s session %s has started_at=%r "
                    "ended_at=%r frame_source=%r; the listing contract types "
                    "the first two as numbers (`ended_at` may be null) and the "
                    "third as a string, and the phone rejects the whole "
                    "listing on one such row, so this session is omitted like "
                    "an unreadable one",
                    world_id, session_id, session.started_at, session.ended_at,
                    session.frame_source,
                )
                continue
            # The SAME manifest rule the status producer and the geometry
            # route use -- the session's own copy, then the world's but
            # only if it names this session. Four readers, one rule; a
            # fifth reader with its own idea is how this campaign's
            # defects kept coming back.
            # `purpose="figures"`, NOT the identity rule the geometry
            # module uses. Everything this listing does with a manifest
            # reads COUNTS out of it -- `has_geometry` and
            # `_nothing_to_open` -- and a manifest whose fields this build
            # cannot vouch for must not supply them. `manifest_for` is the
            # identity question and answers a different one.
            manifest = manifest_describing(
                store, world_id, session_id, purpose="figures"
            )
            has_geometry = _has_geometry(store, world_id, session_id, manifest)
            sessions.append({
                "session_id": session.session_id,
                "started_at": session.started_at,
                "ended_at": session.ended_at,
                "end_reason": session.end_reason,
                "frame_source": session.frame_source,
                "capture_id": session.capture_id,
                "keyframes_accepted": session.keyframes_accepted,
                # The journal's own count, beside the record's. On a record
                # that never stopped the record says zero and the journal
                # says what actually landed (467 on the 2026-09-06 walk).
                "keyframes_journaled": _keyframes_journaled(store, world_id, session_id),
                "has_geometry": has_geometry,
                # The record is only finalised by `stop_session`; a builder
                # killed before that (the supervisor's shutdown grace, a
                # hard kill) leaves `ended_at: null` behind forever. Open
                # with nobody writing is not "still open", and the phone
                # would otherwise say exactly that. Counts on such a record
                # are the start-of-session values, not the journal length.
                "abandoned": session.ended_at is None and not live,
                # One word, the status channel's vocabulary (additive,
                # 2026-09-06): receiving | finalizing | complete |
                # interrupted | unbuilt.
                "state": session_state(
                    session,
                    live=live,
                    has_geometry=has_geometry,
                    manifest=manifest,
                ),
                # The builder's own account of how finalization went, or
                # null on a record written before it existed.
                "finalization": session.finalization,
            })
        # `_sortable` HERE TOO, and its absence here was the whole
        # argument for it thirty lines below.
        #
        # `session_from_json_dict` does not coerce -- `started_at =
        # data["started_at"]`, raw -- so a `session.json` carrying a
        # string or a null reaches this comparison. This sort is at the
        # top of the per-world loop, OUTSIDE the inner try that skips an
        # unreadable session, outside `build_world_listing`'s only
        # handler, on a route with none: the raise escapes before any
        # world is returned, so **one bad session record empties Saved
        # Worlds entirely** -- all 163 worlds, not a shortened list.
        #
        # A reviewer reproduced it by writing an ISO timestamp string into
        # one `started_at`, immediately after the round that hardened the
        # world sort against the identical shape and did not look up.
        #
        # Since the `_is_timestamp` check above, no such row reaches this
        # sort from this function. `_sortable` stays because its promise
        # -- a key that cannot raise -- is the sort's own, and the two
        # guards fail differently: one keeps a bad row out of the phone's
        # decoder, the other keeps a bad row from taking every good one
        # down with it.
        sessions.sort(key=lambda s: _sortable(s["started_at"]))
        worlds.append({
            "world_id": world.world_id,
            "display_name": world.display_name,
            "created_at": world.created_at,
            "updated_at": world.updated_at,
            "live": live,
            "session_count": len(sessions),
            "sessions": sessions,
        })
    # A TOTAL ORDER, not just a key. Windows' clock granularity is about
    # 15.6 ms, so two worlds created in one tick share an `updated_at` --
    # and a sort on that alone leaves their order to whatever
    # `list_world_ids` happened to yield, which can differ between polls.
    # The phone redraws this list every time it arrives, so a tie makes
    # rows swap places under the wearer's finger.
    #
    # Found as a suite flake (`assert 1 < 0` on two worlds created
    # back-to-back) under a loaded machine, which is the same tie.
    # `created_at` breaks most of them and the id breaks the rest; the id
    # is arbitrary but it is STABLE, which is the property that matters.
    #
    # `_sortable` IS NOT DEFENSIVE CLUTTER. `created_at` is a required,
    # un-defaulted field and `world_from_json_dict` subscripts it, so a
    # world missing it never reaches here -- but a world carrying `null`,
    # or a string, does. Comparing that to a float raises `TypeError`
    # only when a tie on `updated_at` makes Python look at the second key.
    # This sort is outside `build_world_listing`'s try/except and
    # `routes/geometry.py` has no handler, so the raise is a 500 on
    # `GET /worlds`: one malformed row and the picker loses every world.
    #
    # CORRECTION. An earlier version of this comment called that a
    # regression introduced by the tiebreak, "which is why the single-key
    # sort this replaced could not reach it". The sort this replaced was
    # not single-key -- `git show e60d753` has the same three -- and a
    # reviewer ran the old key against the same input and got the same
    # `TypeError`. It is an old wart, and the fix is the same fix; only
    # the story about where it came from was wrong.
    # A reviewer produced both shapes. `_sortable` keeps every
    # comparison within one type, and puts anything that is not a real
    # timestamp LAST rather than merely somewhere -- see there for why
    # "arbitrary but total" was not enough for the primary key.
    worlds.sort(
        key=lambda w: (
            _sortable(w["updated_at"]),
            _sortable(w["created_at"]),
            _sortable(w["world_id"]),
        ),
        reverse=True,
    )
    return {"contract": WORLDS_CONTRACT, "world_count": len(worlds), "worlds": worlds}
