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

from tower.world_builder.records import (
    FINAL_SOLVE_SOLVED,
    FINALIZATION_COMPLETE,
    FINALIZATION_PENDING,
)
# The settled photographic vocabulary, at module scope for the same reason
# `results/world_builder.py` imports it at module scope: `photographic`
# pulls in nothing but `records`, which this module already imports on the
# line above, so it costs no import graph. The PROBES inside it are the
# expensive part and they are imported at call time, there.
from tower.world_builder.photographic import (
    PHOTOGRAPHIC_UNOBSERVABLE,
    is_unsettled,
)
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


def lock_speaks_for(session) -> bool:
    """Whether the WORLD's writer lock is evidence about THIS session.

    The lock is per world; a session is written by at most one holder, and
    only while its record is open or its finalization is still pending.
    Outside that window the lock belongs to somebody else's work -- a walk
    into a new session of the same world, or the recovery finisher on a
    sibling session -- and reading it as this session's "finalizing" told the
    wearer a finished, or historical, walk was being improved (review,
    2026-09-23: with the finisher now running at every idle moment, that was
    six to sixteen minutes of false "Improving" on each sibling). The same
    rule `world_builder_render.session_build_running` already applies.

    A session with NO finalization record keeps the lock as its evidence: a
    builder that writes none (the pre-2026-09-06 builder, a replay) holds the
    lock through its finalization and has nothing else to show it. The lock
    is set aside only where the session's own record says its finalization
    SETTLED -- `complete` or `interrupted` -- because then the lock is
    somebody else's.
    """
    finalization = getattr(session, "finalization", None)
    return (
        getattr(session, "ended_at", None) is None
        or not finalization
        or finalization.get("state") == FINALIZATION_PENDING
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


def _dense_summary(store: WorldStore, world_id: str, session_id: str) -> dict | None:
    """What dense reconstruction this session has, or None.

    ADDITIVE, and the contract identifier deliberately does not move. iOS
    parses these payloads with `JSONSerialization` into `[String: Any]` and
    reads them key by key, so a key it does not know is a key it never looks
    at -- but it equality-tests `contract` on the first line of every guard, so
    bumping that would empty the gallery on every older build. A world with no
    dense artifact reports `null` and behaves exactly as it does today.
    """
    from tower.world_builder.dense_pipeline import (  # noqa: PLC0415
        dense_currency,
        read_dense_manifest,
    )

    manifest = read_dense_manifest(store, world_id, session_id)
    if not manifest:
        return None
    levels = manifest.get("levels") or []
    canonical = manifest.get("canonical_level", 0)
    mobile = manifest.get("mobile_level", len(levels) - 1)
    return {
        "format": manifest.get("format"),
        "levels": len(levels),
        "canonical_points": (levels[canonical]["points"]
                             if canonical < len(levels) else None),
        "mobile_points": (levels[mobile]["points"] if mobile < len(levels) else None),
        # Repeated from the manifest rather than re-derived. The dense stage
        # makes no scale claim the sparse solve did not already make.
        "scale": manifest.get("scale"),
        # Whether this cloud was fused against the solve now on disk. False
        # after a re-solve; None when it cannot be known, which is not the
        # same thing and must not be shown as staleness. The render page
        # carries the same fact as a caption; this is so a gallery can mark it
        # without fetching an 8 MB page.
        "solve_current": dense_currency(
            store, world_id, session_id, manifest,
            include_derived=False).get("solve_current"),
    }


APPEARANCE_IMAGERY_NOTE = (
    "first-person keyframe imagery of a private space; best-effort face redaction with "
    "measured false negatives; not anonymised; screens, documents and bodies are not redacted"
)
APPEARANCE_RETENTION_NOTE = (
    "kept with the world under appearance/<session>/ until the session is rebuilt or the "
    "world is purged; derived from the session keyframes, so it is deleted and rebuilt, "
    "never edited"
)


def _appearance_summary(store: WorldStore, world_id: str, session_id: str, world) -> dict | None:
    """The appearance artifact as IMAGERY (WORLD-BUILDER-WORLDS.md §2, privacy
    lane §3.4; review 1, m6), or None when the session has none.

    Reported whether or not it is served now, because it is on disk either
    way: `state` says whether the routes serve it (`served`), would once the
    final build lands (`rebuilding`), or have withdrawn it (`withdrawn`). The
    label, the effective label, the privacy tags and the retention are the
    artifact's own record. Never a URL or a path: the page fetches it by the
    routes of §4b, and only for a session the Tower serves.
    """
    from tower.world_builder import appearance_pipeline as AP  # noqa: PLC0415
    from tower.world_builder import raw_imagery as RAWIMG  # noqa: PLC0415

    manifest = AP.read_appearance_manifest(store, world_id, session_id)
    if not manifest:
        return None
    prov = manifest.get("appearance_provenance") or {}
    if getattr(world, "images_purged", False):
        state = AP.WITHDRAWN
    elif AP.label_matches(store, world_id, session_id, manifest):
        state = AP.SERVED
    else:
        state = AP.withdrawal_state(store, world_id, session_id, manifest)
    keyframes = manifest.get("keyframes") or []
    return {
        "format": manifest.get("format"),
        "state": state,
        "quality": manifest.get("quality"),
        "keyframes": len(keyframes),
        "keyframes_phone": sum(1 for k in keyframes if k.get("tier") == "phone"),
        "bytes": sum(AP.named_files(manifest).values()),
        # §6.6, first and at the top level: the listing must never present a
        # research build as the product. `imagery` below describes redacted
        # imagery, so it is replaced outright rather than qualified.
        "imagery_source": AP.imagery_source_of(manifest),
        "privacy_safe": AP.imagery_source_of(manifest) == RAWIMG.IMAGERY_REDACTED,
        "redaction": prov.get("session_redaction"),
        "redaction_effective": prov.get("redaction_effective"),
        "label_trusted": prov.get("label_trusted"),
        "keyframe_image_set": prov.get("keyframe_image_set"),
        "privacy_tags": list(prov.get("privacy_tags") or []),
        "retains_raw_imagery": prov.get("retains_raw_imagery"),
        "imagery": (RAWIMG.RAW_NOTE
                    if AP.imagery_source_of(manifest) != RAWIMG.IMAGERY_REDACTED
                    else APPEARANCE_IMAGERY_NOTE),
        "retention": APPEARANCE_RETENTION_NOTE,
    }


def _appearance_summary_or_none(store: WorldStore, world_id: str, session_id: str, world):
    """`_appearance_summary`, or None when it cannot be read: a report must
    never take the session's row (or the listing) down with it."""
    try:
        return _appearance_summary(store, world_id, session_id, world)
    except Exception:  # noqa: BLE001
        logger.debug("[Tower][WorldBuilder] appearance summary failed for %s/%s",
                     world_id, session_id, exc_info=True)
        return None


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


def _photographic_build_running(store: WorldStore, world_id: str,
                                session_id: str) -> bool:
    """Whether a photographic stage is working on this session right now.

    THE STATUS PRODUCER'S OWN PROBE, imported rather than re-derived, because
    a second copy of this predicate is how the two surfaces drift -- and the
    drift is not cosmetic here. Measured after the status producer learned
    this and the listing had not: the panel said `finalizing` / "Improving"
    while the Saved Worlds row for the SAME session said `complete`. A wearer
    who opens the picker reads "Complete" and shuts the Tower down, which is
    the precise failure the whole change exists to prevent, told on the
    surface a person actually chooses a walk from.

    Imported at call time. `results/world_builder` imports `_sortable` from
    THIS module at module scope, so the reverse cannot be a module-scope
    import; `session_build_running` reaches back into this module the same
    way, for the same reason.

    Never raises: a probe that cannot read something answers "not building",
    which is what this module said before it asked at all.
    """
    try:
        from tower.results.world_builder import (  # noqa: PLC0415
            _photographic_build_evidence,
        )

        evidence, unobservable = _photographic_build_evidence(
            store, world_id, session_id
        )
        if unobservable is not None:
            logger.debug("[Tower][Worlds] %s/%s: %s", world_id, session_id,
                         unobservable)
        return evidence is not None
    except Exception:  # noqa: BLE001 -- a liveness probe must not 500 a listing
        # FAILING OPEN IS SAFE HERE NOW, AND IT WAS NOT BEFORE. This used to
        # be the row's only source of "something is still working on it", so
        # `False` was a row saying `complete` over a world mid-build -- the
        # warning that stood here said exactly that about itself. The row's
        # authority is now `_photographic_state_for_row`, which answers from
        # the RECORD and turns its own probe failures into the
        # `unobservable` WORD rather than into silence. So a raise here only
        # loses the sharper present-tense sentence, never the correction.
        logger.debug(
            "[Tower][Worlds] photographic build probe raised for %s/%s; the "
            "row's word comes from the settled photographic state instead",
            world_id, session_id, exc_info=True,
        )
        return False


def _photographic_state_for_row(store: WorldStore, world_id: str,
                                session_id: str, session) -> dict:
    """Where this session's photographic room has got to -- THE ROW'S AUTHORITY.

    `tower.world_builder.photographic.photographic_state`, wrapped exactly
    as `results/world_builder._photographic_state_or_none` wraps it, and for
    the same reason: one module answers the question, two surfaces ask it,
    and neither may be taken down by it.

    WHY THIS REPLACED THE PRESENT-TENSE PROBE AS THE AUTHORITY. The row used
    to be decided by `_photographic_build_running` alone -- "is a stage
    running this millisecond" -- and the Mac/iOS validation of 2026-09-22
    (§7, T3) caught all three ways that is the wrong question, on THIS
    surface: a stage that FAILED is not running, a stage that is OWED is not
    running, and a probe that BROKE reported not running. All three read
    `complete` in the picker, which is the surface a person chooses a walk
    from and then shuts the Tower down. `photographic_state` answers the
    settled question instead, from the session's own stage record first, so
    it is true across the gaps between stages and it cannot be flipped by a
    broken probe.

    Never raises, and never returns None: a row with no answer is a row that
    keeps saying `complete`, which is the failure being fixed.
    """
    try:
        from tower.world_builder.photographic import (  # noqa: PLC0415
            photographic_state,
        )

        return photographic_state(store, world_id, session_id, session)
    except Exception:  # noqa: BLE001 -- reported, never swallowed
        logger.warning(
            "[Tower][Worlds] the photographic state could not be computed "
            "for %s/%s; this row will not be reported finished",
            world_id, session_id, exc_info=True,
        )
        return {
            "state": PHOTOGRAPHIC_UNOBSERVABLE,
            "stage": None,
            "detail": "the photographic state could not be computed",
        }


def _client_safe_finalization(finalization):
    """The row's `finalization` (review V11, MED-B): `coherence_publish.client_safe_finalization`
    of the session's record -- `detail` and `notice` one line, with no path, traceback frame or
    user name, and at most 700 characters, the phone's own bound. The same transform `/ws`
    applies to `lifecycle.finalization.detail`. A clean record is returned as it is, so every
    row without raw text is byte for byte what it was; the record on disk is never touched.

    Never raises: this row is outside any handler, and one raise empties Saved Worlds. A
    record the transform cannot take is sent with neither text rather than raw."""
    try:
        from tower.world_builder.coherence_publish import (  # noqa: PLC0415
            client_safe_finalization,
        )

        return client_safe_finalization(finalization)
    except Exception:  # noqa: BLE001 -- withheld, never sent raw
        logger.warning(
            "[Tower][Worlds] a finalization record could not be made client-safe; "
            "the row carries it without its detail and notice", exc_info=True,
        )
        if not isinstance(finalization, dict):
            return None
        safe = {k: v for k, v in finalization.items() if k != "notice"}
        safe["detail"] = None
        return safe


def _components_for_row(store: WorldStore, world_id: str, session_id: str, session,
                        world, *, has_geometry: bool, appearance: dict | None):
    """The row's `components` (contract §2), or None -- never an exception.

    None costs one `stat` for every session without a components record, which is
    every session built before the evidence gate: `components: null`, and the row is
    otherwise byte for byte what it was (§7 rule 1). The room entry's
    `has_geometry` and `keyframes_phone` ARE the row's own (§2.1), passed in rather
    than recomputed so the two cannot disagree.

    A record that cannot be turned into the wire array is null too, and logged: the
    phone behaves as today on null, which is the safe failure of a new key.
    """
    try:
        from tower.world_builder.components import (  # noqa: PLC0415
            read_components_record,
            wire_components,
        )
        from tower.world_builder.photographic import (  # noqa: PLC0415
            room_photographic_state,
        )

        record = read_components_record(store, world_id, session_id)
        if record is None:
            return None
        return wire_components(
            store, world_id, session_id, session, record=record,
            room_has_geometry=has_geometry,
            room_keyframes_phone=(appearance or {}).get("keyframes_phone"),
            room_word=room_photographic_state(store, world_id, session_id, session),
            world=world,
        )
    except Exception:  # noqa: BLE001 -- a new key must never take the row down
        logger.warning(
            "[Tower][Worlds] the components of %s/%s could not be listed; the row "
            "says components: null", world_id, session_id, exc_info=True,
        )
        return None


def session_state(session, *, live: bool, has_geometry: bool, manifest=None,
                  still_building: bool = False,
                  photographic: dict | None = None) -> str:
    """One word for what a session IS, from the record, the lock and the tree.

    Mirrors `_lifecycle` in the status producer for the facts a listing
    has (it does not compute geometry currency, so `complete` on a record
    that predates finalization means "stopped and built", not "current").

    `still_building` is the fact the record and the lock cannot hold: a
    photographic stage working on this session RIGHT NOW. The surface,
    appearance and dense stages run after the writer lock is released, so
    `live` -- which is the lock -- is false throughout the six to sixteen
    minutes they take. Passed in as a fact rather than probed here, exactly
    as `live` and `has_geometry` are, so this stays a pure function of what
    it is told and the direct callers in `test_world_builder_finalize_cli`
    keep working unchanged.

    `photographic` is the SETTLED half of the same question, and it is the
    authority -- `still_building` is kept beside it, not replaced by it, for
    exactly the reason `_still_building` in the status producer keeps both:
    "is a process working right now" and "does this world still owe a
    photographic room" are different facts, and each says something the
    other cannot. `still_building` is true in the middle of a stage and
    false in the gaps between them; `is_unsettled(photographic["state"])` is
    true across the whole of it, including the gaps, including a stage that
    was abandoned, and including a probe that broke. Either one means
    `finalizing`.

    Both are `None`/`False`-able so this stays a pure function of its
    arguments; an absent `photographic` means "nobody asked", which is what
    the direct callers in `test_world_builder_finalize_cli` do, and is NOT
    the same as `never_recorded`.

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
    # ABOVE EVERY SETTLED ARM BELOW, and for the reason `_still_building` in
    # the status producer gives at length: those arms all answer "what
    # happened to this session", and none of them can be right while a
    # process is still working on it. `finalizing` is this listing's word for
    # "stopped, and something is still finishing it", which is exactly true
    # here -- the lock arm above says the same thing about the same session a
    # few minutes earlier, when the lock was still held.
    photo_state = (photographic or {}).get("state")
    if still_building or is_unsettled(photo_state):
        # `is_unsettled` BESIDE the present-tense probe, and above every
        # settled arm, exactly where `_still_building` puts the same test in
        # the status producer. `running`, `owed` and `unobservable` are the
        # three words in it and none of them is a world a wearer can be told
        # is finished: one has a process on it now, one is waiting for the
        # Tower's next idle moment to finish it, and one is a question nobody
        # could answer. Before this, all three read `complete` here.
        return SESSION_FINALIZING
    # `failed` DELIBERATELY FALLS THROUGH TO THE SETTLED ARMS, and this is
    # the one judgement call in the change, so it is written down.
    #
    # It is NOT `finalizing`. `finalizing` tells the wearer to wait, and a
    # stage that ran and raised is not going to be fixed by waiting --
    # `world_finish_pending.py` picks up the INTERRUPTED stages at the next
    # Tower start, not the failed ones. Parking a failed build on "Improving"
    # forever is the same lie as "Saved", told in the other direction.
    #
    # It is NOT `interrupted` either, although that was the tempting answer.
    # `interrupted` is contract-defined (WORLD-BUILDER-WORLDS.md §2) as
    # "killed mid-walk, killed mid-finalization, stopped by a request or an
    # error, or a session whose manifest records real figures and whose
    # derived tree is gone" -- all claims about the CAPTURE and the SOLVE.
    # Here both of those succeeded: `finalization.state == complete`,
    # `final_solve == solved`, the derived tree is on disk and the render
    # route serves it. The world IS saved; what failed is the photographic
    # room on top of it. Saying `interrupted` would tell the wearer their
    # walk was lost, which is false and is the more alarming of the two
    # wrong answers.
    #
    # AND IT WOULD BREAK THE AGREEMENT THIS MODULE EXISTS TO KEEP. The
    # status producer's `_still_building` tests `is_unsettled(photo_state)`
    # and `failed` is not in `UNSETTLED_STATES`, so the panel keeps its
    # settled word (`ready`) for a failed photographic build. A row reading
    # `interrupted` over a panel reading `ready` is the picker/panel
    # disagreement that `test_the_picker_and_the_panel_agree_about_every_
    # session` was written to stop.
    #
    # So the row keeps `complete` -- which is a claim about the SESSION --
    # and the failure is told where it is true and where nothing else can
    # say it: the `photographic` block on the row, `{"state": "failed",
    # "stage": ..., "detail": ...}`, which is the same block the status
    # payload carries. The word never claims photographic success, because
    # the word was never about the photographic room; the block is, and it
    # says `failed`.
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
            # ASKED FOR EVERY ROW, unlike the present-tense probe below.
            # It is record-first -- `session.stages` is already in hand --
            # and it only touches the disk for a session that has no stage
            # record at all, where it is two `stat`s beside the journal scan
            # `_keyframes_journaled` already does for this same row. The
            # probe below is a pid lookup and stays gated.
            photographic = _photographic_state_for_row(
                store, world_id, session_id, session
            )
            appearance = _appearance_summary_or_none(store, world_id, session_id, world)
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
                    # The world's lock, only where it speaks for THIS session.
                    live=live and lock_speaks_for(session),
                    has_geometry=has_geometry,
                    manifest=manifest,
                    # ASKED ONLY WHERE IT CAN CHANGE THE ANSWER. `live` and
                    # an open record are decided above it in `session_state`,
                    # so probing for them would cost every row a lock read
                    # and three stat calls to alter nothing. This keeps the
                    # cost to the stopped, unlocked rows -- the window the
                    # photographic stages actually run in.
                    still_building=(
                        not live
                        and session.ended_at is not None
                        and _photographic_build_running(
                            store, world_id, session_id
                        )
                    ),
                    photographic=photographic,
                ),
                # ADDITIVE, and the contract identifier deliberately does
                # not move, for the reason `_dense_summary` gives above:
                # iOS reads these rows key by key out of a
                # `[String: Any]`, so a key it does not know is a key it
                # never looks at, but it equality-tests `contract` and
                # would empty the gallery on a bump.
                #
                # THE SAME BLOCK THE STATUS PAYLOAD CARRIES, so a phone
                # that reads one and then the other reads one fact. The
                # word above cannot hold this: `failed` and `complete`
                # both render as a saved world, and the difference between
                # "the photographic room is ready" and "the photographic
                # build failed" is the whole of T3. Nothing but this block
                # is sending it.
                "photographic": photographic,
                # The builder's own account of how finalization went, or
                # null on a record written before it existed -- CLIENT-SAFE
                # (review V11, MED-B): this listing is unauthenticated, and
                # a writer may have put raw exception text in `detail` (a
                # `C:\Users\<user>\...` path, a traceback), or an old one in
                # `notice`. See `_client_safe_finalization`.
                "finalization": _client_safe_finalization(session.finalization),
                # Additive, and null on every world built before the dense
                # stage existed: what dense reconstruction this session holds.
                "dense": _dense_summary(store, world_id, session_id),
                # Additive (review 1, m6): the appearance artifact, reported as
                # the imagery it is. Null when the session has none.
                "appearance": appearance,
                # Additive (WORLD-BUILDER-COMPONENTS.md §3.1): the pieces of this
                # session's final solve -- the room and the areas the evidence
                # gate could not place -- or null, "not computed", on every
                # session without a components record. The contract identifier
                # does not move, for `_dense_summary`'s reason.
                "components": _components_for_row(
                    store, world_id, session_id, session, world,
                    has_geometry=has_geometry, appearance=appearance),
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
