"""The interactive viewer of a saved world, for a phone to open.

Contract: `docs/contracts/WORLD-BUILDER-WORLDS.md` §4 (`GET /worlds/{id}/render`).

The fourth adapter named after this cartridge, and for the same reason
the other three are: it reads World Builder's record shapes, so it is the
one file outside the package that may. It answers a different question
from the geometry producer -- "let a person LOOK at the room" rather than
"give a client the points" -- and answers it with a self-contained HTML
page rather than JSON, which is why it is a separate file and a separate
route rather than a query flag on the manifest.

The page is composed on request from the derived tree, by the same code
`scripts/world_render.py` uses for its `world.html`, so what the phone
shows is what the operator's render shows. Nothing is cached and nothing
is written: a world under construction changes with every build, and a
finished one is cheap to compose (a 438-keyframe walk is well under a
second at the mobile point budget).
"""

from __future__ import annotations

import logging

from tower.results.world_builder_geometry import contained_world_id
from tower.world_builder.render import (
    DEFAULT_MAX_POINTS,
    VIEW_PRODUCT,
    render_html,
)
from tower.results.world_builder_library import _sortable
from tower.world_builder.store import (
    WorldStore,
    WorldStoreError,
    session_has_drawable_geometry,
)

logger = logging.getLogger(__name__)

# A phone draws every point on a 2-D canvas on every gesture, so the
# budget is lower than the operator's 200k default. Fractional-stride
# sampling spans the whole cloud (contract §3), so the picture is the same
# room, thinner. A client may ask for more, up to the operator's default.
#
# 80,000 WAS THE NUMBER UNTIL IT WAS MEASURED. The reasoning above was
# right and the value was a guess. Timing the page's own `draw()` -- the
# function an orbit drag calls once per frame -- on the recovered field
# world in desktop Chrome, 1920x819, by cloning its segments to reach each
# count:
#
#      19,329 pts    6.2 ms   161 fps
#      38,658 pts   13.6 ms    74 fps
#      77,316 pts   35.3 ms    28 fps      <- roughly the old budget
#     115,974 pts   65.0 ms    15 fps
#
# Superlinear, and 28 fps is what a FAST DESKTOP manages at the budget the
# phone was being handed. A phone is slower than that at canvas fill by
# some factor this host cannot measure -- so the old value was not a
# margin, it was the cliff.
#
# 40,000 sits at 13.6 ms / 74 fps here, which is a 2.6x margin on the
# measured cliff and still twice the 19,866 points the 2026-09-09 walk
# produced in total. A 20-30 minute walk is the case this protects: at
# ~3.2 keyframes/sec it reaches several times this and would otherwise be
# handed the whole cloud up to the old ceiling.
#
# UNVERIFIED: the desktop-to-phone factor. The next physical test is the
# first chance to measure it, and this number should be revisited with
# that measurement rather than argued about.
MOBILE_MAX_POINTS = 40_000
MAX_POINTS_CEILING = DEFAULT_MAX_POINTS


class WorldRenderUnavailable(Exception):
    """The route's 404: the world, the session, or its geometry is absent.
    `reason` is what the phone shows; it names what is missing."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _has_geometry(store: WorldStore, world_id: str, session_id: str) -> bool:
    """Whether drawing this session would put anything on the page.

    **Existence of the files was the wrong question here specifically.**
    `resolve_session` uses this to CHOOSE which session to draw, newest
    first -- so on a world walked twice where the second walk solved
    nothing, an empty `points.json` (14 bytes, and `engine.build` writes
    one unconditionally) outranked the older walk that has geometry, and
    the wearer got a blank page for a world with a reconstruction in it.
    Found by `test_the_three_surfaces_ask_the_same_question_of_the_same_files`
    the moment the listing's copy was corrected.
    """
    return session_has_drawable_geometry(store, world_id, session_id)


def _clip(value: str, limit: int = 80) -> str:
    """An id as it appears in a 404 detail: bounded, like `_echo_safe` on
    the result channel, because the detail echoes a request-supplied
    string and the phone shows it verbatim."""
    return value if len(value) <= limit else value[:limit] + "..."


def resolve_session(store: WorldStore, world_id: str, session_id: str | None) -> str:
    """The session to draw. An explicit id is taken as given; otherwise
    the newest session of the world that has geometry -- the listing
    orders sessions oldest first by `started_at`, and this follows it.

    `session_id` is checked against the world's own sessions before any
    path is joined from it, so it cannot name a directory outside the
    world. `world_id` is contained by the caller (`build_world_render`).
    """
    try:
        world = store.read_world(world_id)
    except (WorldStoreError, OSError, ValueError, KeyError):
        raise WorldRenderUnavailable(f"no world {_clip(world_id)!r}") from None
    # The directory scan the listing uses, not `world.json`'s list: the
    # engine writes `session.json` before it appends to `world.json`, and a
    # session the listing offers must be one the render can find.
    # `world.session_ids` COMES OFF DISK UNCOERCED, and this line both
    # hashes it and (below) sorts it. A `world.json` carrying
    # `session_ids: [{"a": 1}]` raises `TypeError: unhashable` right here;
    # one carrying `[123, "s0"]` hashes fine and then raises in the sort
    # three lines down. Either is an HTTP **500 on the route the phone
    # opens a saved world with**.
    #
    # A reviewer found both immediately after the round that hardened
    # `candidates.sort` on line 141 and walked past line 127 -- in a round
    # whose own title was about fixing things one line above the line it
    # fixed.
    #
    # Ids are strings by contract; anything else is a corrupt record and
    # is dropped rather than allowed to take the whole render with it. The
    # directory scan is authoritative anyway, so a dropped `world.json`
    # entry costs nothing that is actually on disk.
    listed = {sid for sid in world.session_ids if isinstance(sid, str)}
    sessions = set(store.list_session_ids(world_id)) | listed
    if session_id is not None:
        if session_id not in sessions:
            raise WorldRenderUnavailable(
                f"world {_clip(world_id)!r} has no session {_clip(session_id)!r}")
        if not _has_geometry(store, world_id, session_id):
            raise WorldRenderUnavailable(
                f"session {session_id!r} of world {world_id!r} has no geometry yet")
        return session_id
    candidates = []
    for candidate in sorted(sessions):
        if not _has_geometry(store, world_id, candidate):
            continue
        try:
            started = store.read_session(world_id, candidate).started_at
        except (WorldStoreError, OSError, ValueError, KeyError):
            continue
        candidates.append((started, candidate))
    if not candidates:
        raise WorldRenderUnavailable(f"world {world_id!r} has no session with geometry yet")
    # `_sortable`, for the reason `build_world_listing` gives at length:
    # `started_at` comes off disk uncoerced, and a string beside a float
    # raises `TypeError` out of a route with no handler. Here that is a
    # 500 on the render page instead of the world.
    # `pair[1]` is a session id and is now guaranteed to be a string --
    # the union above drops anything else -- so the tiebreak cannot raise
    # on a rank tie either.
    candidates.sort(key=lambda pair: (_sortable(pair[0]), pair[1]))
    return candidates[-1][1]


class _ViewerModuleMissing(Exception):
    """Stands in for a viewer's own "unavailable" type when its module did
    not import at all.

    The rungs below deliberately put the import OUTSIDE the `try` that
    catches its exception, because binding the exception name inside that
    `try` meant a broken viewer module left the name unbound and the
    `except` clause raised `NameError` instead of falling back. Binding it
    to `None` instead trades that for `TypeError: catching classes that do
    not inherit from BaseException` -- the same failure with a different
    word. It has to be bound to a real exception class, and one nothing
    raises, so the clause is legal and never matches.
    """


REPRESENTATION_AUTO = "auto"
REPRESENTATION_SPARSE = "sparse"
REPRESENTATION_DENSE = "dense"
REPRESENTATION_SURFACE = "surface"

# The fallback ladder, best first. `auto` walks it and serves the first rung
# this session actually has; a named representation starts the walk at its own
# rung so "give me dense" never silently serves something better or worse
# without saying so. The rung that is reached is reported in the page and in
# the worlds listing, so "what am I looking at" is never a guess.
REPRESENTATION_LADDER = (REPRESENTATION_SURFACE, REPRESENTATION_DENSE,
                         REPRESENTATION_SPARSE)


def build_world_render(store: WorldStore, world_id: str, session_id: str | None, *,
                       max_points: int | None = None,
                       view: str | None = None,
                       representation: str = REPRESENTATION_AUTO) -> str:
    """The viewer page for one session of one world, or
    `WorldRenderUnavailable` naming what is missing.

    `world_id` goes through the geometry adapter's containment guard
    first, for the reason its docstring gives: on Windows a backslash is
    a separator Starlette's route pattern does not exclude, and every
    store path below is joined from this id. An id that escapes the root
    is "no world", and the id is answered under its canonical spelling.
    """
    contained = contained_world_id(store, world_id)
    if contained is None:
        raise WorldRenderUnavailable(f"no world {_clip(world_id)!r}")
    world_id = contained
    chosen = resolve_session(store, world_id, session_id)
    # Walk the ladder from the requested rung down. A session that has a
    # surface gets the surface, because that is the whole point of having
    # built one; one that has only points gets points; every world built
    # before either stage existed falls through to the sparse page unchanged.
    wanted = (representation if representation in REPRESENTATION_LADDER
              else REPRESENTATION_SURFACE)
    start = REPRESENTATION_LADDER.index(wanted)

    if start <= REPRESENTATION_LADDER.index(REPRESENTATION_SURFACE):
        # Same shape as the dense rung below, and for the same reason: the
        # import sits OUTSIDE the try that catches its exception, so a
        # surface module that will not import degrades to the next rung
        # instead of raising a NameError out of the except clause.
        try:
            from tower.world_builder.surface_render import (  # noqa: PLC0415
                SurfaceViewerUnavailable,
                build_surface_page,
            )
        except Exception:  # noqa: BLE001 -- a surface module that will not import
            logger.exception(
                "[Tower][WorldBuilder] the surface viewer module did not import "
                "for %s; falling back", world_id,
            )
            SurfaceViewerUnavailable = _ViewerModuleMissing  # noqa: N806
            build_surface_page = None
        try:
            if build_surface_page is None:
                raise RuntimeError("surface viewer unavailable")
            return build_surface_page(store, world_id, chosen,
                                      max_points=max_points)
        except SurfaceViewerUnavailable as exc:
            if representation == REPRESENTATION_SURFACE:
                raise WorldRenderUnavailable(exc.reason) from None
        except Exception:  # noqa: BLE001 -- never lose the world to a surface bug
            logger.exception(
                "[Tower][WorldBuilder] surface viewer failed for %s; falling back",
                world_id,
            )

    if representation != REPRESENTATION_SPARSE:
        # THE IMPORT IS OUTSIDE THE try THAT CATCHES ITS EXCEPTION.
        #
        # `DenseViewerUnavailable` used to be bound by an import inside the
        # same `try` whose first `except` names it. If that import raised --
        # the one class of bug the fallback below exists for, a broken dense
        # module -- Python evaluated the first except clause, hit a NameError
        # on the unbound name, and propagated THAT. Later clauses of the same
        # try are not tried, so the "never lose the sparse page to a dense bug"
        # fallback never ran and the route returned 500.
        try:
            from tower.world_builder.dense import (  # noqa: PLC0415
                POINT_STRIDE_BYTES,
            )
            from tower.world_builder.dense_render import (  # noqa: PLC0415
                MOBILE_BYTE_BUDGET,
                DenseViewerUnavailable,
                build_dense_page,
            )
        except Exception:  # noqa: BLE001 -- a dense module that will not import
            logger.exception(
                "[Tower][WorldBuilder] the dense viewer module did not import "
                "for %s; serving sparse", world_id,
            )
            DenseViewerUnavailable = _ViewerModuleMissing  # noqa: N806
            build_dense_page = None

        try:
            if build_dense_page is None:
                raise RuntimeError("dense viewer unavailable")

            # `max_points` is validated by the route and must not then be
            # ignored: the worlds contract calls it "point budget", and a
            # client that asks for fewer points has to get fewer. It was
            # dropped on this path, so `max_points=1` returned 295,000 points
            # and a 6 MB page. Converted to the byte budget this viewer speaks,
            # and only ever downwards -- the phone default stays the default.
            budget = MOBILE_BYTE_BUDGET
            if max_points is not None:
                budget = min(budget, max(1, int(max_points)) * POINT_STRIDE_BYTES)
            return build_dense_page(store, world_id, chosen, budget_bytes=budget)
        except DenseViewerUnavailable as exc:
            if representation == REPRESENTATION_DENSE:
                raise WorldRenderUnavailable(exc.reason) from None
        except Exception:  # noqa: BLE001 -- never lose the sparse page to a dense bug
            logger.exception(
                "[Tower][WorldBuilder] dense viewer failed for %s; serving sparse",
                world_id,
            )

    budget = MOBILE_MAX_POINTS if max_points is None else min(max_points, MAX_POINTS_CEILING)
    try:
        return render_html(store, world_id, chosen, max_points=budget,
                           view=view or VIEW_PRODUCT)
    except FileNotFoundError:
        # Raced a `clear_derived` between the existence check and the read.
        # Worded here rather than from the exception: the phone shows the
        # detail verbatim, and an OSError's text carries a filesystem path
        # (contract §3 rule 3: no paths on the wire).
        raise WorldRenderUnavailable(
            f"session {chosen!r} of world {world_id!r} has no geometry yet") from None
