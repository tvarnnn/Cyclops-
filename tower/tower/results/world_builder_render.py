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
    sessions = set(store.list_session_ids(world_id)) | set(world.session_ids)
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
    candidates.sort(key=lambda pair: (_sortable(pair[0]), pair[1]))
    return candidates[-1][1]


def build_world_render(store: WorldStore, world_id: str, session_id: str | None, *,
                       max_points: int | None = None,
                       view: str | None = None) -> str:
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
